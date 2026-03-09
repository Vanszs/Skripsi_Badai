"""
Comprehensive 6-Scenario Evaluation Script
===========================================
Runs ALL scenarios from FIX_PLAN.md Task 2.1–2.3:
  1. Persistence baseline
  2. MLP Baseline  
  3. Diffusion Only (retrieved=0, graph_emb=0)
  4. Diffusion + Retrieval (graph_emb=0)
  5. Diffusion + GNN (retrieved=0)
  6. Full Model (Retrieval + GNN)

Metrics: RMSE, MAE, Correlation, CRPS, Brier Score, POD, FAR, CSI
Plots:   6 types as specified in FIX_PLAN.md
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.train import temporal_split, compute_stats_from_training
from src.models.mlp_baseline import MLPBaseline
from src.inference import load_model_and_stats, create_inference_graphs
from src.evaluation.probabilistic_metrics import (
    compute_rmse, compute_mae, compute_correlation,
    compute_crps, compute_brier_score,
    compute_pod, compute_far, compute_csi
)

# ============================================================================
# CONFIG
# ============================================================================
EVAL_STEP = 24          # sample every 24 hours (daily)
NUM_ENSEMBLE = 30       # diffusion ensemble samples
SEQ_LEN = 6
TARGET_COLS = ['precipitation', 'wind_speed_10m', 'relative_humidity_2m']
VAR_NAMES = ['precipitation', 'wind_speed', 'humidity']
THRESHOLDS_PRECIP = [2.0, 5.0, 10.0]  # mm thresholds for POD/FAR/CSI

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
plt.rcParams.update({
    'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
    'figure.dpi': 150
})

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# DATA LOADING
# ============================================================================
def load_test_data():
    """Load and prepare test data."""
    data_path = 'data/raw/pangrango_era5_2005_2025.parquet'
    df = pd.read_parquet(data_path)
    
    # Use same feature_cols as training (from checkpoint config)
    ckpt = torch.load('models/diffusion_chkpt.pth', map_location='cpu', weights_only=False)
    feature_cols = ckpt['config'].get('feature_cols', [
        'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
        'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
        'cloud_cover', 'precipitation_lag1', 'elevation'
    ])
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    train_df, val_df, test_df = temporal_split(df, '2018-12-31', '2021-12-31')
    stats = compute_stats_from_training(train_df, feature_cols)
    
    # Prepare test: average across nodes per timestamp
    test_df = test_df.copy()
    test_df['date'] = pd.to_datetime(test_df['date'])
    if test_df['date'].dt.tz is not None:
        test_df['date'] = test_df['date'].dt.tz_localize(None)
    
    grouped = test_df.groupby('date').agg({
        **{col: 'mean' for col in feature_cols if col in test_df.columns},
        **{col: 'mean' for col in TARGET_COLS if col in test_df.columns}
    }).sort_index().reset_index()
    
    # Also prepare per-node data for GNN fix
    nodes = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur']
    per_node_data = {}
    for node in nodes:
        node_df = test_df[test_df['node'] == node].copy().sort_values('date').reset_index(drop=True)
        per_node_data[node] = node_df
    
    return grouped, per_node_data, feature_cols, stats, nodes


def get_per_node_sequence(per_node_data, nodes, idx, seq_len, feature_cols, stats):
    """Get [seq_len, num_nodes, features] normalized tensor for GNN."""
    c_mean = stats['c_mean'].numpy()
    c_std = stats['c_std'].numpy()
    
    sequences = []
    for t in range(idx - seq_len, idx):
        node_feats = []
        for node in nodes:
            ndf = per_node_data[node]
            if t < 0 or t >= len(ndf):
                node_feats.append(np.zeros(len(feature_cols)))
            else:
                row = ndf.iloc[t]
                feat = np.array([row[c] if c in row.index else 0.0 for c in feature_cols], dtype=np.float32)
                node_feats.append(feat)
        sequences.append(np.stack(node_feats))  # [num_nodes, features]
    
    seq = np.stack(sequences)  # [seq_len, num_nodes, features]
    seq_norm = (seq - c_mean) / (c_std + 1e-5)
    return torch.tensor(seq_norm, dtype=torch.float32)


# ============================================================================
# SCENARIO RUNNERS
# ============================================================================
def run_persistence(grouped, feature_cols, stats, eval_step):
    """Scenario 1: Persistence baseline — pred = y[t-1]."""
    targets_all = []
    preds_all = []
    
    for idx in range(SEQ_LEN + 1, len(grouped), eval_step):
        # Target at time t (raw)
        target = np.array([grouped.iloc[idx][c] for c in TARGET_COLS], dtype=np.float32)
        # Persistence: previous timestep
        pred = np.array([grouped.iloc[idx - 1][c] for c in TARGET_COLS], dtype=np.float32)
        targets_all.append(target)
        preds_all.append(pred)
    
    targets = np.stack(targets_all)
    preds = np.stack(preds_all)
    # For persistence, ensemble = just repeat the prediction
    ensemble = np.stack([preds] * NUM_ENSEMBLE, axis=1)  # [N, ensemble, 3]
    return targets, preds, ensemble


def run_mlp_baseline(grouped, feature_cols, stats, eval_step):
    """Scenario 2: MLP Baseline with MC Dropout."""
    ckpt = torch.load('models/mlp_baseline_chkpt.pth', map_location=DEVICE, weights_only=False)
    config = ckpt['config']
    
    model = MLPBaseline(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_targets=config['num_targets']
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state'])
    
    # Normalize features
    c_mean = stats['c_mean'].numpy()[:len(feature_cols)]
    c_std = stats['c_std'].numpy()[:len(feature_cols)]
    t_mean = stats['t_mean'].numpy()
    t_std = stats['t_std'].numpy()
    
    features_raw = grouped[feature_cols].values.astype(np.float32)
    features_norm = (features_raw - c_mean) / (c_std + 1e-5)
    
    targets_raw = grouped[TARGET_COLS].values.astype(np.float32)
    
    targets_all = []
    preds_all = []
    ensemble_all = []
    
    for idx in range(SEQ_LEN, len(grouped), eval_step):
        x = features_norm[idx - SEQ_LEN:idx].flatten()
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        
        target = targets_raw[idx]
        
        # MC Dropout: 15 forward passes with dropout active
        model.train()  # enable dropout
        mc_preds_norm = []
        with torch.no_grad():
            for _ in range(NUM_ENSEMBLE):
                pred_norm = model(x_t).cpu().numpy()[0]
                mc_preds_norm.append(pred_norm)
        
        mc_preds_norm = np.stack(mc_preds_norm)  # [ensemble, 3]
        
        # Denormalize
        mc_preds_denorm = mc_preds_norm * t_std + t_mean
        mc_preds_denorm[:, 0] = np.expm1(mc_preds_denorm[:, 0])
        mc_preds_denorm[:, 0] = np.clip(mc_preds_denorm[:, 0], 0, None)
        mc_preds_denorm[:, 2] = np.clip(mc_preds_denorm[:, 2], 0, 100)
        
        median_pred = np.median(mc_preds_denorm, axis=0)
        
        targets_all.append(target)
        preds_all.append(median_pred)
        ensemble_all.append(mc_preds_denorm)
    
    model.eval()
    return np.stack(targets_all), np.stack(preds_all), np.stack(ensemble_all)


def run_diffusion_scenario(grouped, per_node_data, nodes, feature_cols, stats,
                           eval_step, use_retrieval=True, use_gnn=True):
    """Run diffusion model with ablation control."""
    model_wrapper, _, retrieval_db = load_model_and_stats('models/diffusion_chkpt.pth')
    model_wrapper.to(DEVICE)
    model_wrapper.eval()
    
    config = model_wrapper.config
    st_gnn = model_wrapper.st_gnn
    forecaster = model_wrapper.forecaster
    
    c_mean = stats['c_mean'].numpy()
    c_std = stats['c_std'].numpy()
    t_mean_t = stats['t_mean'].to(DEVICE)
    t_std_t = stats['t_std'].to(DEVICE)
    
    features_raw = grouped[feature_cols].values.astype(np.float32)
    features_norm = (features_raw - c_mean) / (c_std + 1e-5)
    targets_raw = grouped[TARGET_COLS].values.astype(np.float32)
    
    targets_all = []
    preds_all = []
    ensemble_all = []
    
    for idx in tqdm(range(SEQ_LEN, len(grouped), eval_step), desc=f"Diff(R={use_retrieval},G={use_gnn})"):
        target = targets_raw[idx]
        
        # Context sequence (averaged — for condition + retrieval)
        ctx_seq = torch.tensor(features_norm[idx - SEQ_LEN:idx], dtype=torch.float32).to(DEVICE)
        context_last = ctx_seq[-1].unsqueeze(0)  # [1, feat]
        
        with torch.no_grad():
            # GNN embedding
            if use_gnn:
                per_node_seq = get_per_node_sequence(per_node_data, nodes, idx, SEQ_LEN, feature_cols, stats)
                per_node_seq = per_node_seq.to(DEVICE)
                graphs = create_inference_graphs(per_node_seq, config, num_nodes=3, device=DEVICE)
                graph_emb = st_gnn(graphs)  # [1, graph_dim]
            else:
                graph_emb = torch.zeros(1, config['graph_dim'], device=DEVICE)
            
            # Retrieval
            if use_retrieval:
                ctx_np = context_last.cpu().numpy()
                retrieved = retrieval_db.query(ctx_np, k=config['k_neighbors'])
                retrieved = retrieved.to(DEVICE)
            else:
                k = config.get('k_neighbors', 3)
                feat_dim = config['retrieval_dim'] // k
                retrieved = torch.zeros(1, k, feat_dim, device=DEVICE)
            
            # Sample
            samples = forecaster.sample_fast(
                condition=context_last,
                retrieved=retrieved,
                graph_emb=graph_emb,
                num_samples=NUM_ENSEMBLE,
                num_inference_steps=20
            )
            
            # Denormalize
            samples_denorm = samples * t_std_t + t_mean_t
            samples_denorm[:, 0] = torch.expm1(samples_denorm[:, 0])
            samples_denorm[:, 0] = torch.clamp(samples_denorm[:, 0], min=0.0)
            samples_denorm[:, 2] = torch.clamp(samples_denorm[:, 2], min=0.0, max=100.0)
            
            samples_np = samples_denorm.cpu().numpy()  # [ensemble, 3]
        
        median_pred = np.median(samples_np, axis=0)
        targets_all.append(target)
        preds_all.append(median_pred)
        ensemble_all.append(samples_np)
    
    return np.stack(targets_all), np.stack(preds_all), np.stack(ensemble_all)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================
def compute_scenario_metrics(targets, preds, ensemble):
    """Compute all metrics for one scenario."""
    results = {}
    
    for i, var in enumerate(VAR_NAMES):
        act = targets[:, i]
        pred = preds[:, i]
        ens = ensemble[:, :, i]  # [N, ensemble]
        
        m = {
            'rmse': compute_rmse(pred, act),
            'mae': compute_mae(pred, act),
            'correlation': compute_correlation(pred, act),
            'crps': compute_crps(ens, act),
        }
        
        # Threshold metrics for precipitation only
        if var == 'precipitation':
            for thr in THRESHOLDS_PRECIP:
                m[f'brier_{int(thr)}mm'] = compute_brier_score(ens, act, threshold=thr)
                m[f'pod_{int(thr)}mm'] = compute_pod(ens, act, threshold=thr)
                m[f'far_{int(thr)}mm'] = compute_far(ens, act, threshold=thr)
                m[f'csi_{int(thr)}mm'] = compute_csi(ens, act, threshold=thr)
        
        results[var] = m
    
    return results


# ============================================================================
# PLOTTING
# ============================================================================
def plot_bar_chart(all_results, save_dir):
    """Plot 1: Bar chart of RMSE/MAE/Corr × 3 variables × 6 scenarios."""
    scenarios = list(all_results.keys())
    metrics_to_plot = ['rmse', 'mae', 'correlation']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    for vi, var in enumerate(VAR_NAMES):
        for mi, metric in enumerate(metrics_to_plot):
            ax = axes[vi][mi]
            vals = []
            for s in scenarios:
                v = all_results[s].get(var, {}).get(metric, 0)
                vals.append(v if not np.isnan(v) else 0)
            
            bars = ax.bar(range(len(scenarios)), vals, color=PALETTE[:len(scenarios)])
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8, rotation=45, ha='right')
            ax.set_title(f'{var} — {metric.upper()}')
            ax.set_ylabel(metric.upper())
            
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                       f'{v:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'bar_chart_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_scatter(all_data, save_dir):
    """Plot 2: Scatter actual vs pred for best model (Full)."""
    if 'full_model' not in all_data:
        return
    targets, preds, ensemble = all_data['full_model']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, var in enumerate(VAR_NAMES):
        ax = axes[i]
        act = targets[:, i]
        pred = preds[:, i]
        spread = ensemble[:, :, i].std(axis=1)
        
        sc = ax.scatter(act, pred, c=spread, cmap='YlOrRd', alpha=0.6, s=15, edgecolors='none')
        mn, mx = min(act.min(), pred.min()), max(act.max(), pred.max())
        ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, lw=1)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{var} (Full Model)')
        plt.colorbar(sc, ax=ax, label='Spread (σ)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'scatter_actual_vs_pred.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_time_series(all_data, save_dir, n_points=720):
    """Plot 3: Time series sample — 1 month with P10–P90 band."""
    if 'full_model' not in all_data:
        return
    targets, preds, ensemble = all_data['full_model']
    
    n = min(n_points, len(targets))
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    
    for i, var in enumerate(VAR_NAMES):
        ax = axes[i]
        act = targets[:n, i]
        med = preds[:n, i]
        p10 = np.percentile(ensemble[:n, :, i], 10, axis=1)
        p90 = np.percentile(ensemble[:n, :, i], 90, axis=1)
        
        x = np.arange(n)
        ax.fill_between(x, p10, p90, alpha=0.3, color=PALETTE[0], label='P10–P90')
        ax.plot(x, med, color=PALETTE[0], lw=1, label='Median pred')
        ax.plot(x, act, color=PALETTE[3], lw=1, alpha=0.8, label='Actual')
        ax.set_ylabel(var)
        ax.legend(loc='upper right', fontsize=9)
    
    axes[-1].set_xlabel('Time step (hourly × eval_step)')
    plt.suptitle('Time Series: Actual vs Predicted (Full Model)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'time_series_sample.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_reliability(all_data, save_dir):
    """Plot 4: Reliability diagram for precipitation."""
    if 'full_model' not in all_data:
        return
    targets, _, ensemble = all_data['full_model']
    
    threshold = 2.0  # 2mm
    obs_binary = (targets[:, 0] > threshold).astype(float)
    forecast_prob = np.mean(ensemble[:, :, 0] > threshold, axis=1)
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    
    obs_freq = []
    pred_freq = []
    counts = []
    
    for b in range(n_bins):
        mask = (forecast_prob >= bin_edges[b]) & (forecast_prob < bin_edges[b + 1])
        if mask.sum() > 0:
            obs_freq.append(obs_binary[mask].mean())
            pred_freq.append(forecast_prob[mask].mean())
            counts.append(mask.sum())
        else:
            obs_freq.append(np.nan)
            pred_freq.append((bin_edges[b] + bin_edges[b + 1]) / 2)
            counts.append(0)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect reliability')
    ax.plot(pred_freq, obs_freq, 'o-', color=PALETTE[0], markersize=8, label=f'Precip > {threshold}mm')
    ax.set_xlabel('Forecast probability')
    ax.set_ylabel('Observed frequency')
    ax.set_title('Reliability Diagram — Precipitation')
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reliability_diagram.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_crps_comparison(all_results, save_dir):
    """Plot 5: CRPS comparison across scenarios."""
    scenarios = list(all_results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, var in enumerate(VAR_NAMES):
        ax = axes[i]
        crps_vals = [all_results[s].get(var, {}).get('crps', 0) for s in scenarios]
        bars = ax.barh(range(len(scenarios)), crps_vals, color=PALETTE[:len(scenarios)])
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([s.replace('_', '\n') for s in scenarios], fontsize=9)
        ax.set_xlabel('CRPS (lower = better)')
        ax.set_title(f'{var}')
        
        for bar, v in zip(bars, crps_vals):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
                   f' {v:.4f}', va='center', fontsize=8)
    
    plt.suptitle('CRPS Comparison Across Scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'crps_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation(all_results, save_dir):
    """Plot 6: Ablation contribution — Corr progression."""
    ablation_order = ['diff_only', 'diff_retrieval', 'diff_gnn', 'full_model']
    labels = ['Diff Only', '+Retrieval', '+GNN', 'Full']
    
    available = [s for s in ablation_order if s in all_results]
    if len(available) < 2:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, var in enumerate(VAR_NAMES):
        ax = axes[i]
        corrs = [all_results[s].get(var, {}).get('correlation', 0)
                 for s in available]
        lbls = [labels[ablation_order.index(s)] for s in available]
        
        ax.plot(range(len(corrs)), corrs, 'o-', color=PALETTE[0], markersize=10, lw=2)
        ax.set_xticks(range(len(corrs)))
        ax.set_xticklabels(lbls)
        ax.set_ylabel('Correlation')
        ax.set_title(f'{var}')
        ax.set_ylim(-0.1, 1.05)
        
        for xi, v in enumerate(corrs):
            ax.text(xi, v + 0.03, f'{v:.3f}', ha='center', fontsize=10)
    
    plt.suptitle('Ablation: Component Contribution to Correlation', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ablation_contribution.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 70)
    print("COMPREHENSIVE 6-SCENARIO EVALUATION")
    print("=" * 70)
    
    os.makedirs('result_test/plots', exist_ok=True)
    os.makedirs('result_test/comparison', exist_ok=True)
    
    # Load data
    print("\n[1/8] Loading test data...")
    grouped, per_node_data, feature_cols, stats, nodes = load_test_data()
    print(f"  Test samples: {len(grouped)}, eval_step={EVAL_STEP}")
    print(f"  Eval points: ~{len(grouped) // EVAL_STEP}")
    
    all_results = {}
    all_data = {}
    
    # --- Scenario 1: Persistence ---
    print("\n[2/8] Scenario 1: Persistence...")
    t, p, e = run_persistence(grouped, feature_cols, stats, EVAL_STEP)
    all_results['persistence'] = compute_scenario_metrics(t, p, e)
    all_data['persistence'] = (t, p, e)
    print(f"  Precip Corr: {all_results['persistence']['precipitation']['correlation']:.4f}")
    
    # --- Scenario 2: MLP Baseline ---
    print("\n[3/8] Scenario 2: MLP Baseline (MC Dropout)...")
    t, p, e = run_mlp_baseline(grouped, feature_cols, stats, EVAL_STEP)
    all_results['mlp_baseline'] = compute_scenario_metrics(t, p, e)
    all_data['mlp_baseline'] = (t, p, e)
    print(f"  Precip Corr: {all_results['mlp_baseline']['precipitation']['correlation']:.4f}")
    
    # --- Scenario 3: Diffusion Only ---
    print("\n[4/8] Scenario 3: Diffusion Only (no retrieval, no GNN)...")
    t, p, e = run_diffusion_scenario(grouped, per_node_data, nodes, feature_cols, stats,
                                      EVAL_STEP, use_retrieval=False, use_gnn=False)
    all_results['diff_only'] = compute_scenario_metrics(t, p, e)
    all_data['diff_only'] = (t, p, e)
    print(f"  Precip Corr: {all_results['diff_only']['precipitation']['correlation']:.4f}")
    
    # --- Scenario 4: Diffusion + Retrieval ---
    print("\n[5/8] Scenario 4: Diffusion + Retrieval (no GNN)...")
    t, p, e = run_diffusion_scenario(grouped, per_node_data, nodes, feature_cols, stats,
                                      EVAL_STEP, use_retrieval=True, use_gnn=False)
    all_results['diff_retrieval'] = compute_scenario_metrics(t, p, e)
    all_data['diff_retrieval'] = (t, p, e)
    print(f"  Precip Corr: {all_results['diff_retrieval']['precipitation']['correlation']:.4f}")
    
    # --- Scenario 5: Diffusion + GNN ---
    print("\n[6/8] Scenario 5: Diffusion + GNN (no retrieval)...")
    t, p, e = run_diffusion_scenario(grouped, per_node_data, nodes, feature_cols, stats,
                                      EVAL_STEP, use_retrieval=False, use_gnn=True)
    all_results['diff_gnn'] = compute_scenario_metrics(t, p, e)
    all_data['diff_gnn'] = (t, p, e)
    print(f"  Precip Corr: {all_results['diff_gnn']['precipitation']['correlation']:.4f}")
    
    # --- Scenario 6: Full Model ---
    print("\n[7/8] Scenario 6: Full Model (Retrieval + GNN)...")
    t, p, e = run_diffusion_scenario(grouped, per_node_data, nodes, feature_cols, stats,
                                      EVAL_STEP, use_retrieval=True, use_gnn=True)
    all_results['full_model'] = compute_scenario_metrics(t, p, e)
    all_data['full_model'] = (t, p, e)
    print(f"  Precip Corr: {all_results['full_model']['precipitation']['correlation']:.4f}")
    
    # --- Save per-scenario metrics ---
    for scenario_name, metrics in all_results.items():
        scenario_dir = f'result_test/{scenario_name.replace(" ", "_")}'
        os.makedirs(scenario_dir, exist_ok=True)
        with open(os.path.join(scenario_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    
    # --- Comparison Summary ---
    print("\n[8/8] Generating plots and summary...")
    
    # Build summary table
    rows = []
    for scenario in all_results:
        for var in VAR_NAMES:
            m = all_results[scenario].get(var, {})
            rows.append({
                'scenario': scenario,
                'variable': var,
                'rmse': m.get('rmse', np.nan),
                'mae': m.get('mae', np.nan),
                'correlation': m.get('correlation', np.nan),
                'crps': m.get('crps', np.nan),
            })
    
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv('result_test/comparison/comparison_summary.csv', index=False)
    
    with open('result_test/comparison/comparison_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)
    
    # --- Plots ---
    plot_dir = 'result_test/plots'
    os.makedirs(plot_dir, exist_ok=True)
    
    plot_bar_chart(all_results, plot_dir)
    plot_scatter(all_data, plot_dir)
    plot_time_series(all_data, plot_dir)
    plot_reliability(all_data, plot_dir)
    plot_crps_comparison(all_results, plot_dir)
    plot_ablation(all_results, plot_dir)
    
    # --- Print Summary Table ---
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    for var in VAR_NAMES:
        print(f"\n{'─' * 80}")
        print(f"  {var.upper()}")
        print(f"{'─' * 80}")
        print(f"  {'Scenario':<20} {'RMSE':<10} {'MAE':<10} {'Corr':<10} {'CRPS':<10}")
        print(f"  {'─' * 60}")
        for scenario in all_results:
            m = all_results[scenario].get(var, {})
            print(f"  {scenario:<20} {m.get('rmse',0):<10.4f} {m.get('mae',0):<10.4f} "
                  f"{m.get('correlation',0):<10.4f} {m.get('crps',0):<10.4f}")
    
    # Threshold metrics for precipitation
    print(f"\n{'─' * 80}")
    print(f"  PRECIPITATION THRESHOLD METRICS")
    print(f"{'─' * 80}")
    for thr in THRESHOLDS_PRECIP:
        print(f"\n  Threshold: {thr}mm")
        print(f"  {'Scenario':<20} {'POD':<10} {'FAR':<10} {'CSI':<10} {'Brier':<10}")
        print(f"  {'─' * 50}")
        for scenario in all_results:
            m = all_results[scenario].get('precipitation', {})
            pod = m.get(f'pod_{int(thr)}mm', float('nan'))
            far = m.get(f'far_{int(thr)}mm', float('nan'))
            csi = m.get(f'csi_{int(thr)}mm', float('nan'))
            brier = m.get(f'brier_{int(thr)}mm', float('nan'))
            print(f"  {scenario:<20} {pod:<10.4f} {far:<10.4f} {csi:<10.4f} {brier:<10.4f}")
    
    # --- Write Evaluation Report ---
    with open('result_test/EVALUATION_REPORT.md', 'w', encoding='utf-8') as f:
        f.write("# Evaluation Report — 6-Scenario Comprehensive\n\n")
        f.write(f"**Date**: 2026-03-09\n")
        f.write(f"**Eval Step**: {EVAL_STEP} (daily sampling)\n")
        f.write(f"**Ensemble Size**: {NUM_ENSEMBLE}\n")
        f.write(f"**Test Period**: 2022–2025\n\n")
        
        for var in VAR_NAMES:
            f.write(f"## {var.upper()}\n\n")
            f.write(f"| Scenario | RMSE | MAE | Corr | CRPS |\n")
            f.write(f"|----------|------|-----|------|------|\n")
            for scenario in all_results:
                m = all_results[scenario].get(var, {})
                f.write(f"| {scenario} | {m.get('rmse',0):.4f} | {m.get('mae',0):.4f} | "
                       f"{m.get('correlation',0):.4f} | {m.get('crps',0):.4f} |\n")
            f.write("\n")
        
        f.write("## Precipitation Threshold Metrics\n\n")
        for thr in THRESHOLDS_PRECIP:
            f.write(f"### Threshold: {thr}mm\n\n")
            f.write(f"| Scenario | POD | FAR | CSI | Brier |\n")
            f.write(f"|----------|-----|-----|-----|-------|\n")
            for scenario in all_results:
                m = all_results[scenario].get('precipitation', {})
                pod = m.get(f'pod_{int(thr)}mm', float('nan'))
                far = m.get(f'far_{int(thr)}mm', float('nan'))
                csi = m.get(f'csi_{int(thr)}mm', float('nan'))
                brier = m.get(f'brier_{int(thr)}mm', float('nan'))
                f.write(f"| {scenario} | {pod:.4f} | {far:.4f} | {csi:.4f} | {brier:.4f} |\n")
            f.write("\n")
    
    print(f"\n✓ All results saved to result_test/")
    print(f"✓ Plots saved to result_test/plots/")
    print(f"✓ Report: result_test/EVALUATION_REPORT.md")


if __name__ == '__main__':
    main()
