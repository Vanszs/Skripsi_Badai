"""
4-SCENARIO HONEST EVALUATION
=============================
Membandingkan secara fair 4 pendekatan prediksi:

  1. Pure Persistence  : prediksi = nilai jam sebelumnya
  2. MLP Baseline      : MLP 3-layer (per-node, apple-to-apple)
  3. Diffusion (Pure)  : model diffusion TANPA hybrid
  4. Diffusion+Hybrid  : model diffusion + persistence blending (w=0.9/0.9/0.7)

Semua dievaluasi pada test set 2022-2025, per-node, EVAL_STEP=24h.
Hasil disimpan ke result_test/
"""

import sys
sys.path.append('.')

import os
import time
import torch
import pandas as pd
import numpy as np
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.inference import load_model_and_stats, run_inference_real, run_inference_hybrid
from src.evaluation.probabilistic_metrics import compute_all_metrics

warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EVAL_STEP = 24
NUM_ENSEMBLE = 30
NODE_NAMES = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur']
VARIABLES = ['precipitation', 'wind_speed', 'humidity']
VAR_THRESHOLDS = {
    'precipitation': 10.0,
    'wind_speed': 10.0,
    'humidity': 90.0
}
OUTPUT_DIR = 'result_test'

SCENARIOS = [
    'persistence',        # 1. Pure persistence
    'mlp_baseline',       # 2. MLP baseline per-node
    'diffusion_pure',     # 3. Diffusion tanpa hybrid
    'diffusion_hybrid',   # 4. Diffusion + hybrid (0.9/0.9/0.7)
]

def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        return None if np.isnan(val) else val
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj

# ==============================================================================
# LOAD DATA & MODELS
# ==============================================================================
print("=" * 80)
print("4-SCENARIO HONEST EVALUATION")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Load diffusion model
print("\n[1/4] Loading diffusion model...")
model, stats, retrieval_db = load_model_and_stats('models/diffusion_chkpt.pth')
model.to(device)
model.eval()

seq_len = model.config['seq_len']
feature_cols = model.config.get('feature_cols', [])
target_cols = model.config.get('target_cols', [])
c_mean = stats['c_mean'].numpy()
c_std = stats['c_std'].numpy()
t_mean = stats['t_mean'].numpy()
t_std = stats['t_std'].numpy()

# Load MLP baseline
print("[2/4] Loading MLP baseline...")
mlp_checkpoint = torch.load('models/mlp_baseline_chkpt.pth', map_location=device)
from src.models.mlp_baseline import MLPBaseline
mlp_config = mlp_checkpoint['config']
mlp_model = MLPBaseline(
    input_dim=mlp_config['input_dim'],
    hidden_dim=mlp_config['hidden_dim'],
    num_targets=mlp_config['num_targets']
).to(device)
mlp_model.load_state_dict(mlp_checkpoint['model_state'])
mlp_model.eval()
mlp_feature_cols = mlp_config['feature_cols']
mlp_stats = mlp_checkpoint['stats']
mlp_seq_len = mlp_config['seq_len']

# Load data
print("[3/4] Loading test data...")
df = pd.read_parquet('data/raw/pangrango_era5_2005_2025.parquet')
df['date'] = pd.to_datetime(df['date'])
if df['date'].dt.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)
test_df = df[df['date'] >= '2022-01-01'].copy()

available_cols = [c for c in feature_cols if c in test_df.columns]
mlp_available_cols = [c for c in mlp_feature_cols if c in test_df.columns]

print(f"   Test rows: {len(test_df):,}")
print(f"   Date range: {test_df['date'].min()} to {test_df['date'].max()}")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sc in SCENARIOS:
    os.makedirs(f"{OUTPUT_DIR}/{sc}", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/plots", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/comparison", exist_ok=True)

# ==============================================================================
# EVALUATION LOOP
# ==============================================================================
print("\n[4/4] Running evaluation...")

# Storage for all scenarios
all_scenario_results = {sc: {} for sc in SCENARIOS}       # per-node metrics
all_scenario_aggregated = {sc: {} for sc in SCENARIOS}    # aggregated metrics
all_scenario_data = {sc: {} for sc in SCENARIOS}          # raw predictions

for node_name in NODE_NAMES:
    print(f"\n{'='*60}")
    print(f"  Node: {node_name}")
    print(f"{'='*60}")

    node_df = test_df[test_df['node'] == node_name].copy()
    node_df = node_df.sort_values('date').reset_index(drop=True)

    if len(node_df) < seq_len + 1:
        print(f"  SKIP: insufficient data")
        continue

    n_hours = len(node_df) - seq_len - 1
    eval_indices = list(range(0, n_hours, EVAL_STEP))
    n_eval = len(eval_indices)
    print(f"  Eval samples: {n_eval}")

    # Containers per scenario
    actuals = {v: [] for v in VARIABLES}
    preds = {sc: {v: [] for v in VARIABLES} for sc in SCENARIOS}
    ensembles = {sc: {v: [] for v in VARIABLES} for sc in ['diffusion_pure', 'diffusion_hybrid']}
    nan_count = 0

    for idx in tqdm(eval_indices, desc=f"  {node_name}", ncols=80):
        seq_end = idx + seq_len
        if seq_end >= len(node_df):
            break

        seq_df = node_df.iloc[idx:seq_end]
        target_row = node_df.iloc[seq_end]
        prev_row = node_df.iloc[seq_end - 1]

        # Actual values
        act_precip = target_row['precipitation']
        act_wind = target_row['wind_speed_10m']
        act_humid = target_row['relative_humidity_2m']

        # Previous hour values (for persistence)
        lag_precip = prev_row['precipitation']
        lag_wind = prev_row['wind_speed_10m']
        lag_humid = prev_row['relative_humidity_2m']

        lag_values = {
            'precipitation': lag_precip,
            'wind_speed': lag_wind,
            'humidity': lag_humid
        }

        # ----- Scenario 1: Pure Persistence -----
        pers_precip = lag_precip
        pers_wind = lag_wind
        pers_humid = lag_humid

        # ----- Scenario 2: MLP Baseline (per-node) -----
        mlp_features = seq_df[mlp_available_cols].values.astype(np.float32)
        mlp_c_mean = mlp_stats['c_mean'].cpu().numpy()[:len(mlp_available_cols)]
        mlp_c_std = mlp_stats['c_std'].cpu().numpy()[:len(mlp_available_cols)]
        mlp_features_norm = (mlp_features - mlp_c_mean) / (mlp_c_std + 1e-5)
        mlp_input = torch.tensor(mlp_features_norm.flatten(), dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            mlp_out_norm = mlp_model(mlp_input).cpu().numpy()[0]  # [3]

        # Denormalize MLP
        mlp_out_denorm = mlp_out_norm * t_std + t_mean
        mlp_out_denorm[0] = np.expm1(mlp_out_denorm[0])  # precipitation
        mlp_out_denorm[0] = max(mlp_out_denorm[0], 0)
        mlp_out_denorm[2] = np.clip(mlp_out_denorm[2], 0, 100)

        mlp_precip = mlp_out_denorm[0]
        mlp_wind = mlp_out_denorm[1]
        mlp_humid = mlp_out_denorm[2]

        # ----- Scenarios 3 & 4: Diffusion -----
        features = seq_df[available_cols].values
        features_norm = (features - c_mean[:len(available_cols)]) / (c_std[:len(available_cols)] + 1e-5)

        try:
            # Pure diffusion (no hybrid)
            result_pure = run_inference_real(
                features_norm, model, stats, retrieval_db,
                num_samples=NUM_ENSEMBLE, device=device
            )

            # Hybrid diffusion
            result_hybrid = run_inference_hybrid(
                features_norm, model, stats, retrieval_db,
                lag_values=lag_values, num_samples=NUM_ENSEMBLE, device=device
            )
        except Exception as e:
            nan_count += 1
            continue

        # Check NaN
        ens_pure = {
            'precipitation': result_pure['precipitation'],
            'wind_speed': result_pure['wind_speed'],
            'humidity': result_pure['humidity']
        }
        ens_hybrid = {
            'precipitation': result_hybrid.get('hybrid_precipitation', result_hybrid['precipitation']),
            'wind_speed': result_hybrid.get('hybrid_wind_speed', result_hybrid['wind_speed']),
            'humidity': result_hybrid.get('hybrid_humidity', result_hybrid['humidity'])
        }

        skip = False
        for v in ['precipitation', 'wind_speed', 'humidity']:
            if np.all(np.isnan(ens_pure[v])) or np.all(np.isnan(ens_hybrid[v])):
                skip = True
                break
        if skip:
            nan_count += 1
            continue

        # Replace NaN in ensembles with median
        for ens_dict in [ens_pure, ens_hybrid]:
            for v in ['precipitation', 'wind_speed', 'humidity']:
                arr = ens_dict[v]
                nans = np.isnan(arr)
                if nans.any() and not nans.all():
                    arr[nans] = np.nanmedian(arr)

        # Store actuals
        actuals['precipitation'].append(act_precip)
        actuals['wind_speed'].append(act_wind)
        actuals['humidity'].append(act_humid)

        # Store predictions
        preds['persistence']['precipitation'].append(pers_precip)
        preds['persistence']['wind_speed'].append(pers_wind)
        preds['persistence']['humidity'].append(pers_humid)

        preds['mlp_baseline']['precipitation'].append(mlp_precip)
        preds['mlp_baseline']['wind_speed'].append(mlp_wind)
        preds['mlp_baseline']['humidity'].append(mlp_humid)

        preds['diffusion_pure']['precipitation'].append(float(np.nanmedian(ens_pure['precipitation'])))
        preds['diffusion_pure']['wind_speed'].append(float(np.nanmedian(ens_pure['wind_speed'])))
        preds['diffusion_pure']['humidity'].append(float(np.nanmedian(ens_pure['humidity'])))

        preds['diffusion_hybrid']['precipitation'].append(float(np.nanmedian(ens_hybrid['precipitation'])))
        preds['diffusion_hybrid']['wind_speed'].append(float(np.nanmedian(ens_hybrid['wind_speed'])))
        preds['diffusion_hybrid']['humidity'].append(float(np.nanmedian(ens_hybrid['humidity'])))

        # Store ensembles
        ensembles['diffusion_pure']['precipitation'].append(ens_pure['precipitation'].copy())
        ensembles['diffusion_pure']['wind_speed'].append(ens_pure['wind_speed'].copy())
        ensembles['diffusion_pure']['humidity'].append(ens_pure['humidity'].copy())

        ensembles['diffusion_hybrid']['precipitation'].append(ens_hybrid['precipitation'].copy())
        ensembles['diffusion_hybrid']['wind_speed'].append(ens_hybrid['wind_speed'].copy())
        ensembles['diffusion_hybrid']['humidity'].append(ens_hybrid['humidity'].copy())

    # Convert to arrays
    valid_samples = len(actuals['precipitation'])
    print(f"  Valid: {valid_samples}/{n_eval} (skipped {nan_count})")

    if valid_samples == 0:
        continue

    for v in VARIABLES:
        actuals[v] = np.array(actuals[v], dtype=np.float64)
        for sc in SCENARIOS:
            preds[sc][v] = np.array(preds[sc][v], dtype=np.float64)
        for sc in ['diffusion_pure', 'diffusion_hybrid']:
            ensembles[sc][v] = np.array(ensembles[sc][v], dtype=np.float64)

    # ----- Compute metrics per scenario -----
    for sc in SCENARIOS:
        node_metrics = {}
        for var in VARIABLES:
            if sc in ['diffusion_pure', 'diffusion_hybrid']:
                # Full probabilistic metrics using ensemble
                metrics = compute_all_metrics(
                    ensemble_samples=ensembles[sc][var],
                    observations=actuals[var],
                    deterministic_predictions=preds[sc][var],
                    heavy_rain_threshold=VAR_THRESHOLDS[var],
                    prob_threshold=0.5
                )
            else:
                # Deterministic only (persistence, MLP) — fake 1-member ensemble
                fake_ens = preds[sc][var].reshape(-1, 1)
                metrics = compute_all_metrics(
                    ensemble_samples=fake_ens,
                    observations=actuals[var],
                    deterministic_predictions=preds[sc][var],
                    heavy_rain_threshold=VAR_THRESHOLDS[var],
                    prob_threshold=0.5
                )
            node_metrics[var] = metrics

        all_scenario_results[sc][node_name] = node_metrics

    # Store raw data for aggregation
    for sc in SCENARIOS:
        all_scenario_data[sc][node_name] = {
            'actuals': {v: actuals[v].copy() for v in VARIABLES},
            'preds': {v: preds[sc][v].copy() for v in VARIABLES},
        }
        if sc in ['diffusion_pure', 'diffusion_hybrid']:
            all_scenario_data[sc][node_name]['ensembles'] = {
                v: ensembles[sc][v].copy() for v in VARIABLES
            }

    # Print comparison table for this node
    print(f"\n  {'Scenario':<20} {'Var':<15} {'RMSE':<8} {'MAE':<8} {'Corr':<8}")
    print(f"  {'-'*59}")
    for sc in SCENARIOS:
        for var in VARIABLES:
            m = all_scenario_results[sc][node_name][var]
            corr = m['correlation']
            corr_s = f"{corr:.4f}" if corr is not None and not np.isnan(corr) else "N/A"
            print(f"  {sc:<20} {var:<15} {m['rmse']:.4f}  {m['mae']:.4f}  {corr_s}")
        print()

# ==============================================================================
# AGGREGATE ACROSS ALL NODES
# ==============================================================================
print(f"\n{'='*80}")
print("AGGREGATED RESULTS (All Nodes)")
print(f"{'='*80}")

for sc in SCENARIOS:
    agg = {}
    for var in VARIABLES:
        nodes_with_data = [n for n in all_scenario_data[sc] if len(all_scenario_data[sc][n]['actuals'][var]) > 0]
        if not nodes_with_data:
            continue

        all_act = np.concatenate([all_scenario_data[sc][n]['actuals'][var] for n in nodes_with_data])
        all_pred = np.concatenate([all_scenario_data[sc][n]['preds'][var] for n in nodes_with_data])

        if sc in ['diffusion_pure', 'diffusion_hybrid']:
            all_ens = np.concatenate([all_scenario_data[sc][n]['ensembles'][var] for n in nodes_with_data])
            metrics = compute_all_metrics(
                ensemble_samples=all_ens,
                observations=all_act,
                deterministic_predictions=all_pred,
                heavy_rain_threshold=VAR_THRESHOLDS[var],
                prob_threshold=0.5
            )
        else:
            fake_ens = all_pred.reshape(-1, 1)
            metrics = compute_all_metrics(
                ensemble_samples=fake_ens,
                observations=all_act,
                deterministic_predictions=all_pred,
                heavy_rain_threshold=VAR_THRESHOLDS[var],
                prob_threshold=0.5
            )
        agg[var] = metrics

    all_scenario_aggregated[sc] = agg

# Print aggregated comparison
print(f"\n  {'Scenario':<20} {'Variable':<15} {'RMSE':<8} {'MAE':<8} {'Corr':<8} {'CRPS':<8} {'POD':<8} {'CSI':<8}")
print(f"  {'-'*85}")
for sc in SCENARIOS:
    for var in VARIABLES:
        if var not in all_scenario_aggregated[sc]:
            continue
        m = all_scenario_aggregated[sc][var]
        def fmt(v):
            return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
        print(f"  {sc:<20} {var:<15} {fmt(m['rmse']):<8} {fmt(m['mae']):<8} {fmt(m['correlation']):<8} "
              f"{fmt(m['crps']):<8} {fmt(m['pod']):<8} {fmt(m['csi']):<8}")
    print()

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
print("\n[Saving results...]")

# 1. Per-scenario JSON
for sc in SCENARIOS:
    out = {
        'per_node': make_serializable(all_scenario_results[sc]),
        'aggregated': make_serializable(all_scenario_aggregated[sc]),
    }
    with open(f"{OUTPUT_DIR}/{sc}/metrics.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"  Saved: {OUTPUT_DIR}/{sc}/metrics.json")

# 2. Comparison summary table
comparison_rows = []
for sc in SCENARIOS:
    for var in VARIABLES:
        if var not in all_scenario_aggregated[sc]:
            continue
        m = all_scenario_aggregated[sc][var]
        row = {'scenario': sc, 'variable': var}
        row.update(make_serializable(m))
        comparison_rows.append(row)

comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(f"{OUTPUT_DIR}/comparison/comparison_summary.csv", index=False)

with open(f"{OUTPUT_DIR}/comparison/comparison_summary.json", 'w') as f:
    json.dump(make_serializable(comparison_rows), f, indent=2)

print(f"  Saved: {OUTPUT_DIR}/comparison/comparison_summary.csv")
print(f"  Saved: {OUTPUT_DIR}/comparison/comparison_summary.json")

# ==============================================================================
# PLOTS
# ==============================================================================
print("\n[Generating plots...]")

# --- Plot 1: Bar chart comparison RMSE per variable ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
colors = {'persistence': '#e74c3c', 'mlp_baseline': '#f39c12', 'diffusion_pure': '#3498db', 'diffusion_hybrid': '#2ecc71'}
labels = {'persistence': 'Persistence', 'mlp_baseline': 'MLP Baseline', 'diffusion_pure': 'Diffusion (Pure)', 'diffusion_hybrid': 'Diffusion+Hybrid'}

for i, var in enumerate(VARIABLES):
    ax = axes[i]
    vals = []
    names = []
    cols = []
    for sc in SCENARIOS:
        if var in all_scenario_aggregated[sc]:
            vals.append(all_scenario_aggregated[sc][var]['rmse'])
            names.append(labels[sc])
            cols.append(colors[sc])
    ax.bar(range(len(vals)), vals, color=cols, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title(f'RMSE — {var}', fontsize=11)
    ax.set_ylabel('RMSE')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('RMSE Comparison Across 4 Scenarios (Aggregated)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/rmse_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 2: Bar chart comparison Correlation per variable ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, var in enumerate(VARIABLES):
    ax = axes[i]
    vals = []
    names = []
    cols = []
    for sc in SCENARIOS:
        if var in all_scenario_aggregated[sc]:
            c = all_scenario_aggregated[sc][var]['correlation']
            vals.append(c if c is not None and not np.isnan(c) else 0)
            names.append(labels[sc])
            cols.append(colors[sc])
    ax.bar(range(len(vals)), vals, color=cols, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title(f'Correlation — {var}', fontsize=11)
    ax.set_ylabel('Pearson r')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Correlation Comparison Across 4 Scenarios (Aggregated)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/correlation_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 3: MAE comparison ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, var in enumerate(VARIABLES):
    ax = axes[i]
    vals = []
    names = []
    cols = []
    for sc in SCENARIOS:
        if var in all_scenario_aggregated[sc]:
            vals.append(all_scenario_aggregated[sc][var]['mae'])
            names.append(labels[sc])
            cols.append(colors[sc])
    ax.bar(range(len(vals)), vals, color=cols, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax.set_title(f'MAE — {var}', fontsize=11)
    ax.set_ylabel('MAE')
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('MAE Comparison Across 4 Scenarios (Aggregated)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/mae_comparison.png", dpi=150, bbox_inches='tight')
plt.close()

# --- Plot 4: Time series comparison (first node, max 200 samples) ---
first_node = NODE_NAMES[0]
n_plot = 200

for var, unit in [('precipitation', 'mm/jam'), ('wind_speed', 'm/s'), ('humidity', '%')]:
    fig, ax = plt.subplots(figsize=(16, 5))

    act = all_scenario_data['persistence'].get(first_node, {}).get('actuals', {}).get(var)
    if act is None or len(act) == 0:
        plt.close()
        continue

    n = min(n_plot, len(act))
    x = np.arange(n)

    ax.plot(x, act[:n], label='Actual', color='black', linewidth=1.2, alpha=0.9)
    for sc in SCENARIOS:
        p = all_scenario_data[sc][first_node]['preds'][var][:n]
        ax.plot(x, p, label=labels[sc], color=colors[sc], linewidth=0.8, alpha=0.7)

    ax.set_title(f'{var} — {first_node} (Test Set, First {n} samples)', fontsize=12)
    ax.set_xlabel('Sample Index (daily)')
    ax.set_ylabel(f'{var} ({unit})')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/timeseries_{var}_{first_node}.png", dpi=150, bbox_inches='tight')
    plt.close()

# --- Plot 5: Scatter actual vs predicted per scenario ---
for var in VARIABLES:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for j, sc in enumerate(SCENARIOS):
        ax = axes[j]
        all_act_list = []
        all_pred_list = []
        for n in NODE_NAMES:
            if n in all_scenario_data[sc]:
                all_act_list.append(all_scenario_data[sc][n]['actuals'][var])
                all_pred_list.append(all_scenario_data[sc][n]['preds'][var])
        if not all_act_list:
            continue
        act_all = np.concatenate(all_act_list)
        pred_all = np.concatenate(all_pred_list)
        corr = np.corrcoef(act_all, pred_all)[0, 1] if len(act_all) > 2 else 0
        ax.scatter(act_all, pred_all, alpha=0.3, s=10, color=colors[sc])
        lims = [min(act_all.min(), pred_all.min()), max(act_all.max(), pred_all.max())]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect')
        ax.set_title(f'{labels[sc]} (r={corr:.3f})', fontsize=10)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    plt.suptitle(f'Scatter: Actual vs Predicted — {var}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/plots/scatter_{var}.png", dpi=150, bbox_inches='tight')
    plt.close()

# --- Plot 6: Per-node RMSE heatmap ---
fig, ax = plt.subplots(figsize=(10, 6))
rmse_matrix = []
row_labels = []
for sc in SCENARIOS:
    row = []
    for var in VARIABLES:
        if var in all_scenario_aggregated[sc]:
            row.append(all_scenario_aggregated[sc][var]['rmse'])
        else:
            row.append(0)
    rmse_matrix.append(row)
    row_labels.append(labels[sc])

rmse_matrix = np.array(rmse_matrix)
im = ax.imshow(rmse_matrix, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(VARIABLES)))
ax.set_xticklabels(VARIABLES)
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels)

for i in range(len(row_labels)):
    for j in range(len(VARIABLES)):
        ax.text(j, i, f'{rmse_matrix[i,j]:.3f}', ha='center', va='center', fontsize=11, fontweight='bold')

plt.colorbar(im, label='RMSE')
ax.set_title('RMSE Heatmap — All Scenarios', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plots/rmse_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"\n[Plots saved to {OUTPUT_DIR}/plots/]")

# ==============================================================================
# GENERATE MARKDOWN REPORT
# ==============================================================================
print("\n[Generating report...]")

report_lines = []
report_lines.append("# Hasil Evaluasi 4 Skenario — Honest Comparison")
report_lines.append(f"\n**Tanggal:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report_lines.append(f"**Test Period:** 2022-2025")
report_lines.append(f"**Eval Step:** {EVAL_STEP}h (daily)")
report_lines.append(f"**Ensemble Size:** {NUM_ENSEMBLE} (untuk diffusion)")
report_lines.append(f"**Nodes:** {', '.join(NODE_NAMES)}")
report_lines.append("")

report_lines.append("## Skenario yang Dibandingkan")
report_lines.append("")
report_lines.append("| # | Skenario | Deskripsi |")
report_lines.append("|---|----------|-----------|")
report_lines.append("| 1 | Persistence | Prediksi = nilai jam sebelumnya (naive baseline) |")
report_lines.append("| 2 | MLP Baseline | MLP 3-layer, evaluasi per-node (apple-to-apple) |")
report_lines.append("| 3 | Diffusion (Pure) | Model diffusion TANPA hybrid blending |")
report_lines.append("| 4 | Diffusion+Hybrid | Diffusion + persistence blending (w=0.9/0.9/0.7) |")
report_lines.append("")

# Aggregated table
report_lines.append("## Metrik Agregasi (Semua Node)")
report_lines.append("")
report_lines.append("| Skenario | Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |")
report_lines.append("|----------|----------|------|-----|------|------|-------|-----|-----|-----|")

for sc in SCENARIOS:
    for var in VARIABLES:
        if var not in all_scenario_aggregated[sc]:
            continue
        m = all_scenario_aggregated[sc][var]
        def fmt(v):
            return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
        report_lines.append(
            f"| {labels[sc]} | {var} | {fmt(m['rmse'])} | {fmt(m['mae'])} | {fmt(m['correlation'])} | "
            f"{fmt(m['crps'])} | {fmt(m['brier_score'])} | {fmt(m['pod'])} | {fmt(m['far'])} | {fmt(m['csi'])} |"
        )
report_lines.append("")

# Per-node tables
for node_name in NODE_NAMES:
    report_lines.append(f"## Metrik Per Node: {node_name}")
    report_lines.append("")
    report_lines.append("| Skenario | Variable | RMSE | MAE | Corr |")
    report_lines.append("|----------|----------|------|-----|------|")

    for sc in SCENARIOS:
        if node_name not in all_scenario_results[sc]:
            continue
        for var in VARIABLES:
            m = all_scenario_results[sc][node_name][var]
            def fmt(v):
                return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
            report_lines.append(
                f"| {labels[sc]} | {var} | {fmt(m['rmse'])} | {fmt(m['mae'])} | {fmt(m['correlation'])} |"
            )
    report_lines.append("")

# Winner analysis
report_lines.append("## Analisis Pemenang per Metrik")
report_lines.append("")
report_lines.append("| Variable | Best RMSE | Best MAE | Best Corr |")
report_lines.append("|----------|-----------|----------|-----------|")
for var in VARIABLES:
    best_rmse_sc = None
    best_rmse_v = float('inf')
    best_mae_sc = None
    best_mae_v = float('inf')
    best_corr_sc = None
    best_corr_v = -1

    for sc in SCENARIOS:
        if var not in all_scenario_aggregated[sc]:
            continue
        m = all_scenario_aggregated[sc][var]
        if m['rmse'] is not None and m['rmse'] < best_rmse_v:
            best_rmse_v = m['rmse']
            best_rmse_sc = labels[sc]
        if m['mae'] is not None and m['mae'] < best_mae_v:
            best_mae_v = m['mae']
            best_mae_sc = labels[sc]
        c = m['correlation']
        if c is not None and not np.isnan(c) and c > best_corr_v:
            best_corr_v = c
            best_corr_sc = labels[sc]

    report_lines.append(
        f"| {var} | {best_rmse_sc} ({best_rmse_v:.4f}) | {best_mae_sc} ({best_mae_v:.4f}) | {best_corr_sc} ({best_corr_v:.4f}) |"
    )
report_lines.append("")

report_lines.append("## Interpretasi")
report_lines.append("")
report_lines.append("- **Persistence** adalah baseline paling sederhana: prediksi = nilai jam sebelumnya.")
report_lines.append("- Jika model lain tidak signifikan lebih baik dari persistence, berarti model belum memberikan added value.")
report_lines.append("- **Diffusion+Hybrid** dengan w=0.9 berarti 90% prediksi berasal dari persistence — sehingga hasilnya akan mirip persistence.")
report_lines.append("- **Diffusion (Pure)** menunjukkan kemampuan sebenarnya dari model diffusion tanpa 'bantuan' persistence.")
report_lines.append("- Perbandingan yang fair: **MLP Baseline vs Diffusion (Pure)** — keduanya murni model learning.")
report_lines.append("")

report_lines.append("## Plot yang Dihasilkan")
report_lines.append("")
report_lines.append("1. `plots/rmse_comparison.png` — Bar chart RMSE per variabel")
report_lines.append("2. `plots/correlation_comparison.png` — Bar chart Correlation per variabel")
report_lines.append("3. `plots/mae_comparison.png` — Bar chart MAE per variabel")
report_lines.append("4. `plots/timeseries_*.png` — Time series perbandingan 4 skenario")
report_lines.append("5. `plots/scatter_*.png` — Scatter plot actual vs predicted")
report_lines.append("6. `plots/rmse_heatmap.png` — Heatmap RMSE semua skenario")
report_lines.append("")

with open(f"{OUTPUT_DIR}/EVALUATION_REPORT.md", 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"  Saved: {OUTPUT_DIR}/EVALUATION_REPORT.md")

# ==============================================================================
# DONE
# ==============================================================================
elapsed = time.time() - time.time()  # will recalculate
print(f"\n{'='*80}")
print("4-SCENARIO EVALUATION COMPLETE!")
print(f"{'='*80}")
print(f"\nAll results saved to: {OUTPUT_DIR}/")
print(f"  {OUTPUT_DIR}/persistence/metrics.json")
print(f"  {OUTPUT_DIR}/mlp_baseline/metrics.json")
print(f"  {OUTPUT_DIR}/diffusion_pure/metrics.json")
print(f"  {OUTPUT_DIR}/diffusion_hybrid/metrics.json")
print(f"  {OUTPUT_DIR}/comparison/comparison_summary.csv")
print(f"  {OUTPUT_DIR}/comparison/comparison_summary.json")
print(f"  {OUTPUT_DIR}/EVALUATION_REPORT.md")
print(f"  {OUTPUT_DIR}/plots/*.png")
