"""
FINAL EVALUATION - Retrieval-Augmented Diffusion Model
dengan Spatio-Temporal Graph Conditioning

Evaluasi pada test set LENGKAP:
- Periode: 2022-2025
- Node: Puncak, Lereng_Cibodas, Hilir_Cianjur

Metrik:
- Deterministik: RMSE, MAE, Correlation
- Probabilistik: CRPS, Brier Score, POD, FAR, CSI

Variabel:
- precipitation (mm/jam)
- wind_speed (m/s)
- humidity (%)
"""

import sys
sys.path.append('.')

import os
import torch
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.inference import load_model_and_stats, run_inference_hybrid
from src.evaluation.probabilistic_metrics import (
    compute_all_metrics,
    compute_reliability_data
)

print("=" * 80)
print("FINAL EVALUATION - FULL TEST SET (2022-2025)")
print("Retrieval-Augmented Diffusion Model + Spatio-Temporal Graph Conditioning")
print("=" * 80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, stats, retrieval_db = load_model_and_stats('models/diffusion_chkpt.pth')
model.to(device)
model.eval()

print(f"Device: {device}")
print(f"Targets: {model.config.get('target_cols')}")

# ==============================================================================
# LOAD DATA & FILTER TEST SET
# ==============================================================================
df = pd.read_parquet('data/raw/pangrango_era5_2005_2025.parquet')
df['date'] = pd.to_datetime(df['date'])
if df['date'].dt.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)

# Test set: 2022-2025
test_df = df[df['date'] >= '2022-01-01'].copy()
print(f"\nTest set: {len(test_df):,} rows ({test_df['date'].min()} to {test_df['date'].max()})")

# Nodes
NODE_NAMES = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur']
available_cols = [c for c in model.config.get('feature_cols', []) if c in test_df.columns]
c_mean = stats['c_mean'].numpy()[:len(available_cols)]
c_std = stats['c_std'].numpy()[:len(available_cols)]

seq_len = model.config['seq_len']
num_samples = 50  # Ensemble size

# ==============================================================================
# EVALUASI PER NODE
# ==============================================================================
all_results = {}
ensemble_storage = {}

for node_name in NODE_NAMES:
    print(f"\n{'='*60}")
    print(f"  Evaluating Node: {node_name}")
    print(f"{'='*60}")
    
    node_df = test_df[test_df['node'] == node_name].copy().sort_values('date').reset_index(drop=True)
    
    if len(node_df) < seq_len + 1:
        print(f"  SKIP: not enough data ({len(node_df)} rows)")
        continue
    
    n_hours = len(node_df) - seq_len - 1
    print(f"  Data points: {len(node_df):,}, Eval samples: {n_hours:,}")
    
    actuals = {'precipitation': [], 'wind_speed': [], 'humidity': []}
    hybrid_preds = {'precipitation': [], 'wind_speed': [], 'humidity': []}
    ensemble_preds = {'precipitation': [], 'wind_speed': [], 'humidity': []}
    
    for idx in tqdm(range(n_hours), desc=f"  {node_name}", leave=True):
        seq_end = idx + seq_len
        if seq_end >= len(node_df):
            break
        
        seq_df = node_df.iloc[idx:seq_end]
        features = seq_df[available_cols].values
        features_norm = (features - c_mean) / (c_std + 1e-5)
        
        target_row = node_df.iloc[seq_end]
        prev_row = node_df.iloc[seq_end - 1]
        
        lag_values = {
            'precipitation': prev_row['precipitation'],
            'wind_speed': prev_row['wind_speed_10m'],
            'humidity': prev_row['relative_humidity_2m']
        }
        
        result = run_inference_hybrid(
            features_norm, model, stats, retrieval_db,
            lag_values=lag_values, num_samples=num_samples, device=device
        )
        
        actuals['precipitation'].append(target_row['precipitation'])
        actuals['wind_speed'].append(target_row['wind_speed_10m'])
        actuals['humidity'].append(target_row['relative_humidity_2m'])
        
        hybrid_preds['precipitation'].append(np.median(result['hybrid_precipitation']))
        hybrid_preds['wind_speed'].append(np.median(result['hybrid_wind_speed']))
        hybrid_preds['humidity'].append(np.median(result['hybrid_humidity']))
        
        ensemble_preds['precipitation'].append(result['hybrid_precipitation'].copy())
        ensemble_preds['wind_speed'].append(result['hybrid_wind_speed'].copy())
        ensemble_preds['humidity'].append(result['hybrid_humidity'].copy())
    
    for key in actuals:
        actuals[key] = np.array(actuals[key])
        hybrid_preds[key] = np.array(hybrid_preds[key])
        ensemble_preds[key] = np.array(ensemble_preds[key])
    
    node_results = {}
    var_map = {
        'precipitation': ('precipitation', 10.0),
        'wind_speed': ('wind_speed', 10.0),
        'humidity': ('humidity', 90.0)
    }
    
    for var, (var_key, threshold) in var_map.items():
        metrics = compute_all_metrics(
            ensemble_samples=ensemble_preds[var],
            observations=actuals[var],
            deterministic_predictions=hybrid_preds[var],
            heavy_rain_threshold=threshold,
            prob_threshold=0.5
        )
        node_results[var] = metrics
    
    all_results[node_name] = node_results
    ensemble_storage[node_name] = {
        'actuals': actuals,
        'hybrid_preds': hybrid_preds,
        'ensemble_preds': ensemble_preds
    }
    
    print(f"\n  {'Variable':<15} {'RMSE':<8} {'MAE':<8} {'Corr':<8} {'CRPS':<8} {'Brier':<8} {'POD':<8} {'FAR':<8} {'CSI':<8}")
    print(f"  {'-'*79}")
    for var in ['precipitation', 'wind_speed', 'humidity']:
        m = node_results[var]
        pod_str = f"{m['pod']:.3f}" if not np.isnan(m['pod']) else "N/A"
        far_str = f"{m['far']:.3f}" if not np.isnan(m['far']) else "N/A"
        csi_str = f"{m['csi']:.3f}" if not np.isnan(m['csi']) else "N/A"
        print(f"  {var:<15} {m['rmse']:<8.3f} {m['mae']:<8.3f} {m['correlation']:<8.3f} "
              f"{m['crps']:<8.3f} {m['brier_score']:<8.3f} {pod_str:<8} {far_str:<8} {csi_str:<8}")

# ==============================================================================
# AGGREGATE ACROSS ALL NODES
# ==============================================================================
print(f"\n{'='*80}")
print("AGGREGATED RESULTS (All Nodes)")
print(f"{'='*80}")

aggregated_results = {}
for var in ['precipitation', 'wind_speed', 'humidity']:
    all_actuals = np.concatenate([ensemble_storage[n]['actuals'][var] for n in ensemble_storage])
    all_hybrid = np.concatenate([ensemble_storage[n]['hybrid_preds'][var] for n in ensemble_storage])
    all_ensemble = np.concatenate([ensemble_storage[n]['ensemble_preds'][var] for n in ensemble_storage])
    
    var_threshold = {'precipitation': 10.0, 'wind_speed': 10.0, 'humidity': 90.0}[var]
    
    metrics = compute_all_metrics(
        ensemble_samples=all_ensemble,
        observations=all_actuals,
        deterministic_predictions=all_hybrid,
        heavy_rain_threshold=var_threshold,
        prob_threshold=0.5
    )
    aggregated_results[var] = metrics

print(f"\n{'Variable':<15} {'RMSE':<8} {'MAE':<8} {'Corr':<8} {'CRPS':<8} {'Brier':<8} {'POD':<8} {'FAR':<8} {'CSI':<8}")
print(f"{'-'*79}")
for var in ['precipitation', 'wind_speed', 'humidity']:
    m = aggregated_results[var]
    pod_str = f"{m['pod']:.3f}" if not np.isnan(m['pod']) else "N/A"
    far_str = f"{m['far']:.3f}" if not np.isnan(m['far']) else "N/A"
    csi_str = f"{m['csi']:.3f}" if not np.isnan(m['csi']) else "N/A"
    print(f"{var:<15} {m['rmse']:<8.3f} {m['mae']:<8.3f} {m['correlation']:<8.3f} "
          f"{m['crps']:<8.3f} {m['brier_score']:<8.3f} {pod_str:<8} {far_str:<8} {csi_str:<8}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
os.makedirs("results/diffusion_results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

def make_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
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

with open("results/diffusion_results/diffusion_metrics.json", 'w') as f:
    json.dump(make_serializable({
        'per_node': all_results,
        'aggregated': aggregated_results,
        'config': {
            'test_period': '2022-2025',
            'nodes': NODE_NAMES,
            'ensemble_size': num_samples,
            'heavy_rain_threshold_mm': 10.0
        }
    }), f, indent=2)

prob_metrics = {}
for var in ['precipitation', 'wind_speed', 'humidity']:
    m = aggregated_results[var]
    prob_metrics[var] = {
        'crps': make_serializable(m['crps']),
        'brier_score': make_serializable(m['brier_score']),
        'pod': make_serializable(m['pod']),
        'far': make_serializable(m['far']),
        'csi': make_serializable(m['csi'])
    }

with open("results/probabilistic_metrics.json", 'w') as f:
    json.dump(prob_metrics, f, indent=2)

summary_rows = []
for var in ['precipitation', 'wind_speed', 'humidity']:
    m = aggregated_results[var]
    row = {'variable': var}
    row.update(make_serializable(m))
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/tables/metrics_summary.csv", index=False)

with open("results/tables/metrics_summary.json", 'w') as f:
    json.dump(make_serializable(summary_rows), f, indent=2)

# ==============================================================================
# GENERATE PLOTS
# ==============================================================================
print("\nGenerating plots...")

plot_node = list(ensemble_storage.keys())[0]
plot_data = ensemble_storage[plot_node]
max_plot = 500

for var, var_label, unit in [
    ('precipitation', 'Rain Prediction vs Actual', 'mm/jam'),
    ('wind_speed', 'Wind Prediction vs Actual', 'm/s'),
    ('humidity', 'Humidity Prediction vs Actual', '%')
]:
    actual = plot_data['actuals'][var][:max_plot]
    pred = plot_data['hybrid_preds'][var][:max_plot]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(actual, label='Actual', alpha=0.8, linewidth=0.8)
    ax.plot(pred, label='Predicted (Hybrid)', alpha=0.8, linewidth=0.8)
    ax.set_title(f'{var_label} - {plot_node} (Test Set 2022-2025)')
    ax.set_xlabel('Time Step (hours)')
    ax.set_ylabel(f'{var} ({unit})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f"results/plots/{var}_prediction_vs_actual.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

fig, axes = plt.subplots(3, 1, figsize=(14, 12))
for i, (var, var_label) in enumerate([
    ('precipitation', 'Precipitation (mm/jam)'),
    ('wind_speed', 'Wind Speed (m/s)'),
    ('humidity', 'Humidity (%)')
]):
    actual = plot_data['actuals'][var][:max_plot]
    ens = plot_data['ensemble_preds'][var][:max_plot]
    
    median_pred = np.median(ens, axis=1)
    q10 = np.percentile(ens, 10, axis=1)
    q90 = np.percentile(ens, 90, axis=1)
    
    axes[i].fill_between(range(len(actual)), q10, q90, alpha=0.3, color='blue', label='10-90% interval')
    axes[i].plot(actual, label='Actual', alpha=0.8, linewidth=0.8, color='black')
    axes[i].plot(median_pred, label='Median Prediction', alpha=0.8, linewidth=0.8, color='blue')
    axes[i].set_title(f'Ensemble Spread - {var_label}')
    axes[i].set_ylabel(var_label)
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

axes[2].set_xlabel('Time Step (hours)')
plt.suptitle(f'Ensemble Spread - {plot_node} (Test Set 2022-2025)', y=1.01)
plt.tight_layout()
plt.savefig("results/plots/ensemble_spread.png", dpi=150, bbox_inches='tight')
plt.close()

all_actuals_precip = np.concatenate([ensemble_storage[n]['actuals']['precipitation'] for n in ensemble_storage])
all_ensemble_precip = np.concatenate([ensemble_storage[n]['ensemble_preds']['precipitation'] for n in ensemble_storage])

rel_data = compute_reliability_data(all_ensemble_precip, all_actuals_precip, threshold=10.0, n_bins=10)

fig, ax = plt.subplots(figsize=(8, 8))
valid_mask = [not np.isnan(o) for o in rel_data['observed_freqs']]
fp = [rel_data['forecast_probs'][i] for i in range(len(valid_mask)) if valid_mask[i]]
of = [rel_data['observed_freqs'][i] for i in range(len(valid_mask)) if valid_mask[i]]

ax.plot([0, 1], [0, 1], 'k--', label='Perfect reliability')
if fp and of:
    ax.plot(fp, of, 'bo-', label='Model', markersize=8)
ax.set_xlabel('Forecast Probability')
ax.set_ylabel('Observed Frequency')
ax.set_title('Reliability Diagram - Heavy Rain (>10 mm/jam)')
ax.legend()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("results/plots/reliability_diagram.png", dpi=150, bbox_inches='tight')
plt.close()

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print(f"\nTest Period: 2022-2025")
print(f"Nodes: {', '.join(NODE_NAMES)}")
print(f"Ensemble Size: {num_samples}")
print(f"\nResults saved to:")
print(f"  - results/diffusion_results/diffusion_metrics.json")
print(f"  - results/probabilistic_metrics.json")
print(f"  - results/tables/metrics_summary.csv")
print(f"  - results/tables/metrics_summary.json")
print(f"  - results/plots/")
print("=" * 80)
