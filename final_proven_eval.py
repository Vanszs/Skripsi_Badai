"""
FINAL EVALUATION - Retrieval-Augmented Diffusion Model
dengan Spatio-Temporal Graph Conditioning

Gunung Gede-Pangrango Nowcasting System

Dataset Split (metodologi penelitian):
  Training:   2005-2018 (14 tahun)
  Validation: 2019-2021 (3 tahun) - early stopping & tuning only
  Test:       2022-2025 (4 tahun) - evaluasi performa final

Evaluasi:
  - Periode test: 2022-2025
  - Subsampling: EVAL_STEP = 24 (setiap 24 jam)
  - Node: Puncak, Lereng_Cibodas, Hilir_Cianjur
  - Variabel: precipitation, wind_speed, humidity

Metrik:
  Deterministik: RMSE, MAE, Correlation
  Probabilistik: CRPS, Brier Score, POD, FAR, CSI
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

from src.inference import load_model_and_stats, run_inference_hybrid
from src.evaluation.probabilistic_metrics import (
    compute_all_metrics,
    compute_reliability_data
)

warnings.filterwarnings('ignore', category=FutureWarning)
start_time = time.time()

# ==============================================================================
# CONFIGURATION
# ==============================================================================
EVAL_STEP = 24        # Evaluasi setiap 24 jam (daily subsampling)
NUM_ENSEMBLE = 30     # Ukuran ensemble (cukup untuk CRPS/Brier)
NODE_NAMES = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur']
VARIABLES = ['precipitation', 'wind_speed', 'humidity']
HEAVY_RAIN_THRESHOLD = 10.0  # mm/jam

# Threshold per variabel untuk Brier/POD/FAR/CSI
VAR_THRESHOLDS = {
    'precipitation': 10.0,   # hujan lebat > 10 mm/jam
    'wind_speed': 10.0,      # angin kencang > 10 m/s
    'humidity': 90.0          # kelembapan sangat tinggi > 90%
}

MAX_PLOT_SAMPLES = 400  # Maksimal sampel untuk time series plot

# ==============================================================================
# HEADER
# ==============================================================================
print("=" * 80)
print("FINAL EVALUATION - FULL TEST SET (2022-2025)")
print("Retrieval-Augmented Diffusion Model + Spatio-Temporal Graph Conditioning")
print("Gunung Gede-Pangrango Nowcasting System")
print("=" * 80)

# ==============================================================================
# LOAD MODEL
# ==============================================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, stats, retrieval_db = load_model_and_stats('models/diffusion_chkpt.pth')
model.to(device)
model.eval()

seq_len = model.config['seq_len']
feature_cols = model.config.get('feature_cols', [])
target_cols = model.config.get('target_cols', [])

print(f"\n[Config]")
print(f"  Device:          {device}")
print(f"  Targets:         {target_cols}")
print(f"  Seq Length:      {seq_len}")
print(f"  Ensemble Size:   {NUM_ENSEMBLE}")
print(f"  Eval Step:       {EVAL_STEP}h (daily subsampling)")
print(f"  DDIM Steps:      20 (fast inference)")

# ==============================================================================
# LOAD & SPLIT DATA
# ==============================================================================
df = pd.read_parquet('data/raw/pangrango_era5_2005_2025.parquet')
df['date'] = pd.to_datetime(df['date'])
if df['date'].dt.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)

# Strict temporal split
test_df = df[df['date'] >= '2022-01-01'].copy()

available_cols = [c for c in feature_cols if c in test_df.columns]
c_mean = stats['c_mean'].numpy()[:len(available_cols)]
c_std = stats['c_std'].numpy()[:len(available_cols)]

total_test_hours = len(test_df)
print(f"\n[Data Split]")
print(f"  Training:     2005-2018 (normalization stats)")
print(f"  Validation:   2019-2021 (early stopping)")
print(f"  Test:         2022-2025 (evaluasi final)")
print(f"  Total test rows: {total_test_hours:,}")
print(f"  Date range:   {test_df['date'].min()} to {test_df['date'].max()}")

# ==============================================================================
# EVALUATION LOOP PER NODE
# ==============================================================================
all_results = {}
ensemble_storage = {}
eval_stats = {}

for node_name in NODE_NAMES:
    print(f"\n{'='*60}")
    print(f"  Node: {node_name}")
    print(f"{'='*60}")

    node_df = test_df[test_df['node'] == node_name].copy()
    node_df = node_df.sort_values('date').reset_index(drop=True)

    if len(node_df) < seq_len + 1:
        print(f"  SKIP: insufficient data ({len(node_df)} rows)")
        continue

    n_hours = len(node_df) - seq_len - 1
    eval_indices = list(range(0, n_hours, EVAL_STEP))
    n_eval = len(eval_indices)

    print(f"  Total hours:   {len(node_df):,}")
    print(f"  Eval samples:  {n_eval:,} (every {EVAL_STEP}h)")

    actuals = {v: [] for v in VARIABLES}
    hybrid_preds = {v: [] for v in VARIABLES}
    ensemble_preds = {v: [] for v in VARIABLES}
    nan_count = 0

    for idx in tqdm(eval_indices, desc=f"  {node_name}", leave=True, ncols=80):
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

        try:
            result = run_inference_hybrid(
                features_norm, model, stats, retrieval_db,
                lag_values=lag_values, num_samples=NUM_ENSEMBLE, device=device
            )
        except Exception:
            nan_count += 1
            continue

        # Extract ensemble predictions and check for NaN
        ens_precip = result.get('hybrid_precipitation', result.get('precipitation'))
        ens_wind = result.get('hybrid_wind_speed', result.get('wind_speed'))
        ens_humid = result.get('hybrid_humidity', result.get('humidity'))

        # Skip if any ensemble is entirely NaN
        if (np.all(np.isnan(ens_precip)) or np.all(np.isnan(ens_wind))
                or np.all(np.isnan(ens_humid))):
            nan_count += 1
            continue

        # Replace remaining NaN in ensemble with median of valid values
        for ens in [ens_precip, ens_wind, ens_humid]:
            nans = np.isnan(ens)
            if nans.any() and not nans.all():
                ens[nans] = np.nanmedian(ens)

        actuals['precipitation'].append(target_row['precipitation'])
        actuals['wind_speed'].append(target_row['wind_speed_10m'])
        actuals['humidity'].append(target_row['relative_humidity_2m'])

        hybrid_preds['precipitation'].append(float(np.nanmedian(ens_precip)))
        hybrid_preds['wind_speed'].append(float(np.nanmedian(ens_wind)))
        hybrid_preds['humidity'].append(float(np.nanmedian(ens_humid)))

        ensemble_preds['precipitation'].append(ens_precip.copy())
        ensemble_preds['wind_speed'].append(ens_wind.copy())
        ensemble_preds['humidity'].append(ens_humid.copy())

    # Convert to arrays
    valid_samples = len(actuals['precipitation'])
    for key in VARIABLES:
        actuals[key] = np.array(actuals[key], dtype=np.float64)
        hybrid_preds[key] = np.array(hybrid_preds[key], dtype=np.float64)
        ensemble_preds[key] = np.array(ensemble_preds[key], dtype=np.float64)

    eval_stats[node_name] = {
        'total_hours': len(node_df),
        'eval_attempted': n_eval,
        'valid_samples': valid_samples,
        'nan_skipped': nan_count
    }

    if valid_samples == 0:
        print(f"  WARNING: No valid samples for {node_name}")
        continue

    print(f"  Valid samples: {valid_samples}/{n_eval} (skipped {nan_count} NaN)")

    # Compute metrics per variable
    node_results = {}
    for var in VARIABLES:
        metrics = compute_all_metrics(
            ensemble_samples=ensemble_preds[var],
            observations=actuals[var],
            deterministic_predictions=hybrid_preds[var],
            heavy_rain_threshold=VAR_THRESHOLDS[var],
            prob_threshold=0.5
        )
        node_results[var] = metrics

    all_results[node_name] = node_results
    ensemble_storage[node_name] = {
        'actuals': actuals,
        'hybrid_preds': hybrid_preds,
        'ensemble_preds': ensemble_preds
    }

    # Print per-node results
    print(f"\n  {'Variable':<15} {'RMSE':<8} {'MAE':<8} {'Corr':<8} {'CRPS':<8} {'Brier':<8} {'POD':<8} {'FAR':<8} {'CSI':<8}")
    print(f"  {'-'*79}")
    for var in VARIABLES:
        m = node_results[var]
        def fmt(v):
            return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
        print(f"  {var:<15} {fmt(m['rmse']):<8} {fmt(m['mae']):<8} {fmt(m['correlation']):<8} "
              f"{fmt(m['crps']):<8} {fmt(m['brier_score']):<8} {fmt(m['pod']):<8} {fmt(m['far']):<8} {fmt(m['csi']):<8}")

# ==============================================================================
# AGGREGATE ACROSS ALL NODES
# ==============================================================================
print(f"\n{'='*80}")
print("AGGREGATED RESULTS (All Nodes)")
print(f"{'='*80}")

aggregated_results = {}
for var in VARIABLES:
    nodes_with_data = [n for n in ensemble_storage if len(ensemble_storage[n]['actuals'][var]) > 0]
    if not nodes_with_data:
        continue

    all_actuals = np.concatenate([ensemble_storage[n]['actuals'][var] for n in nodes_with_data])
    all_hybrid = np.concatenate([ensemble_storage[n]['hybrid_preds'][var] for n in nodes_with_data])
    all_ensemble = np.concatenate([ensemble_storage[n]['ensemble_preds'][var] for n in nodes_with_data])

    metrics = compute_all_metrics(
        ensemble_samples=all_ensemble,
        observations=all_actuals,
        deterministic_predictions=all_hybrid,
        heavy_rain_threshold=VAR_THRESHOLDS[var],
        prob_threshold=0.5
    )
    aggregated_results[var] = metrics

total_valid = sum(s['valid_samples'] for s in eval_stats.values())
total_hours_all = sum(s['total_hours'] for s in eval_stats.values())

print(f"\n  Total test hours (all nodes): {total_hours_all:,}")
print(f"  Total evaluated samples:      {total_valid:,}")
print(f"  Evaluation step:              {EVAL_STEP}h\n")

print(f"  {'Variable':<15} {'RMSE':<8} {'MAE':<8} {'Corr':<8} {'CRPS':<8} {'Brier':<8} {'POD':<8} {'FAR':<8} {'CSI':<8}")
print(f"  {'-'*79}")
for var in VARIABLES:
    if var not in aggregated_results:
        continue
    m = aggregated_results[var]
    def fmt(v):
        return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
    print(f"  {var:<15} {fmt(m['rmse']):<8} {fmt(m['mae']):<8} {fmt(m['correlation']):<8} "
          f"{fmt(m['crps']):<8} {fmt(m['brier_score']):<8} {fmt(m['pod']):<8} {fmt(m['far']):<8} {fmt(m['csi']):<8}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
os.makedirs("results/diffusion_results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

def make_serializable(obj):
    """Convert numpy types to Python native for JSON."""
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

# Diffusion metrics JSON
with open("results/diffusion_results/diffusion_metrics.json", 'w') as f:
    json.dump(make_serializable({
        'per_node': all_results,
        'aggregated': aggregated_results,
        'eval_stats': eval_stats,
        'config': {
            'test_period': '2022-2025',
            'nodes': NODE_NAMES,
            'ensemble_size': NUM_ENSEMBLE,
            'eval_step_hours': EVAL_STEP,
            'ddim_steps': 20,
            'variables': VARIABLES,
            'thresholds': VAR_THRESHOLDS
        }
    }), f, indent=2)

# Probabilistic metrics JSON
prob_metrics = {}
for var in VARIABLES:
    if var not in aggregated_results:
        continue
    m = aggregated_results[var]
    prob_metrics[var] = make_serializable({
        'crps': m['crps'],
        'brier_score': m['brier_score'],
        'pod': m['pod'],
        'far': m['far'],
        'csi': m['csi']
    })

with open("results/probabilistic_metrics.json", 'w') as f:
    json.dump(prob_metrics, f, indent=2)

# Summary table CSV + JSON
summary_rows = []
for var in VARIABLES:
    if var not in aggregated_results:
        continue
    m = aggregated_results[var]
    row = {'variable': var}
    row.update(make_serializable(m))
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("results/tables/metrics_summary.csv", index=False)

with open("results/tables/metrics_summary.json", 'w') as f:
    json.dump(make_serializable(summary_rows), f, indent=2)

print(f"\n[Results saved]")
print(f"  results/diffusion_results/diffusion_metrics.json")
print(f"  results/probabilistic_metrics.json")
print(f"  results/tables/metrics_summary.csv")
print(f"  results/tables/metrics_summary.json")

# ==============================================================================
# PLOTS
# ==============================================================================
print(f"\n[Generating plots...]")

plot_node = list(ensemble_storage.keys())[0]
plot_data = ensemble_storage[plot_node]
n_plot = min(MAX_PLOT_SAMPLES, len(plot_data['actuals']['precipitation']))

# --- 1. Time Series: Actual vs Predicted ---
for var, label, unit in [
    ('precipitation', 'Precipitation', 'mm/jam'),
    ('wind_speed', 'Wind Speed', 'm/s'),
    ('humidity', 'Relative Humidity', '%')
]:
    actual = plot_data['actuals'][var][:n_plot]
    pred = plot_data['hybrid_preds'][var][:n_plot]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(actual, label='Actual', alpha=0.85, linewidth=0.8, color='#333333')
    ax.plot(pred, label='Predicted (Hybrid)', alpha=0.85, linewidth=0.8, color='#1f77b4')
    ax.set_title(f'{label}: Actual vs Predicted - {plot_node} (Test 2022-2025)', fontsize=12)
    ax.set_xlabel('Sample Index (daily)')
    ax.set_ylabel(f'{label} ({unit})')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    fname_map = {'precipitation': 'rain', 'wind_speed': 'wind', 'humidity': 'humidity'}
    plt.savefig(f"results/plots/{fname_map[var]}_prediction_vs_actual.png", dpi=150)
    plt.close()

# --- 2. Scatter Plots: Actual vs Predicted ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (var, label, unit) in enumerate([
    ('precipitation', 'Precipitation', 'mm/jam'),
    ('wind_speed', 'Wind Speed', 'm/s'),
    ('humidity', 'Humidity', '%')
]):
    # Aggregate all nodes
    all_a = np.concatenate([ensemble_storage[n]['actuals'][var] for n in ensemble_storage])
    all_p = np.concatenate([ensemble_storage[n]['hybrid_preds'][var] for n in ensemble_storage])

    mask = ~(np.isnan(all_a) | np.isnan(all_p))
    a, p = all_a[mask], all_p[mask]

    axes[i].scatter(a, p, alpha=0.15, s=8, color='#1f77b4')
    vmin = min(a.min(), p.min()) if len(a) > 0 else 0
    vmax = max(a.max(), p.max()) if len(a) > 0 else 1
    axes[i].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1, label='Perfect')
    axes[i].set_xlabel(f'Actual {label} ({unit})')
    axes[i].set_ylabel(f'Predicted {label} ({unit})')
    axes[i].set_title(f'{label}')
    axes[i].legend(loc='upper left', fontsize=8)
    axes[i].grid(True, alpha=0.3)
    axes[i].set_aspect('equal', adjustable='datalim')

plt.suptitle('Scatter: Actual vs Predicted (All Nodes)', fontsize=13)
plt.tight_layout()
plt.savefig("results/plots/scatter_actual_vs_predicted.png", dpi=150)
plt.close()

# --- 3. Ensemble Spread ---
fig, axes = plt.subplots(3, 1, figsize=(14, 11))
for i, (var, label, unit) in enumerate([
    ('precipitation', 'Precipitation (mm/jam)', 'mm/jam'),
    ('wind_speed', 'Wind Speed (m/s)', 'm/s'),
    ('humidity', 'Humidity (%)', '%')
]):
    actual = plot_data['actuals'][var][:n_plot]
    ens = plot_data['ensemble_preds'][var][:n_plot]

    median_pred = np.nanmedian(ens, axis=1)
    q10 = np.nanpercentile(ens, 10, axis=1)
    q90 = np.nanpercentile(ens, 90, axis=1)

    x = np.arange(len(actual))
    axes[i].fill_between(x, q10, q90, alpha=0.25, color='#1f77b4', label='P10-P90')
    axes[i].plot(x, actual, label='Actual', alpha=0.85, linewidth=0.8, color='#333333')
    axes[i].plot(x, median_pred, label='Median Pred', alpha=0.85, linewidth=0.8, color='#1f77b4')
    axes[i].set_ylabel(label)
    axes[i].legend(loc='upper right', fontsize=8)
    axes[i].grid(True, alpha=0.3)

axes[0].set_title(f'Ensemble Spread - {plot_node} (Test 2022-2025)', fontsize=12)
axes[2].set_xlabel('Sample Index (daily)')
plt.tight_layout()
plt.savefig("results/plots/ensemble_spread.png", dpi=150)
plt.close()

# --- 4. Reliability Diagram ---
all_actuals_precip = np.concatenate([ensemble_storage[n]['actuals']['precipitation'] for n in ensemble_storage])
all_ensemble_precip = np.concatenate([ensemble_storage[n]['ensemble_preds']['precipitation'] for n in ensemble_storage])

rel_data = compute_reliability_data(all_ensemble_precip, all_actuals_precip,
                                     threshold=HEAVY_RAIN_THRESHOLD, n_bins=10)

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Reliability')

valid = [(i, fp, of) for i, (fp, of) in enumerate(zip(rel_data['forecast_probs'], rel_data['observed_freqs']))
         if not np.isnan(of) and rel_data['bin_counts'][i] > 0]

if valid:
    fp_vals = [v[1] for v in valid]
    of_vals = [v[2] for v in valid]
    sizes = [max(20, min(200, rel_data['bin_counts'][v[0]])) for v in valid]
    ax.scatter(fp_vals, of_vals, s=sizes, color='#1f77b4', zorder=5, edgecolors='white')
    ax.plot(fp_vals, of_vals, 'b-', alpha=0.6)

ax.set_xlabel('Forecast Probability', fontsize=11)
ax.set_ylabel('Observed Frequency', fontsize=11)
ax.set_title(f'Reliability Diagram - Heavy Rain (>{HEAVY_RAIN_THRESHOLD} mm/jam)', fontsize=12)
ax.legend()
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("results/plots/reliability_diagram.png", dpi=150)
plt.close()

print(f"  Plots saved to results/plots/")

# ==============================================================================
# MARKDOWN REPORT
# ==============================================================================
elapsed = time.time() - start_time

report_lines = []
report_lines.append("# Evaluation Report")
report_lines.append("")
report_lines.append("## Retrieval-Augmented Diffusion Model + Spatio-Temporal Graph Conditioning")
report_lines.append("### Gunung Gede-Pangrango Nowcasting System")
report_lines.append("")
report_lines.append("---")
report_lines.append("")
report_lines.append("## 1. Konfigurasi Evaluasi")
report_lines.append("")
report_lines.append("| Parameter | Value |")
report_lines.append("|-----------|-------|")
report_lines.append(f"| Device | {device} |")
report_lines.append(f"| Test Period | 2022-2025 |")
report_lines.append(f"| Nodes | {', '.join(NODE_NAMES)} |")
report_lines.append(f"| Ensemble Size | {NUM_ENSEMBLE} |")
report_lines.append(f"| Eval Step | {EVAL_STEP}h (daily) |")
report_lines.append(f"| DDIM Steps | 20 |")
report_lines.append(f"| Heavy Rain Threshold | {HEAVY_RAIN_THRESHOLD} mm/jam |")
report_lines.append(f"| Evaluation Time | {elapsed:.1f}s |")
report_lines.append("")
report_lines.append("## 2. Data Statistics")
report_lines.append("")
report_lines.append("| Node | Total Hours | Eval Samples | Valid | NaN Skipped |")
report_lines.append("|------|-------------|--------------|-------|-------------|")
for node, st in eval_stats.items():
    report_lines.append(f"| {node} | {st['total_hours']:,} | {st['eval_attempted']:,} | {st['valid_samples']:,} | {st['nan_skipped']} |")
report_lines.append(f"| **Total** | **{total_hours_all:,}** | **{sum(s['eval_attempted'] for s in eval_stats.values()):,}** | **{total_valid:,}** | **{sum(s['nan_skipped'] for s in eval_stats.values())}** |")
report_lines.append("")

# Per-node tables
report_lines.append("## 3. Metrik Per Node")
report_lines.append("")
for node in all_results:
    report_lines.append(f"### {node}")
    report_lines.append("")
    report_lines.append("| Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |")
    report_lines.append("|----------|------|-----|------|------|-------|-----|-----|-----|")
    for var in VARIABLES:
        if var not in all_results[node]:
            continue
        m = all_results[node][var]
        def fmt(v):
            return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
        report_lines.append(f"| {var} | {fmt(m['rmse'])} | {fmt(m['mae'])} | {fmt(m['correlation'])} | {fmt(m['crps'])} | {fmt(m['brier_score'])} | {fmt(m['pod'])} | {fmt(m['far'])} | {fmt(m['csi'])} |")
    report_lines.append("")

# Aggregated table
report_lines.append("## 4. Metrik Agregasi (Semua Node)")
report_lines.append("")
report_lines.append("| Variable | RMSE | MAE | Corr | CRPS | Brier | POD | FAR | CSI |")
report_lines.append("|----------|------|-----|------|------|-------|-----|-----|-----|")
for var in VARIABLES:
    if var not in aggregated_results:
        continue
    m = aggregated_results[var]
    def fmt(v):
        return f"{v:.4f}" if v is not None and not np.isnan(v) else "N/A"
    report_lines.append(f"| {var} | {fmt(m['rmse'])} | {fmt(m['mae'])} | {fmt(m['correlation'])} | {fmt(m['crps'])} | {fmt(m['brier_score'])} | {fmt(m['pod'])} | {fmt(m['far'])} | {fmt(m['csi'])} |")
report_lines.append("")

# Interpretation
report_lines.append("## 5. Interpretasi")
report_lines.append("")
report_lines.append("### Metrik Deterministik")
report_lines.append("- **RMSE** (Root Mean Square Error): Semakin rendah semakin baik. Sensitif terhadap outlier.")
report_lines.append("- **MAE** (Mean Absolute Error): Semakin rendah semakin baik. Robust terhadap outlier.")
report_lines.append("- **Corr** (Pearson Correlation): Mendekati 1.0 = model menangkap pola temporal dengan baik.")
report_lines.append("")
report_lines.append("### Metrik Probabilistik")
report_lines.append("- **CRPS**: Semakin rendah semakin baik. Mengukur kualitas distribusi prediksi.")
report_lines.append("- **Brier Score**: Semakin rendah semakin baik (0 = sempurna). Mengukur kalibrasi probabilitas event.")
report_lines.append("- **POD** (Probability of Detection): Mendekati 1.0 = model mendeteksi event dengan baik.")
report_lines.append("- **FAR** (False Alarm Ratio): Mendekati 0.0 = sedikit false alarm.")
report_lines.append("- **CSI** (Critical Success Index): Mendekati 1.0 = keseimbangan POD dan FAR.")
report_lines.append("")

# Files
report_lines.append("## 6. Output Files")
report_lines.append("")
report_lines.append("```")
report_lines.append("results/")
report_lines.append("  diffusion_results/diffusion_metrics.json")
report_lines.append("  probabilistic_metrics.json")
report_lines.append("  tables/metrics_summary.csv")
report_lines.append("  tables/metrics_summary.json")
report_lines.append("  plots/rain_prediction_vs_actual.png")
report_lines.append("  plots/wind_prediction_vs_actual.png")
report_lines.append("  plots/humidity_prediction_vs_actual.png")
report_lines.append("  plots/scatter_actual_vs_predicted.png")
report_lines.append("  plots/ensemble_spread.png")
report_lines.append("  plots/reliability_diagram.png")
report_lines.append("  evaluation_report.md")
report_lines.append("```")

with open("results/evaluation_report.md", 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"  Report: results/evaluation_report.md")

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print(f"\n{'='*80}")
print(f"  EVALUATION COMPLETE  ({elapsed:.1f}s)")
print(f"{'='*80}")
print(f"  Test Period:       2022-2025")
print(f"  Total Test Hours:  {total_hours_all:,}")
print(f"  Evaluated Samples: {total_valid:,}")
print(f"  NaN Skipped:       {sum(s['nan_skipped'] for s in eval_stats.values())}")
print(f"\n  All results saved to results/")
print(f"{'='*80}")
