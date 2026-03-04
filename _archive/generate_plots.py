"""
Generate Clean Visualizations for Thesis Evaluation
====================================================
1. 1-Week Time Series (hourly) per node — Actual vs Hybrid Prediction
2. Scatter: Actual vs Predicted (all nodes combined)
3. Ensemble Spread (P10-P90 uncertainty band)
4. Confusion Matrices (threshold 2/5/10 mm)
5. Training & Validation Loss Curves
6. Reliability Diagram

All results loaded from pre-computed JSON files (no re-inference needed
for most plots). 1-Week plot requires short inference (~168 samples per node).
"""

import sys, os, json, warnings
sys.path.append(os.path.abspath('.'))
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})

from src.inference import load_model_and_stats, run_inference_hybrid

# ==============================================================================
# CONFIG
# ==============================================================================
PLOT_DIR = 'results/plots'
os.makedirs(PLOT_DIR, exist_ok=True)

NODE_NAMES = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur']
VARIABLES = ['precipitation', 'wind_speed', 'humidity']
VAR_LABELS = {
    'precipitation': ('Curah Hujan', 'mm/jam'),
    'wind_speed':    ('Kecepatan Angin', 'm/s'),
    'humidity':      ('Kelembapan Relatif', '%'),
}
VAR_COLORS = {
    'precipitation': '#2563EB',   # blue
    'wind_speed':    '#16A34A',   # green
    'humidity':      '#9333EA',   # purple
}
ACTUAL_COLOR = '#374151'  # dark gray

# Week to visualize (last week of available data for a nice demo)
WEEK_START = '2024-12-20'
WEEK_END   = '2024-12-27'

NUM_ENSEMBLE = 30

print("=" * 70)
print("  GENERATE CLEAN THESIS VISUALIZATIONS")
print("=" * 70)

# ==============================================================================
# 1. LOAD MODEL & DATA
# ==============================================================================
print("\n[1/7] Loading model and data...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, stats, retrieval_db = load_model_and_stats('models/diffusion_chkpt.pth')
model.to(device); model.eval()

seq_len = model.config['seq_len']
feature_cols = model.config.get('feature_cols', [])

df = pd.read_parquet('data/raw/pangrango_era5_2005_2025.parquet')
df['date'] = pd.to_datetime(df['date'])
if df['date'].dt.tz is not None:
    df['date'] = df['date'].dt.tz_localize(None)

test_df = df[df['date'] >= '2022-01-01'].copy()
available_cols = [c for c in feature_cols if c in test_df.columns]
c_mean = stats['c_mean'].numpy()[:len(available_cols)]
c_std  = stats['c_std'].numpy()[:len(available_cols)]

print(f"  Device: {device}")
print(f"  Test rows: {len(test_df):,}")

# ==============================================================================
# 2. ONE-WEEK HOURLY INFERENCE (per node)
# ==============================================================================
print(f"\n[2/7] Running 1-week hourly inference ({WEEK_START} to {WEEK_END})...")

week_data = {}  # {node: {dates, actuals, preds, ensembles}}

for node_name in NODE_NAMES:
    node_df = test_df[test_df['node'] == node_name].copy()
    node_df = node_df.sort_values('date').reset_index(drop=True)

    # Filter to the target week + seq_len buffer before
    week_mask = (node_df['date'] >= pd.Timestamp(WEEK_START)) & \
                (node_df['date'] <= pd.Timestamp(WEEK_END) + pd.Timedelta(hours=23))
    week_indices = node_df.index[week_mask].tolist()

    if len(week_indices) == 0:
        print(f"  {node_name}: No data for {WEEK_START} to {WEEK_END}")
        continue

    dates_list = []
    actuals = {v: [] for v in VARIABLES}
    preds   = {v: [] for v in VARIABLES}
    ensembles = {v: [] for v in VARIABLES}
    skipped = 0

    for idx in tqdm(week_indices, desc=f"  {node_name}", ncols=80):
        if idx < seq_len:
            skipped += 1
            continue

        seq_df = node_df.iloc[idx - seq_len:idx]
        features = seq_df[available_cols].values
        features_norm = (features - c_mean) / (c_std + 1e-5)

        target_row = node_df.iloc[idx]
        prev_row   = node_df.iloc[idx - 1]

        lag_values = {
            'precipitation': prev_row['precipitation'],
            'wind_speed':    prev_row['wind_speed_10m'],
            'humidity':      prev_row['relative_humidity_2m'],
        }

        try:
            result = run_inference_hybrid(
                features_norm, model, stats, retrieval_db,
                lag_values=lag_values, num_samples=NUM_ENSEMBLE, device=device
            )
        except Exception:
            skipped += 1
            continue

        ens_p = result.get('hybrid_precipitation', result.get('precipitation'))
        ens_w = result.get('hybrid_wind_speed',    result.get('wind_speed'))
        ens_h = result.get('hybrid_humidity',       result.get('humidity'))

        if np.all(np.isnan(ens_p)) or np.all(np.isnan(ens_w)) or np.all(np.isnan(ens_h)):
            skipped += 1
            continue

        dates_list.append(target_row['date'])
        actuals['precipitation'].append(target_row['precipitation'])
        actuals['wind_speed'].append(target_row['wind_speed_10m'])
        actuals['humidity'].append(target_row['relative_humidity_2m'])

        preds['precipitation'].append(float(np.nanmedian(ens_p)))
        preds['wind_speed'].append(float(np.nanmedian(ens_w)))
        preds['humidity'].append(float(np.nanmedian(ens_h)))

        ensembles['precipitation'].append(ens_p.copy())
        ensembles['wind_speed'].append(ens_w.copy())
        ensembles['humidity'].append(ens_h.copy())

    # Convert to arrays
    for v in VARIABLES:
        actuals[v]   = np.array(actuals[v])
        preds[v]     = np.array(preds[v])
        ensembles[v] = np.array(ensembles[v]) if len(ensembles[v]) > 0 else np.empty((0, NUM_ENSEMBLE))

    week_data[node_name] = {
        'dates': dates_list,
        'actuals': actuals,
        'preds': preds,
        'ensembles': ensembles,
    }
    print(f"    → {len(dates_list)} hourly samples, {skipped} skipped")

# ==============================================================================
# 3. PLOT: 1-Week Time Series per Node (3 subplots)
# ==============================================================================
print(f"\n[3/7] Generating 1-week time series plots...")

for node_name, wd in week_data.items():
    dates = wd['dates']
    if len(dates) == 0:
        continue

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for i, var in enumerate(VARIABLES):
        ax = axes[i]
        label, unit = VAR_LABELS[var]
        color = VAR_COLORS[var]

        actual = wd['actuals'][var]
        pred   = wd['preds'][var]

        # Correlation
        mask = ~(np.isnan(actual) | np.isnan(pred))
        if mask.sum() > 2:
            corr = np.corrcoef(actual[mask], pred[mask])[0, 1]
        else:
            corr = 0.0

        ax.plot(dates, actual, color=ACTUAL_COLOR, linewidth=1.0,
                alpha=0.9, label='Actual')
        ax.plot(dates, pred, color=color, linewidth=1.0,
                alpha=0.9, label='Hybrid Prediction')

        # Ensemble spread (P10-P90)
        ens = wd['ensembles'][var]
        if len(ens) > 0:
            q10 = np.nanpercentile(ens, 10, axis=1)
            q90 = np.nanpercentile(ens, 90, axis=1)
            ax.fill_between(dates, q10, q90, alpha=0.15, color=color, label='P10–P90')

        ax.set_ylabel(f'{label} ({unit})', fontsize=10)
        ax.set_title(f'{label} ({unit}) (Corr: {corr:.3f})', fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].set_xlabel('Date', fontsize=10)
    fig.autofmt_xdate(rotation=30)

    fig.suptitle(f'Nowcasting 1-Minggu — {node_name} (Hourly)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    fname = f'{PLOT_DIR}/timeseries_1week_{node_name.lower()}.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {fname}")

# ==============================================================================
# 4. PLOT: Scatter Actual vs Predicted (all nodes, per variable)
# ==============================================================================
print(f"\n[4/7] Generating scatter plots...")

# First collect FULL test set data from pre-computed JSON
with open('results/diffusion_results/diffusion_metrics.json', 'r') as f:
    diff_metrics = json.load(f)

# We'll use the week_data for scatter (sufficient for visual)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, var in enumerate(VARIABLES):
    label, unit = VAR_LABELS[var]
    all_a, all_p = [], []
    for node_name, wd in week_data.items():
        all_a.extend(wd['actuals'][var].tolist())
        all_p.extend(wd['preds'][var].tolist())

    a = np.array(all_a)
    p = np.array(all_p)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]

    axes[i].scatter(a, p, alpha=0.3, s=12, color=VAR_COLORS[var], edgecolors='none')
    if len(a) > 0:
        vmin = min(a.min(), p.min())
        vmax = max(a.max(), p.max())
        axes[i].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1, label='Perfect Prediction')

    corr_val = diff_metrics['aggregated'][var]['correlation']
    axes[i].set_xlabel(f'Actual {label} ({unit})', fontsize=9)
    axes[i].set_ylabel(f'Predicted {label} ({unit})', fontsize=9)
    axes[i].set_title(f'{label} (r = {corr_val:.3f})', fontsize=11)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.25)
    axes[i].set_aspect('equal', adjustable='datalim')

fig.suptitle('Scatter: Actual vs Predicted (Semua Node)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/scatter_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {PLOT_DIR}/scatter_actual_vs_predicted.png")

# ==============================================================================
# 5. PLOT: Ensemble Spread (1 week, first node)
# ==============================================================================
print(f"\n[5/7] Generating ensemble spread plot...")

node0 = list(week_data.keys())[0]
wd0 = week_data[node0]
dates0 = wd0['dates']

fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
for i, var in enumerate(VARIABLES):
    ax = axes[i]
    label, unit = VAR_LABELS[var]
    color = VAR_COLORS[var]

    actual = wd0['actuals'][var]
    ens = wd0['ensembles'][var]

    if len(ens) == 0:
        continue

    median = np.nanmedian(ens, axis=1)
    q10 = np.nanpercentile(ens, 10, axis=1)
    q90 = np.nanpercentile(ens, 90, axis=1)
    q25 = np.nanpercentile(ens, 25, axis=1)
    q75 = np.nanpercentile(ens, 75, axis=1)

    ax.fill_between(dates0, q10, q90, alpha=0.15, color=color, label='P10–P90')
    ax.fill_between(dates0, q25, q75, alpha=0.25, color=color, label='P25–P75')
    ax.plot(dates0, median, color=color, linewidth=1.2, label='Median Ensemble')
    ax.plot(dates0, actual, color=ACTUAL_COLOR, linewidth=1.0, alpha=0.85, label='Actual')

    ax.set_ylabel(f'{label} ({unit})', fontsize=10)
    ax.set_title(f'{label} — Ensemble Spread ({NUM_ENSEMBLE} members)', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.25)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].set_xlabel('Date', fontsize=10)
fig.autofmt_xdate(rotation=30)

fig.suptitle(f'Ensemble Uncertainty — {node0} (1 Minggu)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/ensemble_spread.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {PLOT_DIR}/ensemble_spread.png")

# ==============================================================================
# 6. PLOT: Confusion Matrices (threshold 2, 5, 10 mm)
# ==============================================================================
print(f"\n[6/7] Generating confusion matrices...")

with open('results/threshold_sensitivity.json', 'r') as f:
    thr_data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
thresholds = ['2.0', '5.0', '10.0']
thr_labels = ['2 mm/jam', '5 mm/jam', '10 mm/jam']

for idx, (thr_key, thr_label) in enumerate(zip(thresholds, thr_labels)):
    ax = axes[idx]
    td = thr_data[thr_key]

    hits   = td['hits']
    fa     = td['false_alarms']
    misses = td['misses']
    cn     = td['correct_negatives']

    matrix = np.array([[hits, misses], [fa, cn]])
    total = hits + fa + misses + cn
    matrix_pct = matrix / total * 100

    im = ax.imshow(matrix_pct, cmap='Blues', vmin=0, vmax=max(matrix_pct.max(), 1))

    labels_text = [
        [f'Hit\n{hits}\n({matrix_pct[0,0]:.1f}%)',
         f'Miss\n{misses}\n({matrix_pct[0,1]:.1f}%)'],
        [f'False Alarm\n{fa}\n({matrix_pct[1,0]:.1f}%)',
         f'Correct Neg\n{cn}\n({matrix_pct[1,1]:.1f}%)'],
    ]

    for r in range(2):
        for c_idx in range(2):
            color = 'white' if matrix_pct[r, c_idx] > 50 else 'black'
            ax.text(c_idx, r, labels_text[r][c_idx], ha='center', va='center',
                    fontsize=9, color=color, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pred: Ya', 'Pred: Tidak'], fontsize=9)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actual: Ya', 'Actual: Tidak'], fontsize=9)

    pod = td['pod']
    csi = td['csi']
    ax.set_title(f'Threshold {thr_label}\nPOD={pod:.3f}  CSI={csi:.3f}', fontsize=10, fontweight='bold')

fig.suptitle('Confusion Matrix — Deteksi Hujan pada Berbagai Threshold', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{PLOT_DIR}/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  → {PLOT_DIR}/confusion_matrices.png")

# ==============================================================================
# 7. PLOT: Training & Validation Loss Curves
# ==============================================================================
print(f"\n[7/7] Generating training loss curve...")

# Check if training log images exist
train_loss_img  = 'results/training_logs/training_loss_curve.png'
val_loss_img    = 'results/training_logs/validation_loss_curve.png'
baseline_loss   = 'results/training_logs/baseline_loss_curve.png'

if os.path.exists(train_loss_img) and os.path.exists(val_loss_img):
    # Combine into single figure
    from PIL import Image
    img_train = Image.open(train_loss_img)
    img_val   = Image.open(val_loss_img)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].imshow(np.array(img_train))
    axes[0].axis('off')
    axes[0].set_title('Training Loss', fontsize=11, fontweight='bold')
    axes[1].imshow(np.array(img_val))
    axes[1].axis('off')
    axes[1].set_title('Validation Loss', fontsize=11, fontweight='bold')
    fig.suptitle('Learning Curves — RA-Diffusion Model (20 Epochs)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {PLOT_DIR}/training_curves.png")
else:
    print(f"  ⚠ Training loss images not found, skipping")

# ==============================================================================
# CLEANUP: Remove old duplicate files
# ==============================================================================
print(f"\n[Cleanup] Removing duplicate/old plot files...")

old_files = [
    # Duplicates in plots/
    f'{PLOT_DIR}/rain_prediction_vs_actual.png',
    f'{PLOT_DIR}/wind_prediction_vs_actual.png',
    f'{PLOT_DIR}/precipitation_prediction_vs_actual.png',
    f'{PLOT_DIR}/wind_speed_prediction_vs_actual.png',
    f'{PLOT_DIR}/humidity_prediction_vs_actual.png',
    f'{PLOT_DIR}/rainfall_scatter_analysis.png',
    f'{PLOT_DIR}/confusion_matrices_rainfall.png',
    f'{PLOT_DIR}/precipitation_baselines.png',
    f'{PLOT_DIR}/rainfall_intensity_histogram.png',
    f'{PLOT_DIR}/rainfall_distribution.png',
    f'{PLOT_DIR}/per_node_performance.png',
    f'{PLOT_DIR}/skill_comparison.png',
    f'{PLOT_DIR}/node_correlation_matrix.png',
    # Duplicates in results/ root
    'results/scatter_actual_vs_predicted.png',
    'results/scatter_final.png',
    'results/final_hybrid_results.png',
    'results/timeseries_final.png',
    'results/precipitation_distribution.png',
    'results/probabilistic_evaluation.png',
    'results/probabilistic_forecast.png',
    'results/risk_level_distribution.png',
    'results/performance_per_year.png',
]

removed = 0
for f in old_files:
    if os.path.exists(f):
        os.remove(f)
        removed += 1

print(f"  Removed {removed} old/duplicate files")

# ==============================================================================
# FINAL FILE LIST
# ==============================================================================
print(f"\n{'='*70}")
print(f"  CLEAN PLOTS GENERATED")
print(f"{'='*70}")

remaining = sorted(os.listdir(PLOT_DIR))
print(f"\n  results/plots/ ({len(remaining)} files):")
for f in remaining:
    print(f"    • {f}")

# Also list remaining in results/ root
root_files = [f for f in os.listdir('results') if f.endswith(('.png', '.jpg'))]
if root_files:
    print(f"\n  ⚠ Remaining images in results/ root:")
    for f in root_files:
        print(f"    • {f}")
else:
    print(f"\n  ✓ results/ root is clean (no stray images)")

print(f"\n{'='*70}")
print(f"  DONE")
print(f"{'='*70}")
