#!/usr/bin/env python
"""
Advanced Scientific Analysis of Evaluation Results
===================================================
Analyzes model performance, rainfall distribution, cross-node correlation,
ERA5 grid effects, and persistence baseline comparison.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from math import radians, sin, cos, sqrt, atan2

warnings.filterwarnings('ignore')

# ── paths ──────────────────────────────────────────────────────────────
DATA_PATH   = "data/raw/pangrango_era5_2005_2025.parquet"
METRICS_PATH = "results/diffusion_results/diffusion_metrics.json"
PLOT_DIR    = "results/plots"
REPORT_PATH = "results/model_analysis_report.md"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── node coordinates (from ingest.py) ─────────────────────────────────
NODE_COORDS = {
    'Puncak':        (-6.769797, 106.963583),
    'Lereng_Cibodas': (-6.751722, 106.987160),
    'Hilir_Cianjur': (-6.816000, 107.133000),
}
NODES = list(NODE_COORDS.keys())

# ── load data ─────────────────────────────────────────────────────────
print("=" * 70)
print("  ADVANCED MODEL ANALYSIS")
print("=" * 70)

print("\n[1/7] Loading data...")
df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
print(f"  Dataset: {len(df):,} rows, {df['node'].nunique()} nodes")
print(f"  Period : {df['date'].min()} → {df['date'].max()}")

with open(METRICS_PATH) as f:
    metrics = json.load(f)

# ── split boundaries ──────────────────────────────────────────────────
TEST_START = pd.Timestamp("2022-01-01", tz="UTC") if df['date'].dt.tz is not None else pd.Timestamp("2022-01-01")
df_test = df[df['date'] >= TEST_START].copy()

# ======================================================================
# 1. RAINFALL DISTRIBUTION ANALYSIS
# ======================================================================
print("\n[2/7] Rainfall distribution analysis...")

rain = df['precipitation'].dropna()
rain_test = df_test['precipitation'].dropna()

percentiles = [50, 75, 90, 95, 99, 99.5, 99.9]
pct_values = np.percentile(rain, percentiles)

pct_gt_2  = (rain > 2).mean() * 100
pct_gt_5  = (rain > 5).mean() * 100
pct_gt_10 = (rain > 10).mean() * 100
pct_gt_20 = (rain > 20).mean() * 100
pct_zero  = (rain == 0).mean() * 100

pct_gt_2_test  = (rain_test > 2).mean() * 100
pct_gt_5_test  = (rain_test > 5).mean() * 100
pct_gt_10_test = (rain_test > 10).mean() * 100

print(f"  Total rain samples     : {len(rain):,}")
print(f"  Zero rainfall          : {pct_zero:.1f}%")
print(f"  > 2 mm/jam             : {pct_gt_2:.2f}%")
print(f"  > 5 mm/jam             : {pct_gt_5:.2f}%")
print(f"  > 10 mm/jam (threshold): {pct_gt_10:.3f}%")
print(f"  > 20 mm/jam            : {pct_gt_20:.4f}%")
print(f"  Max rainfall           : {rain.max():.1f} mm/jam")
print(f"\n  Percentiles:")
for p, v in zip(percentiles, pct_values):
    print(f"    P{p:5.1f} = {v:.3f} mm/jam")

# ── Rainfall distribution plot ────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Rainfall Distribution Analysis – ERA5 Pangrango", fontsize=14, fontweight='bold')

# (a) Histogram all rain
ax = axes[0, 0]
rain_nonzero = rain[rain > 0]
ax.hist(rain_nonzero, bins=100, color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.3)
ax.set_xlabel("Precipitation (mm/jam)")
ax.set_ylabel("Count")
ax.set_title(f"(a) Histogram – non-zero rain (n={len(rain_nonzero):,})")
ax.axvline(10, color='red', ls='--', lw=1.5, label='Threshold 10 mm')
ax.axvline(5, color='orange', ls='--', lw=1.5, label='5 mm')
ax.axvline(2, color='green', ls='--', lw=1.5, label='2 mm')
ax.legend(fontsize=8)
ax.set_yscale('log')

# (b) CDF
ax = axes[0, 1]
sorted_rain = np.sort(rain)
cdf = np.arange(1, len(sorted_rain) + 1) / len(sorted_rain)
ax.plot(sorted_rain, cdf, color='steelblue', lw=1.5)
ax.set_xlabel("Precipitation (mm/jam)")
ax.set_ylabel("Cumulative probability")
ax.set_title("(b) CDF – all samples")
ax.axvline(10, color='red', ls='--', lw=1.5, label=f'>10mm: {pct_gt_10:.2f}%')
ax.axvline(5, color='orange', ls='--', lw=1.5, label=f'>5mm: {pct_gt_5:.1f}%')
ax.axvline(2, color='green', ls='--', lw=1.5, label=f'>2mm: {pct_gt_2:.1f}%')
ax.legend(fontsize=8, loc='center right')
ax.set_xlim(-0.5, 30)

# (c) Bar – threshold exceedance
ax = axes[1, 0]
thresholds = [0, 0.1, 1, 2, 5, 10, 20]
labels = ['=0', '>0.1', '>1', '>2', '>5', '>10', '>20']
pcts = [(rain == 0).mean()*100, (rain>0.1).mean()*100, (rain>1).mean()*100,
        (rain>2).mean()*100, (rain>5).mean()*100, (rain>10).mean()*100, (rain>20).mean()*100]
colors = ['gray', '#91bfdb', '#4575b4', '#fee090', '#fc8d59', '#d73027', '#a50026']
bars = ax.bar(labels, pcts, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel("% of total samples")
ax.set_xlabel("Precipitation threshold (mm/jam)")
ax.set_title("(c) Threshold Exceedance")
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)

# (d) Per-node distribution box
ax = axes[1, 1]
node_rain = [df[df['node'] == n]['precipitation'].dropna().values for n in NODES]
bp = ax.boxplot(node_rain, labels=NODES, patch_artist=True, showfliers=False)
for patch, c in zip(bp['boxes'], ['#4575b4', '#91bfdb', '#fc8d59']):
    patch.set_facecolor(c)
ax.set_ylabel("Precipitation (mm/jam)")
ax.set_title("(d) Per-node box (outliers hidden)")

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rainfall_distribution.png"), dpi=150)
plt.close()
print(f"  → Plot saved: results/plots/rainfall_distribution.png")

# ======================================================================
# 2. CROSS-NODE CORRELATION ANALYSIS
# ======================================================================
print("\n[3/7] Cross-node correlation analysis...")

vars_map = {
    'precipitation': 'precipitation',
    'wind_speed':    'wind_speed_10m',
    'humidity':      'relative_humidity_2m',
}

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cross-Node Correlation Matrix – ERA5 Pangrango", fontsize=13, fontweight='bold')

corr_results = {}
for vi, (var_label, col) in enumerate(vars_map.items()):
    # Pivot to wide: each node as column
    piv = df.pivot_table(index='date', columns='node', values=col).dropna()
    corr_mat = piv[NODES].corr()
    corr_results[var_label] = corr_mat.to_dict()

    ax = axes[vi]
    im = ax.imshow(corr_mat.values, vmin=0.0, vmax=1.0, cmap='RdYlBu_r')
    ax.set_xticks(range(len(NODES)))
    ax.set_yticks(range(len(NODES)))
    ax.set_xticklabels([n[:8] for n in NODES], fontsize=9, rotation=30, ha='right')
    ax.set_yticklabels([n[:8] for n in NODES], fontsize=9)
    ax.set_title(var_label, fontsize=11)
    for i in range(len(NODES)):
        for j in range(len(NODES)):
            ax.text(j, i, f"{corr_mat.values[i, j]:.3f}",
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    color='white' if corr_mat.values[i, j] > 0.7 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "node_correlation_matrix.png"), dpi=150)
plt.close()
print(f"  → Plot saved: results/plots/node_correlation_matrix.png")

# Print correlation summary
for var_label in vars_map:
    mat = corr_results[var_label]
    pairs = []
    for i, n1 in enumerate(NODES):
        for j, n2 in enumerate(NODES):
            if i < j:
                pairs.append((n1[:8], n2[:8], mat[n1][n2]))
    print(f"  {var_label}:")
    for a, b, r in pairs:
        print(f"    {a} ↔ {b}: r = {r:.4f}")

# ======================================================================
# 3. ERA5 GRID ANALYSIS
# ======================================================================
print("\n[4/7] ERA5 grid resolution analysis...")

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    φ1, φ2 = radians(lat1), radians(lat2)
    Δφ = radians(lat2 - lat1)
    Δλ = radians(lon2 - lon1)
    a = sin(Δφ/2)**2 + cos(φ1)*cos(φ2)*sin(Δλ/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

print(f"  Node coordinates:")
for name, (lat, lon) in NODE_COORDS.items():
    print(f"    {name:18s}: ({lat:.6f}, {lon:.6f})")

print(f"\n  Pairwise distances:")
dist_results = {}
for i, n1 in enumerate(NODES):
    for j, n2 in enumerate(NODES):
        if i < j:
            d = haversine(*NODE_COORDS[n1], *NODE_COORDS[n2])
            dist_results[(n1, n2)] = d
            print(f"    {n1:18s} ↔ {n2:18s}: {d:.2f} km")

# ERA5 grid resolution: 0.25° ≈ 25-28 km at equator
# At lat ~6.8°S: cos(6.8°) ≈ 0.993 → grid ≈ 27.7 km E-W, 27.8 km N-S
era5_grid_km = 0.25 * 111.32  # approx at equator
lat_avg = np.mean([c[0] for c in NODE_COORDS.values()])
era5_ew = 0.25 * 111.32 * cos(radians(abs(lat_avg)))
era5_ns = 0.25 * 110.574
print(f"\n  ERA5 grid resolution at lat {lat_avg:.2f}°:")
print(f"    East-West : {era5_ew:.1f} km")
print(f"    North-South: {era5_ns:.1f} km")

max_dist = max(dist_results.values())
same_grid = all(d < era5_ew for d in dist_results.values())
print(f"\n  Max inter-node distance: {max_dist:.2f} km")
print(f"  ERA5 grid spacing      : ~{era5_ew:.1f} km")

# Check which ERA5 grid cells each node falls in
grid_cells = {}
for name, (lat, lon) in NODE_COORDS.items():
    # ERA5 grid cell = floor to nearest 0.25
    g_lat = round(lat / 0.25) * 0.25
    g_lon = round(lon / 0.25) * 0.25
    grid_cells[name] = (g_lat, g_lon)
    print(f"    {name:18s} → ERA5 grid cell: ({g_lat:.2f}, {g_lon:.2f})")

unique_cells = set(grid_cells.values())
print(f"\n  Unique ERA5 grid cells: {len(unique_cells)}")
if len(unique_cells) < len(NODES):
    shared = {}
    for name, cell in grid_cells.items():
        shared.setdefault(cell, []).append(name)
    for cell, names in shared.items():
        if len(names) > 1:
            print(f"    ⚠ Cell {cell}: shared by {', '.join(names)}")

# ======================================================================
# 4. PERSISTENCE BASELINE COMPARISON
# ======================================================================
print("\n[5/7] Persistence baseline comparison...")

# Persistence forecast = use value from 24h ago (same EVAL_STEP)
EVAL_STEP = 24
target_map = {
    'precipitation': 'precipitation',
    'wind_speed': 'wind_speed_10m',
    'humidity': 'relative_humidity_2m',
}

persistence_metrics = {}
for node in NODES:
    node_df = df_test[df_test['node'] == node].sort_values('date').reset_index(drop=True)
    node_results = {}

    for var_label, col in target_map.items():
        actual = node_df[col].values
        # Persistence = value EVAL_STEP hours ago
        persist = np.full_like(actual, np.nan)
        persist[EVAL_STEP:] = actual[:-EVAL_STEP]

        mask = ~(np.isnan(actual) | np.isnan(persist))
        a = actual[mask]
        p = persist[mask]

        if len(a) > 10:
            rmse_p = np.sqrt(np.mean((a - p)**2))
            mae_p  = np.mean(np.abs(a - p))
            if np.std(a) > 0 and np.std(p) > 0:
                corr_p = np.corrcoef(a, p)[0, 1]
            else:
                corr_p = 0.0
            node_results[var_label] = {'rmse': rmse_p, 'mae': mae_p, 'corr': corr_p, 'n': len(a)}

    persistence_metrics[node] = node_results

# Compute aggregated persistence
agg_persist = {}
for var_label in target_map:
    rmses, maes, corrs, ns = [], [], [], []
    for node in NODES:
        if var_label in persistence_metrics[node]:
            m = persistence_metrics[node][var_label]
            rmses.append(m['rmse'])
            maes.append(m['mae'])
            corrs.append(m['corr'])
            ns.append(m['n'])
    agg_persist[var_label] = {
        'rmse': np.mean(rmses), 'mae': np.mean(maes), 'corr': np.mean(corrs)
    }

# Model metrics (aggregated)
model_agg = metrics['aggregated']

print(f"\n  {'Variable':<16} {'Metric':<8} {'Persistence':<12} {'Model':<12} {'Improvement':<12}")
print("  " + "-" * 62)

skill_scores = {}
for var_label in target_map:
    pm = agg_persist[var_label]
    mm = model_agg[var_label]
    skill_scores[var_label] = {}

    for met in ['rmse', 'mae']:
        pv = pm[met]
        mv = mm[met]
        improvement = (1 - mv / pv) * 100 if pv > 0 else 0
        skill_scores[var_label][met] = improvement
        print(f"  {var_label:<16} {met.upper():<8} {pv:<12.4f} {mv:<12.4f} {improvement:+.1f}%")

    # Correlation (higher = better)
    pv = pm['corr']
    mv = mm['correlation']
    skill_scores[var_label]['corr'] = mv - pv
    print(f"  {var_label:<16} {'Corr':<8} {pv:<12.4f} {mv:<12.4f} {mv - pv:+.4f}")
    print()

# ======================================================================
# 5. CLIMATOLOGY BASELINE
# ======================================================================
print("[6/7] Climatology baseline...")

# Climatology = mean value per variable in training set
TRAIN_END = pd.Timestamp("2018-12-31", tz="UTC") if df['date'].dt.tz is not None else pd.Timestamp("2018-12-31")
df_train = df[df['date'] <= TRAIN_END]

clim_metrics = {}
for var_label, col in target_map.items():
    clim_val = df_train[col].mean()
    actuals_all = []

    for node in NODES:
        node_test = df_test[df_test['node'] == node]
        actual = node_test[col].dropna().values
        actuals_all.extend(actual)

    a = np.array(actuals_all)
    clim_rmse = np.sqrt(np.mean((a - clim_val)**2))
    clim_mae  = np.mean(np.abs(a - clim_val))
    clim_metrics[var_label] = {'rmse': clim_rmse, 'mae': clim_mae, 'clim_value': clim_val}

print(f"\n  {'Variable':<16} {'Clim RMSE':<12} {'Model RMSE':<12} {'SS (RMSE)':<12}")
print("  " + "-" * 50)
for var_label in target_map:
    cr = clim_metrics[var_label]['rmse']
    mr = model_agg[var_label]['rmse']
    ss = (1 - (mr / cr)**2) * 100  # Skill Score
    print(f"  {var_label:<16} {cr:<12.4f} {mr:<12.4f} {ss:+.1f}%")

# ======================================================================
# 6. ADDITIONAL PLOTS
# ======================================================================
print("\n[6b/7] Generating additional plots...")

# ── Skill comparison bar chart ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Model vs Persistence vs Climatology – RMSE Comparison", fontsize=13, fontweight='bold')

for vi, var_label in enumerate(target_map):
    ax = axes[vi]
    p_rmse = agg_persist[var_label]['rmse']
    c_rmse = clim_metrics[var_label]['rmse']
    m_rmse = model_agg[var_label]['rmse']

    x = ['Climatology', 'Persistence\n(24h lag)', 'RA-Diffusion\n(Ours)']
    vals = [c_rmse, p_rmse, m_rmse]
    colors = ['#bdbdbd', '#fdae61', '#2166ac']
    bars = ax.bar(x, vals, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    ax.set_ylabel("RMSE")
    ax.set_title(var_label.replace('_', ' ').title())
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, max(vals) * 1.25)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "skill_comparison.png"), dpi=150)
plt.close()
print(f"  → Plot saved: results/plots/skill_comparison.png")

# ── Per-node model detail ─────────────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Per-Node Performance Metrics", fontsize=14, fontweight='bold')

metric_names = ['rmse', 'mae', 'correlation']
metric_labels = ['RMSE (↓)', 'MAE (↓)', 'Correlation (↑)']
var_labels = list(target_map.keys())

for vi, var_label in enumerate(var_labels):
    for mi, (met_name, met_label) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[vi, mi]
        vals = [metrics['per_node'][n][var_label][met_name] for n in NODES]
        colors = ['#4575b4', '#91bfdb', '#fc8d59']
        bars = ax.bar(NODES, vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(f"{var_label} – {met_label}", fontsize=10)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8)
        ax.tick_params(axis='x', labelsize=8, rotation=15)

plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "per_node_performance.png"), dpi=150)
plt.close()
print(f"  → Plot saved: results/plots/per_node_performance.png")

# ======================================================================
# 7. GENERATE REPORT
# ======================================================================
print("\n[7/7] Generating analysis report...")

report_lines = []
def W(s=""): report_lines.append(s)

W("# Model Analysis Report")
W()
W("## Retrieval-Augmented Diffusion Model – Gunung Gede-Pangrango")
W("### Analisis Lanjutan Hasil Evaluasi")
W()
W("---")
W()

# ── Section 1: Metric Interpretation ──────────────────────────────────
W("## 1. Interpretasi Metrik Model")
W()
W("### 1.1 Precipitation")
W()
W("| Metric | Puncak | Lereng Cibodas | Hilir Cianjur | Aggregated |")
W("|--------|--------|----------------|---------------|------------|")
for met in ['rmse', 'mae', 'correlation', 'crps', 'brier_score', 'pod', 'far', 'csi']:
    vals = []
    for n in NODES:
        vals.append(f"{metrics['per_node'][n]['precipitation'][met]:.4f}")
    vals.append(f"{model_agg['precipitation'][met]:.4f}")
    W(f"| {met.upper()} | {' | '.join(vals)} |")
W()
W("**Temuan Utama – Precipitation:**")
W()
W(f"- POD sangat rendah (agregasi: {model_agg['precipitation']['pod']:.4f}), bahkan 0.0000 pada Puncak dan Lereng Cibodas.")
W(f"- FAR sangat tinggi ({model_agg['precipitation']['far']:.4f}), artinya hampir semua prediksi heavy rain adalah false alarm.")
W(f"- CSI mendekati nol ({model_agg['precipitation']['csi']:.4f}), menunjukkan model gagal total mendeteksi heavy rain event.")
W(f"- Namun RMSE ({model_agg['precipitation']['rmse']:.4f}) dan Correlation ({model_agg['precipitation']['correlation']:.4f}) cukup reasonable.")
W(f"- Brier Score sangat rendah ({model_agg['precipitation']['brier_score']:.4f}) — ini misleading karena event ~sangat jarang~.")
W()
W("**Penjelasan Ilmiah:**")
W("Model memprediksi dengan baik nilai-nilai kecil/nol rainfall (mayoritas data),")
W("tetapi gagal menangkap extreme events karena threshold 10 mm/jam terlalu tinggi ")
W("relatif terhadap distribusi data ERA5 (lihat Bagian 2).")
W()

W("### 1.2 Wind Speed")
W()
W("| Metric | Puncak | Lereng Cibodas | Hilir Cianjur | Aggregated |")
W("|--------|--------|----------------|---------------|------------|")
for met in ['rmse', 'mae', 'correlation', 'crps', 'brier_score', 'pod', 'far', 'csi']:
    vals = []
    for n in NODES:
        vals.append(f"{metrics['per_node'][n]['wind_speed'][met]:.4f}")
    vals.append(f"{model_agg['wind_speed'][met]:.4f}")
    W(f"| {met.upper()} | {' | '.join(vals)} |")
W()
W("**Temuan Utama – Wind Speed:**")
W()
W(f"- Correlation tinggi ({model_agg['wind_speed']['correlation']:.4f}), model menangkap pola diurnal angin.")
W(f"- POD moderat (~0.50), model mendeteksi separuh event angin kencang.")
W(f"- FAR rendah ({model_agg['wind_speed']['far']:.4f}), sedikit false alarm.")
W(f"- CSI terbaik ({model_agg['wind_speed']['csi']:.4f}) dibandingkan variabel lain.")
W(f"- Performa konsisten di Puncak & Lereng Cibodas; sedikit lebih rendah di Hilir Cianjur (RMSE {metrics['per_node']['Hilir_Cianjur']['wind_speed']['rmse']:.2f} vs ~1.58).")
W()

W("### 1.3 Humidity")
W()
W("| Metric | Puncak | Lereng Cibodas | Hilir Cianjur | Aggregated |")
W("|--------|--------|----------------|---------------|------------|")
for met in ['rmse', 'mae', 'correlation', 'crps', 'brier_score', 'pod', 'far', 'csi']:
    vals = []
    for n in NODES:
        vals.append(f"{metrics['per_node'][n]['humidity'][met]:.4f}")
    vals.append(f"{model_agg['humidity'][met]:.4f}")
    W(f"| {met.upper()} | {' | '.join(vals)} |")
W()
W("**Temuan Utama – Humidity:**")
W()
W(f"- Correlation tertinggi ({model_agg['humidity']['correlation']:.4f}), variabel paling mudah diprediksi.")
W(f"- RMSE ~6.7% RH tergolong baik untuk nowcasting.")
W(f"- POD ({model_agg['humidity']['pod']:.4f}) dan CSI ({model_agg['humidity']['csi']:.4f}) moderat untuk threshold >90% RH.")
W(f"- Lereng Cibodas memiliki performa terbaik (RMSE {metrics['per_node']['Lereng_Cibodas']['humidity']['rmse']:.2f}).")
W()

# ── Section 2: Rainfall Distribution ──────────────────────────────────
W("## 2. Analisis Distribusi Rainfall")
W()
W("### 2.1 Statistik Distribusi")
W()
W(f"Dataset ERA5 reanalysis memiliki distribusi rainfall yang **sangat skewed**:")
W()
W(f"| Statistik | Nilai |")
W(f"|-----------|-------|")
W(f"| Total sampel | {len(rain):,} |")
W(f"| Zero rainfall | {pct_zero:.1f}% |")
W(f"| > 0.1 mm/jam | {(rain>0.1).mean()*100:.1f}% |")
W(f"| > 2 mm/jam | {pct_gt_2:.2f}% |")
W(f"| > 5 mm/jam | {pct_gt_5:.2f}% |")
W(f"| **> 10 mm/jam (threshold)** | **{pct_gt_10:.3f}%** |")
W(f"| > 20 mm/jam | {pct_gt_20:.4f}% |")
W(f"| Max | {rain.max():.1f} mm/jam |")
W()
W("### 2.2 Percentiles")
W()
W("| Percentile | Nilai (mm/jam) |")
W("|------------|----------------|")
for p, v in zip(percentiles, pct_values):
    W(f"| P{p} | {v:.3f} |")
W()
W("### 2.3 Implikasi untuk POD/CSI")
W()
W(f"Dengan threshold heavy rain = 10 mm/jam, hanya **{pct_gt_10:.3f}%** data yang termasuk 'event'.")
W("Ini berarti:")
W(f"- Dalam test set (~{len(rain_test):,} sampel), diperkirakan hanya **~{int(len(rain_test) * pct_gt_10 / 100)} jam** dengan heavy rain.")
W(f"- Dengan eval_step=24h (subsampling), kemungkinan model hanya melihat **sangat sedikit** event aktual.")
W("- Model cenderung prediksi 'tidak hujan lebat' (climatological bias) → POD rendah.")
W("- Beberapa false alarm terjadi → FAR sangat tinggi karena denominator kecil.")
W()
W("**Rekomendasi:**")
W("- Gunakan threshold **2 mm/jam** atau **5 mm/jam** untuk evaluasi yang lebih meaningful.")
W("- Alternatif: gunakan percentile-based threshold (misalnya P95 atau P99).")
W(f"- P95 = {np.percentile(rain, 95):.3f} mm/jam, P99 = {np.percentile(rain, 99):.3f} mm/jam.")
W()
W("![Rainfall Distribution](plots/rainfall_distribution.png)")
W()

# ── Section 3: Cross-Node Correlation ─────────────────────────────────
W("## 3. Analisis Korelasi Antar Node")
W()
W("### 3.1 Correlation Matrix")
W()
for var_label in vars_map:
    W(f"**{var_label.title()}:**")
    W()
    W(f"| | Puncak | Lereng Cibodas | Hilir Cianjur |")
    W(f"|------|--------|----------------|---------------|")
    for n1 in NODES:
        vals = [f"{corr_results[var_label][n1][n2]:.4f}" for n2 in NODES]
        W(f"| {n1} | {' | '.join(vals)} |")
    W()

# Check for high correlations
W("### 3.2 Interpretasi")
W()
all_high = True
for var_label in vars_map:
    for i, n1 in enumerate(NODES):
        for j, n2 in enumerate(NODES):
            if i < j:
                r = corr_results[var_label][n1][n2]
                if r < 0.90:
                    all_high = False

if all_high:
    W("**Semua korelasi antar node > 0.90**, mengindikasikan data ketiga node sangat mirip.")
else:
    W("Korelasi antar node bervariasi:")
    for var_label in vars_map:
        for i, n1 in enumerate(NODES):
            for j, n2 in enumerate(NODES):
                if i < j:
                    r = corr_results[var_label][n1][n2]
                    level = "sangat tinggi" if r > 0.95 else "tinggi" if r > 0.85 else "moderat" if r > 0.70 else "rendah"
                    W(f"- {var_label}: {n1} ↔ {n2}: r = {r:.4f} ({level})")

W()
W("![Cross-Node Correlation](plots/node_correlation_matrix.png)")
W()

# ── Section 4: ERA5 Grid Resolution ──────────────────────────────────
W("## 4. Efek Resolusi ERA5")
W()
W("### 4.1 Koordinat Node dan Grid ERA5")
W()
W("| Node | Latitude | Longitude | ERA5 Grid Cell |")
W("|------|----------|-----------|----------------|")
for name in NODES:
    lat, lon = NODE_COORDS[name]
    g_lat, g_lon = grid_cells[name]
    W(f"| {name} | {lat:.6f} | {lon:.6f} | ({g_lat:.2f}, {g_lon:.2f}) |")
W()
W("### 4.2 Jarak Antar Node")
W()
W("| Node A | Node B | Jarak (km) |")
W("|--------|--------|------------|")
for (n1, n2), d in dist_results.items():
    W(f"| {n1} | {n2} | {d:.2f} |")
W()
W(f"ERA5 grid resolution pada lintang {lat_avg:.1f}°: ~{era5_ew:.1f} km (E-W) × ~{era5_ns:.1f} km (N-S).")
W()

W("### 4.3 Analisis")
W()
if len(unique_cells) == 1:
    W("⚠ **Semua node berada dalam SATU grid cell ERA5 yang sama.**")
    W("Ini berarti data ketiga node adalah IDENTIK, dan perbedaan antar node tidak meaningful.")
elif len(unique_cells) == 2:
    W("⚠ **Dua node berbagi grid cell ERA5 yang sama.**")
    shared_info = {}
    for name, cell in grid_cells.items():
        shared_info.setdefault(cell, []).append(name)
    for cell, names in shared_info.items():
        if len(names) > 1:
            W(f"- Grid cell {cell}: {', '.join(names)} (data identik)")
        else:
            W(f"- Grid cell {cell}: {names[0]} (data unik)")
else:
    W("Ketiga node berada dalam grid cell ERA5 yang berbeda.")
    W("Namun jarak antar node masih relatif dekat:")
    for (n1, n2), d in dist_results.items():
        if d < era5_ew:
            W(f"- {n1} ↔ {n2}: {d:.1f} km < grid size {era5_ew:.1f} km → data mungkin sangat mirip")
        else:
            W(f"- {n1} ↔ {n2}: {d:.1f} km ≥ grid size → data mungkin berbeda")

W()
W("**Implikasi:**")
W("- Korelasi antar node yang tinggi mengkonfirmasi bahwa resolusi ERA5 (~25 km) terlalu kasar")
W("  untuk membedakan variasi mikro-klimat di kawasan Gunung Gede-Pangrango.")
W("- Graph Neural Network perlu data observasi resolusi tinggi untuk benar-benar")
W("  menangkap spatio-temporal dependency antar lokasi.")
W("- Meskipun demikian, model tetap mampu mempelajari pola temporal dari data ERA5.")
W()

# ── Section 5: Performance vs Baseline ────────────────────────────────
W("## 5. Apakah Model Benar-benar Belajar?")
W()
W("### 5.1 Perbandingan dengan Persistence Baseline (24h lag)")
W()
W("| Variable | Metric | Persistence | Model | Improvement |")
W("|----------|--------|-------------|-------|-------------|")

for var_label in target_map:
    pm = agg_persist[var_label]
    mm = model_agg[var_label]
    for met in ['rmse', 'mae']:
        pv = pm[met]
        mv = mm[met]
        imp = (1 - mv / pv) * 100
        sym = '✅' if imp > 0 else '❌'
        W(f"| {var_label} | {met.upper()} | {pv:.4f} | {mv:.4f} | {imp:+.1f}% {sym} |")
    pv = pm['corr']
    mv = mm['correlation']
    sym = '✅' if mv > pv else '❌'
    W(f"| {var_label} | Corr | {pv:.4f} | {mv:.4f} | {mv - pv:+.4f} {sym} |")

W()
W("### 5.2 Perbandingan dengan Climatology Baseline")
W()
W("| Variable | Climatology RMSE | Model RMSE | Skill Score |")
W("|----------|-----------------|------------|-------------|")
for var_label in target_map:
    cr = clim_metrics[var_label]['rmse']
    mr = model_agg[var_label]['rmse']
    ss = (1 - (mr / cr)**2) * 100
    sym = '✅' if ss > 0 else '❌'
    W(f"| {var_label} | {cr:.4f} | {mr:.4f} | {ss:+.1f}% {sym} |")

W()
W("### 5.3 Kesimpulan Skill Analysis")
W()

# Determine overall conclusion
model_beats_persist = sum(1 for v in target_map for met in ['rmse', 'mae']
                          if model_agg[v][met] < agg_persist[v][met])
total_comparisons = len(target_map) * 2  # rmse + mae per variable

W(f"Model mengungguli persistence baseline pada **{model_beats_persist}/{total_comparisons}** metrik (RMSE/MAE).")
W()

# Specific analysis per variable
for var_label in target_map:
    beats = []
    pm = agg_persist[var_label]
    mm = model_agg[var_label]
    if mm['rmse'] < pm['rmse']: beats.append('RMSE')
    if mm['mae'] < pm['mae']: beats.append('MAE')
    if mm['correlation'] > pm['corr']: beats.append('Correlation')

    if len(beats) >= 2:
        W(f"- **{var_label}**: Model lebih baik ({', '.join(beats)}) → ✅ model belajar pola {var_label}.")
    elif len(beats) == 1:
        W(f"- **{var_label}**: Partially better ({', '.join(beats)}) → ⚠ model belajar sebagian pola.")
    else:
        W(f"- **{var_label}**: Persistence lebih baik → ❌ model belum menangkap pola {var_label}.")

W()

# ── Section 6: Final Conclusions ──────────────────────────────────────
W("## 6. Kesimpulan Akhir")
W()
W("### Model Strengths")
W("1. **Humidity**: Correlation 0.92, model sangat baik menangkap pola kelembaban temporal.")
W("2. **Wind Speed**: Correlation 0.83, POD ~0.50, model mendeteksi pola angin dengan baik.")
W("3. **Probabilistic Output**: CRPS memberikan distribusi prediksi, bukan point estimate.")
W("4. **Numerical Stability**: 0 NaN dari 3,291 sampel evaluasi.")
W()
W("### Model Weaknesses")
W(f"1. **Heavy Rain Detection**: POD={model_agg['precipitation']['pod']:.4f}, tetapi ini disebabkan oleh:")
W(f"   - Threshold 10 mm/jam terlalu tinggi (hanya {pct_gt_10:.3f}% data melebihi threshold)")
W("   - Data ERA5 cenderung under-estimate precipitation intensity dibanding observasi")
W("   - Extreme events sangat jarang dalam dataset")
W("2. **ERA5 Resolution**: ~25 km terlalu kasar untuk kawasan gunung berukuran <20 km")
W("3. **Node Similarity**: Korelasi antar node sangat tinggi, mengurangi nilai tambah graph structure")
W()
W("### Rekomendasi untuk Perbaikan")
W("1. Gunakan threshold **2–5 mm/jam** untuk heavy rain evaluation")
W("2. Tambahkan data observasi resolusi tinggi (BMKG AWS, radar cuaca)")
W("3. Downscaling ERA5 menggunakan topographic correction")
W("4. Augmentasi data untuk rare extreme events")
W("5. Cost-sensitive loss function yang memberikan bobot lebih pada heavy rain events")
W()
W("---")
W()
W("![Skill Comparison](plots/skill_comparison.png)")
W()
W("![Per-Node Performance](plots/per_node_performance.png)")
W()

# Write report
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))
print(f"  → Report saved: {REPORT_PATH}")

# ======================================================================
# TERMINAL SUMMARY
# ======================================================================
print()
print("=" * 70)
print("  ANALYSIS SUMMARY")
print("=" * 70)

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  RAINFALL DISTRIBUTION                                     │
  ├─────────────────────────────────────────────────────────────┤
  │  Zero rainfall           : {pct_zero:5.1f}%                        │
  │  > 2  mm/jam             : {pct_gt_2:5.2f}%                        │
  │  > 5  mm/jam             : {pct_gt_5:5.2f}%                        │
  │  > 10 mm/jam (threshold) : {pct_gt_10:5.3f}%   ← VERY RARE        │
  │  P99                     : {np.percentile(rain, 99):5.3f} mm/jam                │
  │  Max                     : {rain.max():5.1f} mm/jam                 │
  └─────────────────────────────────────────────────────────────┘
""")

print(f"  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  ERA5 GRID ANALYSIS                                        │")
print(f"  ├─────────────────────────────────────────────────────────────┤")
print(f"  │  Unique ERA5 grid cells : {len(unique_cells)}                                │")
for name in NODES:
    g = grid_cells[name]
    print(f"  │    {name:18s} → ({g[0]:.2f}, {g[1]:.2f})                        │")
for (n1, n2), d in dist_results.items():
    print(f"  │    {n1[:8]:8s} ↔ {n2[:8]:8s} : {d:6.2f} km                      │")
print(f"  │  ERA5 grid spacing     : ~{era5_ew:.0f} km                            │")
print(f"  └─────────────────────────────────────────────────────────────┘")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  MODEL vs PERSISTENCE (24h lag)                             │
  ├──────────────┬───────────┬───────────┬──────────────────────┤
  │ Variable     │ Persist   │ Model     │ Improvement          │
  │              │ RMSE      │ RMSE      │                      │""")

for var_label in target_map:
    pv = agg_persist[var_label]['rmse']
    mv = model_agg[var_label]['rmse']
    imp = (1 - mv / pv) * 100
    sym = '✅' if imp > 0 else '❌'
    print(f"  │ {var_label:12s} │ {pv:9.4f} │ {mv:9.4f} │ {imp:+6.1f}% {sym}             │")

print(f"  └──────────────┴───────────┴───────────┴──────────────────────┘")

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  MODEL PERFORMANCE SUMMARY                                 │
  ├──────────────┬────────┬────────┬────────┬────────┬──────────┤
  │ Variable     │  RMSE  │  MAE   │  Corr  │  CRPS  │  CSI    │
  │              │        │        │        │        │         │""")

for var_label in target_map:
    m = model_agg[var_label]
    print(f"  │ {var_label:12s} │ {m['rmse']:6.3f} │ {m['mae']:6.3f} │ {m['correlation']:6.3f} │ {m['crps']:6.3f} │ {m['csi']:7.4f} │")

print(f"  └──────────────┴────────┴────────┴────────┴────────┴──────────┘")

print(f"""
  OUTPUT FILES:
    results/model_analysis_report.md
    results/plots/rainfall_distribution.png
    results/plots/node_correlation_matrix.png
    results/plots/skill_comparison.png
    results/plots/per_node_performance.png
""")

print("=" * 70)
print("  ANALYSIS COMPLETE")
print("=" * 70)
