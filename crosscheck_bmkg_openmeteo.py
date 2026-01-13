"""
Cross-check BMKG Station Data vs Open-Meteo ERA5

Generic script that handles both station files:
- Stasiun Meteorologi Naha (Sangihe)
- Stasiun Meteorologi Maritim Bitung

Handles special values:
- 8888: Data tidak terukur
- 9999: Tidak ada data
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import requests
from scipy import stats
import sys

# ============================================================
# CONFIGURATION
# ============================================================
# File to analyze - change this path as needed
FILE_PATH = r'd:\SKRIPSI\Skripsi_Bevan\laporan_iklim_harian-260111233348.xlsx'

# ============================================================
# 1. LOAD BMKG DATA
# ============================================================
print("=" * 60)
print("CROSS-CHECK: BMKG STATION vs OPEN-METEO (ERA5)")
print("=" * 60)

df_raw = pd.read_excel(FILE_PATH, header=None)

# Extract metadata
wmo_id = str(df_raw.iloc[0, 2]).replace(': ', '').strip()
station_name = str(df_raw.iloc[1, 2]).replace(': ', '').strip()
lat = float(str(df_raw.iloc[2, 2]).replace(': ', '').strip())
lon = float(str(df_raw.iloc[3, 2]).replace(': ', '').strip())
elev = str(df_raw.iloc[4, 2]).replace(': ', '').strip()

print(f"\nStation: {station_name}")
print(f"WMO ID: {wmo_id}")
print(f"Latitude: {lat}°N")
print(f"Longitude: {lon}°E")
print(f"Elevation: {elev}")

# Find data rows (after TANGGAL header)
header_row = df_raw[df_raw.iloc[:, 0] == 'TANGGAL'].index[0]
data_rows = df_raw.iloc[header_row+1:].copy()
data_rows = data_rows[data_rows.iloc[:, 0].notna()]  # Remove NaN rows

df_bmkg = data_rows.iloc[:, [0, 1]].copy()
df_bmkg.columns = ['date', 'rr_bmkg']

# Parse dates
df_bmkg['date'] = pd.to_datetime(df_bmkg['date'], format='%d-%m-%Y', errors='coerce')

# Handle special values
df_bmkg['rr_bmkg'] = pd.to_numeric(df_bmkg['rr_bmkg'], errors='coerce')
special_mask = df_bmkg['rr_bmkg'].isin([8888, 9999])
print(f"\nSpecial values (8888/9999) found: {special_mask.sum()}")
df_bmkg.loc[special_mask, 'rr_bmkg'] = np.nan

# Drop invalid rows
df_bmkg = df_bmkg.dropna().reset_index(drop=True)

print(f"Data Period: {df_bmkg['date'].min().date()} to {df_bmkg['date'].max().date()}")
print(f"Valid Days: {len(df_bmkg)}")

# ============================================================
# 2. FETCH OPEN-METEO DATA
# ============================================================
print("\nFetching Open-Meteo data...")

start_date = df_bmkg['date'].min().strftime('%Y-%m-%d')
end_date = df_bmkg['date'].max().strftime('%Y-%m-%d')

response = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": lat, "longitude": lon,
    "start_date": start_date, "end_date": end_date,
    "daily": "precipitation_sum", "timezone": "Asia/Makassar"
})
data = response.json()

if 'daily' not in data:
    print("Error fetching Open-Meteo data!")
    print(data)
    sys.exit(1)

df_meteo = pd.DataFrame({
    'date': pd.to_datetime(data['daily']['time']),
    'rr_meteo': data['daily']['precipitation_sum']
})

print(f"Open-Meteo data fetched: {len(df_meteo)} days")

# ============================================================
# 3. MERGE AND COMPARE
# ============================================================
df_compare = pd.merge(df_bmkg, df_meteo, on='date', how='inner')
df_compare['diff'] = df_compare['rr_meteo'] - df_compare['rr_bmkg']

print(f"Matched days: {len(df_compare)}")

# Calculate metrics
corr = df_compare['rr_bmkg'].corr(df_compare['rr_meteo'])
mae = np.abs(df_compare['diff']).mean()
rmse = np.sqrt((df_compare['diff']**2).mean())
bias = df_compare['diff'].mean()

# Handle edge cases for regression
if len(df_compare) > 2 and df_compare['rr_bmkg'].std() > 0:
    slope, intercept, r_value, p_value, _ = stats.linregress(df_compare['rr_bmkg'], df_compare['rr_meteo'])
else:
    slope, intercept, r_value, p_value = 0, 0, 0, 1

# ============================================================
# 4. PRINT RESULTS
# ============================================================
print("\n" + "=" * 60)
print("METRICS")
print("=" * 60)
print(f"Pearson Correlation: {corr:.4f}")
print(f"R-Squared: {r_value**2:.4f}")
print(f"MAE: {mae:.2f} mm/day")
print(f"RMSE: {rmse:.2f} mm/day")
print(f"Mean Bias: {bias:+.2f} mm/day")

print("\n" + "=" * 60)
print("DETAILED COMPARISON")
print("=" * 60)
print(df_compare.to_string(index=False))

# ============================================================
# 5. VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Time Series
ax1 = axes[0, 0]
ax1.plot(df_compare['date'], df_compare['rr_bmkg'], 'o-', label=f'BMKG ({station_name})', 
         color='blue', linewidth=2, markersize=6)
ax1.plot(df_compare['date'], df_compare['rr_meteo'], 's--', label='Open-Meteo (ERA5)', 
         color='red', linewidth=2, markersize=5, alpha=0.8)
ax1.set_xlabel('Date')
ax1.set_ylabel('Precipitation (mm/day)')
ax1.set_title(f'Time Series: {station_name}', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Scatter
ax2 = axes[0, 1]
ax2.scatter(df_compare['rr_bmkg'], df_compare['rr_meteo'], s=80, alpha=0.7, 
            c='steelblue', edgecolors='black')
max_val = max(df_compare['rr_bmkg'].max(), df_compare['rr_meteo'].max(), 1)
ax2.plot([0, max_val], [0, max_val], 'k--', label='1:1 Line', linewidth=2)
if slope != 0:
    x_line = np.linspace(0, max_val, 100)
    ax2.plot(x_line, slope * x_line + intercept, 'r-', 
             label=f'Fit (R²={r_value**2:.3f})', linewidth=2)
ax2.set_xlabel('BMKG (mm/day)')
ax2.set_ylabel('Open-Meteo (mm/day)')
ax2.set_title(f'Scatter: Corr = {corr:.3f}', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Error Distribution
ax3 = axes[1, 0]
ax3.hist(df_compare['diff'], bins=15, color='forestgreen', edgecolor='white', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', linewidth=2)
ax3.axvline(bias, color='orange', linestyle='-', linewidth=2, label=f'Bias: {bias:.2f}mm')
ax3.set_xlabel('Error (Open-Meteo - BMKG) [mm]')
ax3.set_ylabel('Frequency')
ax3.set_title('Error Distribution', fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Bar comparison
ax4 = axes[1, 1]
x = np.arange(len(df_compare))
width = 0.35
ax4.bar(x - width/2, df_compare['rr_bmkg'], width, label='BMKG', color='blue', alpha=0.7)
ax4.bar(x + width/2, df_compare['rr_meteo'], width, label='Open-Meteo', color='red', alpha=0.7)
ax4.set_xlabel('Day Index')
ax4.set_ylabel('Precipitation (mm)')
ax4.set_title('Day-by-Day Comparison', fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.suptitle(f'Cross-Check: {station_name} ({lat}°N, {lon}°E)', fontsize=14, fontweight='bold')
plt.tight_layout()

# Save with station name
safe_name = station_name.replace(' ', '_').replace('/', '_')[:30]
output_file = f'crosscheck_{safe_name}.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")

# ============================================================
# 6. SAVE RESULTS
# ============================================================
results_file = f'crosscheck_{safe_name}_results.txt'
with open(results_file, 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("CROSS-CHECK: BMKG STATION vs OPEN-METEO (ERA5)\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Stasiun: {station_name}\n")
    f.write(f"WMO ID: {wmo_id}\n")
    f.write(f"Koordinat: {lat}°N, {lon}°E\n")
    f.write(f"Elevasi: {elev}\n")
    f.write(f"Periode: {start_date} to {end_date}\n")
    f.write(f"Valid Days: {len(df_compare)}\n\n")
    f.write("METRICS:\n")
    f.write(f"  Pearson Correlation: {corr:.4f}\n")
    f.write(f"  R-Squared: {r_value**2:.4f}\n")
    f.write(f"  MAE: {mae:.2f} mm/day\n")
    f.write(f"  RMSE: {rmse:.2f} mm/day\n")
    f.write(f"  Bias: {bias:+.2f} mm/day\n\n")
    f.write("DATA:\n")
    f.write(df_compare.to_string(index=False))

print(f"✓ Saved: {results_file}")

# Summary
interp = "GOOD" if corr >= 0.7 else ("MODERATE" if corr >= 0.4 else "LOW")
print(f"\n{'='*60}")
print(f"SUMMARY: Correlation = {corr:.3f} ({interp}), Bias = {bias:+.2f} mm")
print(f"{'='*60}")
