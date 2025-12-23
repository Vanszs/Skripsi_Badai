import pandas as pd
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())
from data.preprocessing import label_anomalies

# Load Senyar
df = pd.read_csv('data/senyar_cyclone.csv')
print(f"Total Rows: {len(df)}")
print(f"Pressure Range: {df['pressure'].min()} - {df['pressure'].max()} hPa")

# Test Labeling with 0.6 threshold
labels = label_anomalies(df, threshold_drop=0.6, window_hours=3)
n_anomalies = labels.sum()
print(f"Senyar (0.6 hPa): {int(n_anomalies)} anomalies ({(n_anomalies/len(df))*100:.2f}%)")

# Test Labeling with Default 3.0 threshold (Should trigger for real cyclone)
labels_real = label_anomalies(df, threshold_drop=3.0, window_hours=3)
n_anomalies_real = labels_real.sum()
print(f"Senyar (3.0 hPa - Real Storm): {int(n_anomalies_real)} anomalies ({(n_anomalies_real/len(df))*100:.2f}%)")
