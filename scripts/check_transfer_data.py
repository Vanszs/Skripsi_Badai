import pandas as pd
import numpy as np

# Re-implement Logic so we don't depend on local file imports for this quick check
def label_anomalies(df):
    labels = np.zeros(len(df), dtype=int)
    labels[df['wind_speed'] >= 18.0] = 2
    mask_anomaly = (df['wind_speed'] >= 11.0) & (df['wind_speed'] < 18.0)
    labels[mask_anomaly] = 1
    return labels

def check_file(filepath):
    print(f"\n📂 Checking: {filepath}")
    try:
        df = pd.read_csv(filepath)
        print(f"   ✅ Rows: {len(df)}")
        print(f"   📅 Range: {pd.to_datetime(df['dt'], unit='s').min()} to {pd.to_datetime(df['dt'], unit='s').max()}")
        
        # Apply Labeling Logic
        df['label'] = label_anomalies(df)
        
        counts = df['label'].value_counts().sort_index()
        print(f"   📊 Class Distribution (WMO Saffir-Simpson):")
        print(counts)
        
        if 2 in counts:
            print(f"   🌪️  STORM EVENTS DETECTED! ({counts[2]} hours)")
        else:
            print(f"   ⚠️  No Storms (Class 2) found. Max wind: {df['wind_speed'].max():.2f} m/s")
            
        return df
    except FileNotFoundError:
        print(f"   ❌ File NOT FOUND")

# Check Both
check_file("data/china_history_2y.csv")
check_file("data/tapanuli_test_1y.csv")
