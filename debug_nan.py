"""Quick debug script to find NaN source in evaluation."""
import torch, numpy as np, pandas as pd
from src.inference import load_model_and_stats, run_inference_hybrid

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, stats, retrieval_db = load_model_and_stats('models/diffusion_chkpt.pth')
model.to(device); model.eval()

df = pd.read_parquet('data/raw/pangrango_era5_2005_2025.parquet')
df['date'] = pd.to_datetime(df['date'])
test_df = df[df['date'] >= '2022-01-01']
node_df = test_df[test_df['node'] == 'Puncak'].sort_values('date').reset_index(drop=True)

feature_cols = model.config.get('feature_cols', [])
available_cols = [c for c in feature_cols if c in node_df.columns]
c_mean = stats['c_mean'].numpy()[:len(available_cols)]
c_std = stats['c_std'].numpy()[:len(available_cols)]
seq_len = model.config['seq_len']

print(f"Testing {len(range(0, 1000, 24))} samples...")

nan_indices = []
for idx in range(0, 1000, 24):
    seq_end = idx + seq_len
    if seq_end >= len(node_df): break
    seq_df = node_df.iloc[idx:seq_end]
    features = seq_df[available_cols].values
    features_norm = (features - c_mean) / (c_std + 1e-5)
    
    prev_row = node_df.iloc[seq_end - 1]
    lag_values = {
        'precipitation': prev_row['precipitation'],
        'wind_speed': prev_row['wind_speed_10m'],
        'humidity': prev_row['relative_humidity_2m']
    }
    
    result = run_inference_hybrid(features_norm, model, stats, retrieval_db,
        lag_values=lag_values, num_samples=5, device=device)
    
    hp = np.median(result['hybrid_precipitation'])
    hw = np.median(result['hybrid_wind_speed'])
    hh = np.median(result['hybrid_humidity'])
    
    if np.isnan(hp) or np.isnan(hw) or np.isnan(hh):
        nan_indices.append(idx)
        key = 'hybrid_precipitation'
        print(f"NaN at idx={idx}! hp={hp:.4f}, hw={hw:.4f}, hh={hh:.4f}")
        print(f"  raw model precip: {result['precipitation'][:3]}")
        print(f"  hybrid precip: {result[key][:3]}")
        print(f"  lag_values: {lag_values}")
        if len(nan_indices) >= 5: break
    
    if np.isinf(hp) or np.isinf(hw) or np.isinf(hh):
        print(f"INF at idx={idx}! hp={hp}, hw={hw}, hh={hh}")
        print(f"  raw precip: {result['precipitation'][:3]}")

if not nan_indices:
    print("No NaN found in first 1000 samples!")
else:
    print(f"\nFound NaN at {len(nan_indices)} positions: {nan_indices}")
