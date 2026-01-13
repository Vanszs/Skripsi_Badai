import pandas as pd

df = pd.read_parquet('data/raw/sitaro_era5_2005_2025.parquet')

print("="*60)
print("STRUKTUR DATA sitaro_era5_2005_2025.parquet")
print("="*60)

print(f"\n1. DIMENSI: {df.shape[0]:,} baris × {df.shape[1]} kolom")

print(f"\n2. KOLOM:")
for col in df.columns:
    print(f"   - {col}")

print(f"\n3. DISTRIBUSI PER PULAU:")
print(df['node_id'].value_counts())

print(f"\n4. RENTANG WAKTU:")
print(f"   Dari: {df['date'].min()}")
print(f"   Sampai: {df['date'].max()}")

print(f"\n5. SAMPLE DATA (3 baris pertama per pulau):")
for node in df['node_id'].unique():
    print(f"\n   --- {node} ---")
    sample = df[df['node_id'] == node].head(3)[['date', 'precipitation', 'temperature_2m', 'node_id']]
    print(sample.to_string(index=False))
