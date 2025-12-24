import pandas as pd
import os

files = {
    "WNP (Test)": "data/wnp_1y.csv",
    "CHINA (Train)": "data/china_history_2y.csv"
}

for name, fpath in files.items():
    print(f"\n--- {name} ---")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        stats = df[['wind_speed', 'wind_gust']].describe(percentiles=[0.9, 0.95, 0.99, 0.999])
        print(stats)
        
        # Check specific count > 18.0
        n_storm = (df['wind_speed'] >= 18.0).sum()
        n_gust_storm = (df['wind_gust'] >= 18.0).sum()
        print(f"Rows with wind_speed >= 18.0: {n_storm}")
        print(f"Rows with wind_gust >= 18.0: {n_gust_storm}")
    else:
        print("File not found.")
