import pandas as pd
import os

HISTORY_FILE = 'data/tapanuli_history_30days.csv'
CURRENT_FILE = 'data/tapanuli_weather.csv'
MERGED_FILE = 'data/tapanuli_weather.csv' # Overwrite target

def merge_datasets():
    dfs = []
    
    if os.path.exists(HISTORY_FILE):
        print(f"Loading History: {HISTORY_FILE}")
        dfs.append(pd.read_csv(HISTORY_FILE))
        
    if os.path.exists(CURRENT_FILE):
        print(f"Loading Current: {CURRENT_FILE}")
        dfs.append(pd.read_csv(CURRENT_FILE))
        
    if dfs:
        # Concatenate
        full_df = pd.concat(dfs)
        
        # Deduplicate by 'dt'
        full_df.drop_duplicates(subset=['dt'], keep='last', inplace=True)
        
        # Sort
        full_df['dt'] = pd.to_datetime(full_df['dt'])
        full_df.sort_values('dt', inplace=True)
        
        # Save
        full_df.to_csv(MERGED_FILE, index=False)
        print(f"✅ Merged Dataset Saved: {MERGED_FILE}")
        print(f"   Total Rows: {len(full_df)}")
        print(f"   Range: {full_df['dt'].min()} to {full_df['dt'].max()}")
    else:
        print("❌ No data files found to merge.")

if __name__ == "__main__":
    merge_datasets()
