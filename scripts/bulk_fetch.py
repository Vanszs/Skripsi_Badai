import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import sys

# Ensure root is in path
sys.path.append(os.getcwd())

from data.fetch_weather import WeatherFetcher

# Config
DAYS_TO_FETCH = 30  # Max safe limit (720 calls)
LAT = 1.75
LON = 98.75
OUTPUT_FILE = 'data/tapanuli_history_30days.csv'

load_dotenv()
API_KEY = os.getenv('OWM_API_KEY')

def fetch_single_hour(fetcher, target_dt):
    """
    Helper to fetch one hour. Returns dict or None.
    """
    timestamp = int(target_dt.timestamp())
    resp = fetcher.fetch_timestamp(timestamp)
    if resp and 'data' in resp and len(resp['data']) > 0:
        data = resp['data'][0]
        return {
            'dt': target_dt,
            'pressure': data.get('pressure'),
            'wind_speed': data.get('wind_speed'),
            'wind_gust': data.get('wind_gust', data.get('wind_speed')) # Fallback
        }
    return None

def main():
    if not API_KEY:
        print("❌ No API Key found.")
        return

    fetcher = WeatherFetcher(API_KEY, LAT, LON)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_TO_FETCH)
    
    # Generate list of hourly timestamps
    timestamps = []
    curr = start_date
    while curr < end_date:
        timestamps.append(curr)
        curr += timedelta(hours=1)
        
    print(f"🚀 Starting Bulk Fetch: {len(timestamps)} hours ({DAYS_TO_FETCH} days)")
    print(f"   Quota Cost: {len(timestamps)} calls")
    print("   Note: 2 Months (1440 calls) would exceed daily limit (1000).")
    print("   Fetching 30 days (720 calls) instead to be safe.")
    
    results = []
    
    # Parallel Fetching (Speed up)
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_dt = {executor.submit(fetch_single_hour, fetcher, dt): dt for dt in timestamps}
        
        completed = 0
        total = len(timestamps)
        
        for future in as_completed(future_to_dt):
            completed += 1
            if completed % 50 == 0:
                print(f"   Progress: {completed}/{total} ({(completed/total)*100:.1f}%)")
            
            res = future.result()
            if res:
                results.append(res)
                
    # Save
    if results:
        df = pd.DataFrame(results)
        df.sort_values('dt', inplace=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ DONE! Saved {len(df)} rows to {OUTPUT_FILE}")
        print(df.head())
        print(df.tail())
    else:
        print("❌ No data fetched.")

if __name__ == "__main__":
    main()
