import pandas as pd
import sys
import os
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Ensure root is in path
sys.path.append(os.getcwd())
from data.fetch_weather import WeatherFetcher

# Config - Cyclone Senyar (Selat Malaka)
LAT = 5.0
LON = 98.0
START_DATE = datetime(2025, 11, 21) # Bibit Detected
END_DATE = datetime(2025, 11, 28)   # +7 Days (Covering Nov 26 peak)
OUTPUT_FILE = 'data/senyar_cyclone.csv'

load_dotenv()
API_KEY = os.getenv('OWM_API_KEY')

def fetch_hour(fetcher, dt):
    ts = int(dt.timestamp())
    resp = fetcher.fetch_timestamp(ts)
    if resp and 'data' in resp and len(resp['data']) > 0:
        d = resp['data'][0]
        return {
            'dt': dt,
            'pressure': d.get('pressure'),
            'wind_speed': d.get('wind_speed'),
            'wind_gust': d.get('wind_gust', d.get('wind_speed')),
            'temp': d.get('temp'),
            'humidity': d.get('humidity') # Extra details for analysis
        }
    return None

def main():
    if not API_KEY:
        print("❌ Missing API Key")
        return

    fetcher = WeatherFetcher(API_KEY, LAT, LON)
    
    # Generate Timestamps
    timestamps = []
    curr = START_DATE
    while curr < END_DATE:
        timestamps.append(curr)
        curr += timedelta(hours=1)
        
    print(f"🌪️ Fetching Cyclone SENYAR Data: {len(timestamps)} hours")
    print(f"   Location: {LAT}, {LON}")
    print(f"   Period: {START_DATE.date()} to {END_DATE.date()}")
    
    results = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(fetch_hour, fetcher, t): t for t in timestamps}
        
        done = 0
        for f in as_completed(futures):
            res = f.result()
            if res: results.append(res)
            done += 1
            if done % 24 == 0: print(f"   Progress: {done}/{len(timestamps)}")
            
    if results:
        df = pd.DataFrame(results).sort_values('dt')
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Success! Saved to {OUTPUT_FILE}")
        print(df.head())
    else:
        print("❌ No data fetched.")

if __name__ == "__main__":
    main()
