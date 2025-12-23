import pandas as pd
from dotenv import load_dotenv
import sys
import os

# Ensure root is in path
sys.path.append(os.getcwd())

from data.fetch_weather import WeatherFetcher

# Load API Key
load_dotenv()
API_KEY = os.getenv('OWM_API_KEY')
LAT = 1.75   # Tapanuli Tengah
LON = 98.75 

def create_static_dataset():
    if not API_KEY:
        print("❌ Error: API Key not found in .env")
        return

    print(f"📡 Connecting to OpenWeatherMap (Lat: {LAT}, Lon: {LON})...")
    fetcher = WeatherFetcher(api_key=API_KEY, lat=LAT, lon=LON)
    
    try:
        # Fetch Real Data (Full One Call Response)
        raw_data = fetcher.fetch_all_data()
        
        if raw_data:
            import json
            
            # 1. Save Raw JSON (Preserve Alerts, Daily, Current)
            json_path = 'data/tapanuli_full_raw.json'
            with open(json_path, 'w') as f:
                json.dump(raw_data, f, indent=2)
            print(f"📦 Full Data (JSON) saved to: {json_path}")
            
            # 2. Process 'Hourly' for Training (CSV)
            if 'hourly' in raw_data:
                hourly = raw_data['hourly']
                df = pd.DataFrame(hourly)
                
                # Convert timestamp
                df['dt'] = pd.to_datetime(df['dt'], unit='s')
                
                # Ensure columns exist (Key Data Fields)
                cols_to_keep = ['dt', 'pressure', 'wind_speed']
                if 'wind_gust' in df.columns:
                    cols_to_keep.append('wind_gust')
                else:
                    df['wind_gust'] = df['wind_speed'] # Fallback
                    cols_to_keep.append('wind_gust')
                    
                # Select features
                df = df[cols_to_keep]
                
                # Save to CSV
                csv_path = 'data/tapanuli_weather.csv'
                df.to_csv(csv_path, index=False)
                print(f"📊 Hourly Processed Data (CSV) saved to: {csv_path}")
                print(f"   Rows: {len(df)}")
                print(df.head(3))
            
            # 3. Log Key Information
            print("\n🔍 --- Data Snapshot ---")
            if 'current' in raw_data:
                cur = raw_data['current']
                print(f"   • Current Pressure: {cur.get('pressure')} hPa")
                print(f"   • Current Wind: {cur.get('wind_speed')} m/s")
            
            if 'alerts' in raw_data:
                print(f"   • 🚨 Alerts Active: {len(raw_data['alerts'])}")
                for a in raw_data['alerts']:
                    print(f"     - {a.get('event')} ({a.get('sender_name')})")
            else:
                print("   • Alerts: None")
                
            if 'daily' in raw_data:
                print(f"   • Daily Forecasts: {len(raw_data['daily'])} days")
            
        else:
            print("❌ Failed: API returned None.")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_static_dataset()
