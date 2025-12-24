import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os
import time

# Setup Open-Meteo Client with Cache and Retry
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_and_save(lat, lon, start_date, end_date, filename, region_name):
    print(f"\n🚀 Fetching {region_name} Data...")
    print(f"   📍 Location: {lat}, {lon}")
    print(f"   📅 Period: {start_date} to {end_date}")
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["pressure_msl", "windspeed_10m", "windgusts_10m"],
        "timezone": "Asia/Bangkok" # UTC+7 (WIB)
    }
    
    try:
        responses = openmeteo.weather_api(URL, params=params)
        response = responses[0]
        
        # Process hourly data
        hourly = response.Hourly()
        hourly_pressure = hourly.Variables(0).ValuesAsNumpy()
        hourly_windspeed = hourly.Variables(1).ValuesAsNumpy()
        hourly_windgust = hourly.Variables(2).ValuesAsNumpy()
        
        # Create DateTime index
        hourly_data = {"date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )}
        
        hourly_data["pressure"] = hourly_pressure
        # Convert km/h to m/s (Open-Meteo default is km/h, we need m/s for Saffir-Simpson)
        hourly_data["wind_speed"] = hourly_windspeed / 3.6 
        hourly_data["wind_gust"] = hourly_windgust / 3.6
        
        df = pd.DataFrame(data=hourly_data)
        
        # Make timestamps timezone-aware/consistent
        # Fix: Cast to int64 first to avoid pandas error
        df['dt'] = df['date'].astype('int64') // 10**9 # Unix timestamp
        # Keep 'date' column for robustness/readability
        df = df[['dt', 'date', 'pressure', 'wind_speed', 'wind_gust']] # Reorder cols
        
        # Save
        save_path = os.path.join("data", filename)
        os.makedirs("data", exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"✅ Saved to {save_path} ({len(df)} rows)")
        
    except Exception as e:
        print(f"❌ Failed to fetch {region_name}: {e}")

if __name__ == "__main__":
    # 1. CHINA SOUTH SEA (Training Data - High Storm Frequency)
    # 15.0N, 112.0E
    # Period: 2 Years (Jan 1 2023 - Dec 24 2024)
    fetch_and_save(
        lat=15.0, 
        lon=112.0, 
        start_date="2023-01-01", 
        end_date="2024-12-24", 
        filename="china_history_2y.csv",
        region_name="South China Sea (TRAIN)"
    )
    
    # 2. TAPANULI (Testing Data - Local Case)
    # 1.55N, 98.83E
    # Period: Recent 1 Year (Jan 1 2024 - Dec 24 2024)
    fetch_and_save(
        lat=1.55, 
        lon=98.83, 
        start_date="2024-01-01", 
        end_date="2024-12-24", 
        filename="tapanuli_test_1y.csv",
        region_name="Tapanuli (TEST)"
    )
