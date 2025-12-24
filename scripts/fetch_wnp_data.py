import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
import os

# Setup Open-Meteo Client
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

URL = "https://archive-api.open-meteo.com/v1/archive"

def fetch_wnp_data():
    print(f"\n🚀 Fetching WESTERN NORTH PACIFIC Data (2024)...")
    print(f"   📍 Location: 15.0N, 150.0E (Pacific Ocean)")
    print(f"   📅 Period: 2024-01-01 to 2024-12-31")
    
    params = {
        "latitude": 15.0,
        "longitude": 150.0,
        "start_date": "2024-01-01",
        "end_date": "2024-12-31",
        "hourly": ["pressure_msl", "windspeed_10m", "windgusts_10m"],
        "timezone": "Asia/Tokyo" # GMT+9 suitable for WNP
    }
    
    try:
        responses = openmeteo.weather_api(URL, params=params)
        response = responses[0]
        
        hourly = response.Hourly()
        data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "pressure": hourly.Variables(0).ValuesAsNumpy(),
            "wind_speed": hourly.Variables(1).ValuesAsNumpy() / 3.6, # km/h to m/s
            "wind_gust": hourly.Variables(2).ValuesAsNumpy() / 3.6
        }
        
        df = pd.DataFrame(data)
        # Fix: Cast to int64 first to avoid pandas conversion error
        df['dt'] = df['date'].astype('int64') // 10**9
        # Keep 'date' column for robustness/readability
        df = df[['dt', 'date', 'pressure', 'wind_speed', 'wind_gust']]
        
        outfile = "data/wnp_1y.csv"
        df.to_csv(outfile, index=False)
        print(f"✅ Saved to {outfile} ({len(df)} rows)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    fetch_wnp_data()
