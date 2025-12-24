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

def fetch_senyar_1y():
    print(f"\n🚀 Fetching SENYAR 1 YEAR Data (2025)...")
    print(f"   📍 Location: 5.0N, 98.0E (Selat Malaka)")
    print(f"   📅 Period: 2025-01-01 to 2025-12-24")
    
    # 5.0N, 98.0E
    params = {
        "latitude": 5.0,
        "longitude": 98.0,
        "start_date": "2025-01-01",
        "end_date": "2025-12-24",
        "hourly": ["pressure_msl", "windspeed_10m", "windgusts_10m"],
        "timezone": "Asia/Bangkok"
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
        df['dt'] = df['date'].astype('int64') // 10**9
        df = df[['dt', 'pressure', 'wind_speed', 'wind_gust']]
        
        outfile = "data/senyar_1y.csv"
        df.to_csv(outfile, index=False)
        print(f"✅ Saved to {outfile} ({len(df)} rows)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    fetch_senyar_1y()
