import openmeteo_requests
import requests_cache
import requests
import pandas as pd
from retry_requests import retry
import os
from datetime import datetime
import numpy as np

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

SITARO_COORDS = {
    "lat_min": 2.0,
    "lat_max": 3.0,
    "lon_min": 125.0,
    "lon_max": 126.0
}

# Node definitions for Sitaro islands
SITARO_NODES = pd.DataFrame({
    'name': ['Siau', 'Tagulandang', 'Biaro'],
    'lat': [2.75, 2.33, 2.10],
    'lon': [125.40, 125.42, 125.37]
})


def fetch_elevation(nodes_df):
    """
    Fetch elevation data from Open-Meteo Elevation API.
    Returns dict: {node_name: elevation_meters}
    
    Note: For volcanic Sitaro islands, elevation > 0 indicates land.
    This is scientifically valid for steep volcanic terrain without deltas.
    """
    lats = ",".join([str(lat) for lat in nodes_df['lat']])
    lons = ",".join([str(lon) for lon in nodes_df['lon']])
    
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    
    print("Fetching elevation data...")
    response = requests.get(url)
    data = response.json()
    
    elevations = data.get('elevation', [0] * len(nodes_df))
    
    elevation_dict = {}
    for i, name in enumerate(nodes_df['name']):
        elev = elevations[i] if i < len(elevations) else 0
        elevation_dict[name] = elev
        print(f"  {name}: {elev}m")
    
    return elevation_dict


def derive_land_sea_mask(elevation):
    """
    Derive land-sea mask from elevation.
    For volcanic Sitaro islands with steep coastlines:
      elevation > 0  →  1 (land)
      elevation <= 0 →  0 (sea)
    
    This is valid for volcanic island topography without deltas or estuaries.
    """
    return 1 if elevation > 0 else 0


def fetch_era5_data(start_year=2005, end_year=2025, interval="hourly"):
    """
    Fetch ERA5 Reanalysis data for Sitaro Region.
    
    Features added:
    - elevation (static, from Elevation API)
    - land_sea_mask (derived from elevation)
    - dewpoint_2m (humidity proxy)
    - cloudcover (convective proxy)
    - precipitation_lag1 (autoregressive feature)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    nodes = SITARO_NODES.copy()
    
    # --- STEP 1: Fetch Static Features (Elevation) ---
    elevation_dict = fetch_elevation(nodes)
    
    all_data = []

    for _, node in nodes.iterrows():
        # --- STEP 2: Fetch Dynamic Features ---
        params = {
            "latitude": node['lat'],
            "longitude": node['lon'],
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "hourly": [
                "precipitation",           # Target
                "temperature_2m",           # Dynamic feature
                "relative_humidity_2m",     # Dynamic feature
                "dewpoint_2m",              # NEW: Humidity proxy
                "surface_pressure",         # Dynamic feature
                "wind_speed_10m",           # Dynamic feature
                "wind_direction_10m",       # Dynamic feature
                "cloudcover",               # NEW: Convective proxy
            ],
            "timezone": "Asia/Singapore"  # WITA
        }
        
        print(f"Fetching weather data for {node['name']}...")
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        
        # Process hourly data
        hourly = response.Hourly()
        
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )
        }
        
        # Extract all variables in order
        hourly_data["precipitation"] = hourly.Variables(0).ValuesAsNumpy()
        hourly_data["temperature_2m"] = hourly.Variables(1).ValuesAsNumpy()
        hourly_data["relative_humidity_2m"] = hourly.Variables(2).ValuesAsNumpy()
        hourly_data["dewpoint_2m"] = hourly.Variables(3).ValuesAsNumpy()
        hourly_data["surface_pressure"] = hourly.Variables(4).ValuesAsNumpy()
        hourly_data["wind_speed_10m"] = hourly.Variables(5).ValuesAsNumpy()
        hourly_data["wind_direction_10m"] = hourly.Variables(6).ValuesAsNumpy()
        hourly_data["cloudcover"] = hourly.Variables(7).ValuesAsNumpy()
        
        # Add node identifier
        hourly_data["node_id"] = node['name']
        
        # --- STEP 3: Add Static Features ---
        elev = elevation_dict.get(node['name'], 0)
        hourly_data["elevation"] = elev
        hourly_data["land_sea_mask"] = derive_land_sea_mask(elev)
        
        df = pd.DataFrame(data=hourly_data)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # --- STEP 4: Create Lag Features (per node) ---
    print("Creating lag features...")
    combined_df = combined_df.sort_values(['node_id', 'date']).reset_index(drop=True)
    
    # Lag-1: Previous hour's precipitation
    combined_df['precipitation_lag1'] = combined_df.groupby('node_id')['precipitation'].shift(1)
    
    # Lag-3: 3 hours ago (for short-term pattern)
    combined_df['precipitation_lag3'] = combined_df.groupby('node_id')['precipitation'].shift(3)
    
    # Fill NaN from shift with 0 (first few hours have no history)
    combined_df['precipitation_lag1'] = combined_df['precipitation_lag1'].fillna(0)
    combined_df['precipitation_lag3'] = combined_df['precipitation_lag3'].fillna(0)
    
    # --- STEP 5: Save ---
    os.makedirs('data/raw', exist_ok=True)
    output_path = f'data/raw/sitaro_era5_{start_year}_{end_year}.parquet'
    combined_df.to_parquet(output_path)
    
    print(f"\n✅ Data saved to {output_path}")
    print(f"   Shape: {combined_df.shape}")
    print(f"   Columns: {list(combined_df.columns)}")
    print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


def get_node_metadata():
    """Return node metadata including static features for graph construction."""
    elevation_dict = fetch_elevation(SITARO_NODES)
    
    nodes = SITARO_NODES.copy()
    nodes['elevation'] = nodes['name'].map(elevation_dict)
    nodes['land_sea_mask'] = nodes['elevation'].apply(derive_land_sea_mask)
    
    return nodes


if __name__ == "__main__":
    fetch_era5_data(start_year=2005, end_year=2025)
