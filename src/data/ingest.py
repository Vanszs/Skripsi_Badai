import openmeteo_requests
import requests_cache
import requests
import pandas as pd
from retry_requests import retry
import os
from datetime import datetime
import numpy as np

from src.config import FINAL_FEATURE_COLS

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

PANGRANGO_COORDS = {
    "lat_min": -6.85,
    "lat_max": -6.70,
    "lon_min": 106.90,
    "lon_max": 107.20
}

# Node definitions for Gunung Gede-Pangrango
# Sesuai judul skripsi: "Nowcasting Probabilistik Hujan Lebat untuk Keselamatan Pendaki"
PANGRANGO_NODES = pd.DataFrame({
    'name': ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur'],
    'lat': [-6.769797, -6.751722, -6.816000],
    'lon': [106.963583, 106.987160, 107.133000]
})



def fetch_elevation(nodes_df):
    """
    Fetch elevation data from Open-Meteo Elevation API.
    Returns dict: {node_name: elevation_meters}
    
    Note: Untuk kawasan Gunung Gede-Pangrango, elevation > 0 menandakan daratan.
    Valid untuk topografi vulkanik pegunungan.
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
    Untuk kawasan Gunung Gede-Pangrango:
      elevation > 0  →  1 (land)
      elevation <= 0 →  0 (sea)
    
    Valid untuk topografi vulkanik pegunungan.
    """
    return 1 if elevation > 0 else 0


def fetch_era5_data(start_year=2005, end_year=2025, interval="hourly"):
    """
    Fetch ERA5 Reanalysis data for kawasan Gunung Gede-Pangrango.
    
    Features added:
    - elevation (static, from Elevation API)
    - land_sea_mask (derived from elevation)
    - dewpoint_2m (humidity proxy)
    - cloud_cover (convective proxy)
    - precipitation_lag1 (autoregressive feature)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    nodes = PANGRANGO_NODES.copy()
    
    # --- STEP 1: Fetch Static Features (Elevation) ---
    elevation_dict = fetch_elevation(nodes)
    
    all_data = []

    params = {
        "latitude": ",".join(str(lat) for lat in nodes['lat']),
        "longitude": ",".join(str(lon) for lon in nodes['lon']),
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "hourly": [
            "precipitation",
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
        ],
        "timezone": "Asia/Singapore"
    }

    print(f"Fetching weather data for {len(nodes)} nodes in one batch request...")
    responses = openmeteo.weather_api(url, params=params)

    for (_, node), response in zip(nodes.iterrows(), responses):
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
        hourly_data["cloud_cover"] = hourly.Variables(7).ValuesAsNumpy()
        
        # Add node identifier
        hourly_data["node"] = node['name']
        
        # --- STEP 3: Add Static Features ---
        elev = elevation_dict.get(node['name'], 0)
        hourly_data["elevation"] = elev
        hourly_data["land_sea_mask"] = derive_land_sea_mask(elev)
        
        df = pd.DataFrame(data=hourly_data)
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # --- STEP 4: Create Lag Features (per node) ---
    print("Creating lag features...")
    combined_df = combined_df.sort_values(['node', 'date']).reset_index(drop=True)
    
    # Lag-1: Previous hour's precipitation
    combined_df['precipitation_lag1'] = combined_df.groupby('node')['precipitation'].shift(1)
    
    # Fill NaN from shift with 0 (first few hours have no history)
    combined_df['precipitation_lag1'] = combined_df['precipitation_lag1'].fillna(0)

    # Enforce the final column contract order for downstream reproducibility.
    ordered_cols = ['date', *FINAL_FEATURE_COLS, 'node', 'land_sea_mask', 'precipitation']
    existing = [col for col in ordered_cols if col in combined_df.columns]
    remaining = [col for col in combined_df.columns if col not in existing]
    combined_df = combined_df[existing + remaining]
    
    # --- STEP 5: Save ---
    os.makedirs('data/raw', exist_ok=True)
    output_path = f'data/raw/pangrango_era5_{start_year}_{end_year}.parquet'
    combined_df.to_parquet(output_path)
    
    print(f"\nData saved to {output_path}")
    print(f"   Shape: {combined_df.shape}")
    print(f"   Columns: {list(combined_df.columns)}")
    print(f"   Date range: {combined_df['date'].min()} to {combined_df['date'].max()}")
    
    return combined_df


def get_node_metadata():
    """Return node metadata including static features for graph construction."""
    elevation_dict = fetch_elevation(PANGRANGO_NODES)
    
    nodes = PANGRANGO_NODES.copy()
    nodes['elevation'] = nodes['name'].map(elevation_dict)
    nodes['land_sea_mask'] = nodes['elevation'].apply(derive_land_sea_mask)
    
    return nodes


if __name__ == "__main__":
    fetch_era5_data(start_year=2005, end_year=2025)
