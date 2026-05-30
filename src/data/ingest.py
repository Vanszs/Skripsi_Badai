import json
import os
from typing import Dict, List

import openmeteo_requests
import pandas as pd
import requests
import requests_cache
from retry_requests import retry

from src.config import (
    FINAL_FEATURE_COLS,
    MAIN_NODE_NAME,
    NODE_COORDINATES,
    NODE_DEFINITIONS,
    NODE_NAMES,
    NODE_ROLES,
    OPEN_METEO_MODEL,
)

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

CANONICAL_DATA_PATH = "data/raw/pangrango_era5_5node_2005_2025.parquet"
GRID_VALIDATION_PATH = "data/raw/pangrango_era5_5node_grid_validation.json"


def get_node_definitions_df() -> pd.DataFrame:
    """
    Return canonical node table in mandatory [MAIN,UP,DOWN,LEFT,RIGHT] order.
    """
    return pd.DataFrame(NODE_DEFINITIONS).copy()


def fetch_elevation(nodes_df: pd.DataFrame) -> Dict[str, float]:
    """
    Fetch elevation data from Open-Meteo Elevation API.
    Returns dict: {node_name: elevation_meters}.
    """
    lats = ",".join([str(lat) for lat in nodes_df["lat"]])
    lons = ",".join([str(lon) for lon in nodes_df["lon"]])
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lats}&longitude={lons}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    data = response.json()
    elevations = data.get("elevation", [0] * len(nodes_df))
    return {name: float(elevations[i]) for i, name in enumerate(nodes_df["name"])}


def derive_land_sea_mask(elevation: float) -> int:
    return 1 if float(elevation) > 0 else 0


def _extract_hourly_response(response):
    """
    Convert one Open-Meteo response object to dict of arrays.
    """
    hourly = response.Hourly()
    return {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        ),
        "precipitation": hourly.Variables(0).ValuesAsNumpy(),
        "temperature_2m": hourly.Variables(1).ValuesAsNumpy(),
        "relative_humidity_2m": hourly.Variables(2).ValuesAsNumpy(),
        "dewpoint_2m": hourly.Variables(3).ValuesAsNumpy(),
        "surface_pressure": hourly.Variables(4).ValuesAsNumpy(),
        "wind_speed_10m": hourly.Variables(5).ValuesAsNumpy(),
        "wind_direction_10m": hourly.Variables(6).ValuesAsNumpy(),
        "cloud_cover": hourly.Variables(7).ValuesAsNumpy(),
    }


def _validate_grid_identity(nodes_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Validate canonical node -> grid-center mapping with models=era5.
    Fail-fast if any collision happens in active 5-node set.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    grid_meta: Dict[str, Dict[str, float]] = {}
    collisions: Dict[str, List[str]] = {}

    for _, row in nodes_df.iterrows():
        params = {
            "latitude": float(row["lat"]),
            "longitude": float(row["lon"]),
            "start_date": "2025-01-01",
            "end_date": "2025-01-01",
            "hourly": "temperature_2m",
            "timezone": "GMT",
            "models": OPEN_METEO_MODEL,
        }
        resp = requests.get(url, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        grid_lat = float(payload["latitude"])
        grid_lon = float(payload["longitude"])
        key = f"{grid_lat:.6f},{grid_lon:.6f}"

        grid_meta[row["name"]] = {
            "requested_lat": float(row["lat"]),
            "requested_lon": float(row["lon"]),
            "grid_center_lat": grid_lat,
            "grid_center_lon": grid_lon,
            "elevation": float(payload.get("elevation", 0.0)),
            "model_mode": OPEN_METEO_MODEL,
        }
        collisions.setdefault(key, []).append(row["name"])

    collided = {k: v for k, v in collisions.items() if len(v) > 1}
    if collided:
        raise ValueError(f"Grid collision detected for canonical nodes: {collided}")

    main = grid_meta[MAIN_NODE_NAME]
    main_center = (main["grid_center_lat"], main["grid_center_lon"])
    expected_main = NODE_COORDINATES[MAIN_NODE_NAME]
    if main_center != expected_main:
        raise ValueError(
            f"Main node grid mismatch. Expected center {expected_main}, got {main_center}."
        )

    return grid_meta


def _write_grid_validation_report(grid_meta: Dict[str, Dict[str, float]]) -> None:
    os.makedirs("data/raw", exist_ok=True)
    report = {
        "canonical_nodes": NODE_NAMES,
        "node_roles": NODE_ROLES,
        "main_node": MAIN_NODE_NAME,
        "main_node_identifier": NODE_COORDINATES[MAIN_NODE_NAME],
        "model_mode": OPEN_METEO_MODEL,
        "grid_policy": "grid_center_strict",
        "nodes": grid_meta,
        "notes": {
            "non_independence_policy": "fail_if_collision_in_active_nodes",
            "down_node_context": "accepted_maritime_context_if_elevation<=0",
        },
    }
    with open(GRID_VALIDATION_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def fetch_era5_data(start_year=2005, end_year=2025, interval="hourly") -> pd.DataFrame:
    """
    Fetch canonical 5-node ERA5 data from Open-Meteo archive API.
    """
    if interval != "hourly":
        raise ValueError("Only hourly interval is supported in this pipeline.")

    nodes = get_node_definitions_df()
    grid_meta = _validate_grid_identity(nodes)
    _write_grid_validation_report(grid_meta)
    elevation_dict = fetch_elevation(nodes)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": ",".join(str(lat) for lat in nodes["lat"]),
        "longitude": ",".join(str(lon) for lon in nodes["lon"]),
        "start_date": f"{start_year}-01-01",
        "end_date": f"{end_year}-12-31",
        "hourly": [
            "precipitation",
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
        ],
        "timezone": "Asia/Jakarta",
        "models": OPEN_METEO_MODEL,
    }

    print(f"Fetching {OPEN_METEO_MODEL} hourly weather data for {len(nodes)} canonical nodes...")
    responses = openmeteo.weather_api(url, params=params)
    if len(responses) != len(nodes):
        raise RuntimeError(
            f"Response count mismatch. Expected {len(nodes)} nodes, got {len(responses)} responses."
        )

    all_data = []
    for (_, node), response in zip(nodes.iterrows(), responses):
        data = _extract_hourly_response(response)
        name = node["name"]
        data["node"] = name
        data["node_role"] = node["role"]
        data["node_alias"] = node["alias"]
        data["requested_lat"] = float(node["lat"])
        data["requested_lon"] = float(node["lon"])
        data["grid_center_lat"] = float(grid_meta[name]["grid_center_lat"])
        data["grid_center_lon"] = float(grid_meta[name]["grid_center_lon"])
        data["model_mode"] = OPEN_METEO_MODEL
        data["is_main_node"] = 1 if name == MAIN_NODE_NAME else 0
        elev = float(elevation_dict[name])
        data["elevation"] = elev
        data["land_sea_mask"] = derive_land_sea_mask(elev)
        all_data.append(pd.DataFrame(data=data))

    df = pd.concat(all_data, ignore_index=True)
    node_order_map = {name: i for i, name in enumerate(NODE_NAMES)}
    df["_node_order"] = df["node"].map(node_order_map)
    if df["_node_order"].isna().any():
        bad = sorted(df.loc[df["_node_order"].isna(), "node"].unique().tolist())
        raise ValueError(f"Unexpected node names before canonical ordering: {bad}")
    df = df.sort_values(["date", "_node_order"]).reset_index(drop=True)
    df["precipitation_lag1"] = (
        df.groupby("node", sort=False)["precipitation"].shift(1).fillna(0.0)
    )

    ordered_cols = [
        "date",
        "node",
        "node_role",
        "node_alias",
        "is_main_node",
        "requested_lat",
        "requested_lon",
        "grid_center_lat",
        "grid_center_lon",
        "model_mode",
        *FINAL_FEATURE_COLS,
        "land_sea_mask",
        "precipitation",
    ]
    existing = [col for col in ordered_cols if col in df.columns]
    remaining = [col for col in df.columns if col not in existing]
    df = df[existing + remaining]
    df = df.drop(columns=["_node_order"], errors="ignore")

    os.makedirs("data/raw", exist_ok=True)
    output_path = f"data/raw/pangrango_era5_5node_{start_year}_{end_year}.parquet"
    df.to_parquet(output_path, index=False)

    # Write/refresh canonical alias pointer for default consumers.
    if start_year == 2005 and end_year == 2025:
        df.to_parquet(CANONICAL_DATA_PATH, index=False)

    print(f"\nData saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Nodes: {sorted(df['node'].unique().tolist())}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Grid validation report: {GRID_VALIDATION_PATH}")
    return df


def get_node_metadata() -> pd.DataFrame:
    """
    Return canonical node metadata with verified grid centers.
    """
    nodes = get_node_definitions_df()
    grid_meta = _validate_grid_identity(nodes)
    elevation_dict = fetch_elevation(nodes)
    nodes["grid_center_lat"] = nodes["name"].map({k: v["grid_center_lat"] for k, v in grid_meta.items()})
    nodes["grid_center_lon"] = nodes["name"].map({k: v["grid_center_lon"] for k, v in grid_meta.items()})
    nodes["elevation"] = nodes["name"].map(elevation_dict)
    nodes["land_sea_mask"] = nodes["elevation"].apply(derive_land_sea_mask)
    nodes["model_mode"] = OPEN_METEO_MODEL
    nodes["is_main_node"] = nodes["name"].apply(lambda n: 1 if n == MAIN_NODE_NAME else 0)
    return nodes


if __name__ == "__main__":
    fetch_era5_data(start_year=2005, end_year=2025)
