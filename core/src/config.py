"""
Central configuration and canonical system contract for the 5-node migration.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Canonical data/model contract
# ---------------------------------------------------------------------------
OPEN_METEO_MODEL = "era5"
GRAPH_TOPOLOGY = "star"
TARGET_NODE_POLICY = "main_node_only"
CONTEXT_POLICY = "main_node_context"

# Node order is mandatory and must be stable across ingestion, loader, training,
# checkpoint, inference, and evaluation.
NODE_DEFINITIONS: List[Dict[str, object]] = [
    {
        "name": "MAIN",
        "role": "main",
        "alias": "Puncak",
        "lat": -6.75,
        "lon": 107.00,
        "elevation_m": 1529.0,
    },
    {
        "name": "UP",
        "role": "surrounding",
        "alias": "Up",
        "lat": -6.50,
        "lon": 107.00,
        "elevation_m": 162.0,
    },
    {
        "name": "DOWN",
        "role": "surrounding",
        "alias": "Down",
        "lat": -7.00,
        "lon": 107.00,
        "elevation_m": 0.0,
    },
    {
        "name": "LEFT",
        "role": "surrounding",
        "alias": "Left",
        "lat": -6.75,
        "lon": 106.75,
        "elevation_m": 823.0,
    },
    {
        "name": "RIGHT",
        "role": "surrounding",
        "alias": "Right",
        "lat": -6.75,
        "lon": 107.25,
        "elevation_m": 288.0,
    },
]

NODE_NAMES = [node["name"] for node in NODE_DEFINITIONS]
NODE_ROLES = {node["name"]: node["role"] for node in NODE_DEFINITIONS}
NODE_COORDINATES = {node["name"]: (float(node["lat"]), float(node["lon"])) for node in NODE_DEFINITIONS}
NODE_ELEVATIONS = {node["name"]: float(node["elevation_m"]) for node in NODE_DEFINITIONS}
NUM_NODES = len(NODE_DEFINITIONS)

MAIN_NODE_NAME = "MAIN"
MAIN_NODE_IDENTIFIER = NODE_COORDINATES[MAIN_NODE_NAME]

STAR_EDGES = [
    (MAIN_NODE_NAME, "UP"),
    ("UP", MAIN_NODE_NAME),
    (MAIN_NODE_NAME, "DOWN"),
    ("DOWN", MAIN_NODE_NAME),
    (MAIN_NODE_NAME, "LEFT"),
    ("LEFT", MAIN_NODE_NAME),
    (MAIN_NODE_NAME, "RIGHT"),
    ("RIGHT", MAIN_NODE_NAME),
]
STAR_EDGE_COUNT = len(STAR_EDGES)

FINAL_TARGET_COLS = [
    "precipitation",
    "wind_speed_10m",
    "relative_humidity_2m",
]

# Physical upper bound for hourly precipitation (mm/h). Dataset max ~21.5; this generous
# cap is numerical hygiene only (prevents expm1 blow-up on out-of-distribution inputs),
# well above any real value so it never distorts healthy predictions.
PRECIP_PHYSICAL_MAX_MM = 60.0

FINAL_FEATURE_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "dewpoint_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
    "precipitation_lag1",
    "elevation",
]

OPTIONAL_LEGACY_RENAMES = {
    "cloudcover": "cloud_cover",
    "dew_point_2m": "dewpoint_2m",
}


def harmonize_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize legacy column names to the final contract used by the pipeline.
    """
    rename_map = {old: new for old, new in OPTIONAL_LEGACY_RENAMES.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_feature_schema(df: pd.DataFrame, feature_cols=None, target_cols=None) -> None:
    """
    Raise a clear error when the dataset does not match the required contract.
    """
    feature_cols = feature_cols or FINAL_FEATURE_COLS
    target_cols = target_cols or FINAL_TARGET_COLS
    missing = [col for col in [*feature_cols, *target_cols] if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset schema mismatch. Missing required columns: {missing}")


def validate_feature_values(df: pd.DataFrame, feature_cols=None) -> None:
    """
    Raise an error if a required feature exists in schema but contains no usable values.
    """
    feature_cols = feature_cols or FINAL_FEATURE_COLS
    invalid = [col for col in feature_cols if col in df.columns and df[col].notna().sum() == 0]
    if invalid:
        raise ValueError(f"Dataset contains required features with all-NaN values: {invalid}")


def get_node_dataframe() -> pd.DataFrame:
    """
    Return canonical node table in mandatory order.
    """
    return pd.DataFrame(NODE_DEFINITIONS).copy()


def get_node_index_map(node_names: List[str] | None = None) -> Dict[str, int]:
    """
    Return node -> index mapping for stable tensor order.
    """
    names = node_names or NODE_NAMES
    return {name: i for i, name in enumerate(names)}


def build_star_edge_index(node_names: List[str] | None = None) -> torch.Tensor:
    """
    Build canonical star edge index in directed form.
    """
    names = node_names or NODE_NAMES
    idx = get_node_index_map(names)
    missing = sorted({name for edge in STAR_EDGES for name in edge if name not in idx})
    if missing:
        raise ValueError(f"Cannot build star edges. Missing nodes: {missing}")
    sources = [idx[src] for src, _ in STAR_EDGES]
    targets = [idx[dst] for _, dst in STAR_EDGES]
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    if edge_index.shape[1] != STAR_EDGE_COUNT:
        raise ValueError(
            f"Star edge count mismatch. Expected {STAR_EDGE_COUNT}, got {edge_index.shape[1]}"
        )
    return edge_index


def build_star_edge_attr(node_names: List[str] | None = None) -> torch.Tensor:
    """
    Build static per-edge attribute for the star graph. Shape: [num_edges, 1].
    Feature: euclidean lat/lon distance (degrees).

    NOTE (honest limitation): on the canonical 0.25-deg grid every surrounding node is
    exactly 0.25 deg from MAIN, so this distance is constant across edges and therefore
    non-informative; spatial signal comes from the topology (edge_index) and node features,
    not from edge weights. An elevation-difference edge feature was tested but empirically
    degraded training convergence, so it was reverted. Elevation context is retained per
    node (NODE_ELEVATIONS) and documented rather than injected as an edge weight.
    """
    names = node_names or NODE_NAMES
    distances = []
    for src, dst in STAR_EDGES:
        if src not in NODE_COORDINATES or dst not in NODE_COORDINATES:
            raise ValueError(f"Cannot build edge attr. Missing coordinates for edge {(src, dst)}")
        (slat, slon) = NODE_COORDINATES[src]
        (dlat, dlon) = NODE_COORDINATES[dst]
        distances.append(((dlat - slat) ** 2 + (dlon - slon) ** 2) ** 0.5)
    edge_attr = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
    if edge_attr.shape[0] != STAR_EDGE_COUNT:
        raise ValueError(
            f"Star edge attr count mismatch. Expected {STAR_EDGE_COUNT}, got {edge_attr.shape[0]}"
        )
    return edge_attr


def get_checkpoint_node_metadata() -> Dict[str, object]:
    """
    Metadata block that must be stored in model checkpoints.
    """
    return {
        "node_names": list(NODE_NAMES),
        "node_roles": dict(NODE_ROLES),
        "node_coordinates": dict(NODE_COORDINATES),
        "main_node_identifier": MAIN_NODE_IDENTIFIER,
        "main_node_name": MAIN_NODE_NAME,
        "graph_topology": GRAPH_TOPOLOGY,
        "num_nodes": NUM_NODES,
        "target_node_policy": TARGET_NODE_POLICY,
        "context_policy": CONTEXT_POLICY,
        "star_edges": list(STAR_EDGES),
    }

