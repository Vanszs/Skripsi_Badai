"""
Temporal DataLoader for 5-node star spatio-temporal graph sequences.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, Data

from src.config import (
    FINAL_TARGET_COLS,
    MAIN_NODE_NAME,
    NODE_NAMES,
    build_star_edge_attr,
    build_star_edge_index,
    get_node_index_map,
)


class TemporalGraphDataset(Dataset):
    """
    Dataset that creates sliding window sequences of canonical graph snapshots.
    Output target and context are main-node-only (no node averaging).
    """

    TARGET_COLS = FINAL_TARGET_COLS

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: Optional[List[str]] = None,
        seq_len: int = 6,
        node_names: Optional[List[str]] = None,
        main_node_name: str = MAIN_NODE_NAME,
        edge_index: Optional[torch.Tensor] = None,
        stats: Optional[dict] = None,
    ):
        self.df = df.copy()
        self.feature_cols = list(feature_cols)
        self.target_cols = list(target_cols) if target_cols else list(self.TARGET_COLS)
        self.num_targets = len(self.target_cols)
        self.seq_len = int(seq_len)
        self.node_names = list(node_names) if node_names else list(NODE_NAMES)
        self.num_nodes = len(self.node_names)
        self.node_to_idx = get_node_index_map(self.node_names)
        if main_node_name not in self.node_to_idx:
            raise ValueError(
                f"main_node_name '{main_node_name}' not found in node_names {self.node_names}"
            )
        self.main_node_name = main_node_name
        self.main_node_idx = self.node_to_idx[self.main_node_name]
        self.edge_index = edge_index if edge_index is not None else build_star_edge_index(self.node_names)
        self.edge_attr = build_star_edge_attr(self.node_names)
        self.stats = stats
        self._prepare_data()

    def _assert_node_contract(self) -> None:
        unique_nodes = set(self.df["node"].unique().tolist())
        expected_nodes = set(self.node_names)
        missing = sorted(expected_nodes - unique_nodes)
        unknown = sorted(unique_nodes - expected_nodes)
        if missing:
            raise ValueError(f"Missing required nodes in dataset: {missing}")
        if unknown:
            raise ValueError(f"Unknown nodes found in dataset: {unknown}")

        # Every timestamp must contain each canonical node exactly once.
        grouped = self.df.groupby("date")["node"]
        wrong_count = grouped.size() != self.num_nodes
        if bool(wrong_count.any()):
            bad_ts = grouped.size()[wrong_count].index[:5].tolist()
            raise ValueError(
                f"Node count mismatch at timestamps (sample): {bad_ts}. "
                f"Expected {self.num_nodes} rows per timestamp."
            )

        unique_count = grouped.nunique() != self.num_nodes
        if bool(unique_count.any()):
            bad_ts = grouped.nunique()[unique_count].index[:5].tolist()
            raise ValueError(
                f"Duplicate or missing node names at timestamps (sample): {bad_ts}."
            )

        # Strict raw row-order check: each timestamp must already be in canonical order.
        order_per_ts = self.df.groupby("date", sort=False)["node"].agg(list)
        bad_order = []
        for ts, names in order_per_ts.items():
            if names != self.node_names:
                bad_order.append((ts, names))
            if len(bad_order) >= 5:
                break
        if bad_order:
            formatted = [f"{ts}: {names}" for ts, names in bad_order]
            raise ValueError(
                "Wrong node order detected. Expected per-timestamp order "
                f"{self.node_names}. Samples: {formatted}"
            )

    def _prepare_data(self) -> None:
        self._assert_node_contract()

        self.timestamps = np.sort(self.df["date"].unique())
        self.num_timestamps = len(self.timestamps)

        self.df["_node_idx"] = self.df["node"].map(self.node_to_idx)
        if self.df["_node_idx"].isna().any():
            unmapped = sorted(self.df.loc[self.df["_node_idx"].isna(), "node"].unique().tolist())
            raise ValueError(f"Unmapped nodes after node index mapping: {unmapped}")
        self.df["_node_idx"] = self.df["_node_idx"].astype(int)
        self.df = self.df.sort_values(["date", "_node_idx"]).reset_index(drop=True)

        ts_to_idx = {ts: i for i, ts in enumerate(self.timestamps)}
        self.df["_ts_idx"] = self.df["date"].map(ts_to_idx)
        if self.df["_ts_idx"].isna().any():
            raise ValueError("Timestamp mapping failure detected.")
        self.df["_ts_idx"] = self.df["_ts_idx"].astype(int)

        # Within each timestamp, node indices must match strict canonical order 0..N-1.
        expected = list(range(self.num_nodes))
        for ts, g in self.df.groupby("date", sort=False):
            ordered = g["_node_idx"].tolist()
            if ordered != expected:
                raise ValueError(
                    f"Wrong node order/content at timestamp {ts}. "
                    f"Expected node indices {expected}, got {ordered}."
                )

        num_features = len(self.feature_cols)
        feature_data = np.zeros((self.num_timestamps, self.num_nodes, num_features), dtype=np.float32)
        target_data = np.zeros((self.num_timestamps, self.num_nodes, self.num_targets), dtype=np.float32)

        ts_indices = self.df["_ts_idx"].values
        node_indices = self.df["_node_idx"].values
        feature_data[ts_indices, node_indices] = self.df[self.feature_cols].values.astype(np.float32)
        for k, col in enumerate(self.target_cols):
            if col in self.df.columns:
                target_data[ts_indices, node_indices, k] = self.df[col].values.astype(np.float32)

        self.features = torch.tensor(feature_data, dtype=torch.float32)
        self.targets = torch.tensor(target_data, dtype=torch.float32)
        if self.stats:
            self._normalize()
        self.valid_indices = list(range(self.seq_len, self.num_timestamps))

        self.df.drop(columns=["_node_idx", "_ts_idx"], inplace=True, errors="ignore")

        print("[TemporalGraphDataset] Prepared:")
        print(f"  Timestamps: {self.num_timestamps}")
        print(f"  Nodes: {self.num_nodes} ({self.node_names})")
        print(f"  Main node: {self.main_node_name} @ idx {self.main_node_idx}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"  Targets: {self.num_targets} ({self.target_cols})")
        print(f"  Valid samples: {len(self.valid_indices)}")

    def _normalize(self):
        self.targets_transformed = self.targets.clone()
        self.targets_transformed[:, :, 0] = torch.log1p(self.targets[:, :, 0])

        # Stats are main-node-based by contract, but still vectorized by feature/target dimension.
        t_mean = self.stats.get("t_mean")
        t_std = self.stats.get("t_std")
        c_mean = self.stats.get("c_mean")
        c_std = self.stats.get("c_std")
        if t_mean is None or t_std is None or c_mean is None or c_std is None:
            raise ValueError("Normalization stats missing required keys t_mean/t_std/c_mean/c_std")

        self.targets_norm = (self.targets_transformed - t_mean) / (t_std + 1e-5)
        self.features_norm = (self.features - c_mean) / (c_std + 1e-5)

    def set_precomputed_retrieval(self, retrieved_tensor: torch.Tensor) -> None:
        if retrieved_tensor.ndim == 3:
            retrieved_tensor = retrieved_tensor.view(retrieved_tensor.shape[0], -1)
        self.precomputed_retrieval = retrieved_tensor
        print(f"  Pre-computed retrieval set: {retrieved_tensor.shape}")

    def get_all_contexts_for_valid_indices(self) -> torch.Tensor:
        """
        Return [num_samples, feature_dim] contexts from main node at t-1.
        """
        t_minus_1 = np.array(self.valid_indices) - 1
        feats = self.features_norm if hasattr(self, "features_norm") else self.features
        return feats[t_minus_1, self.main_node_idx, :]

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        t = self.valid_indices[idx]
        graphs = []
        for i in range(self.seq_len):
            t_idx = t - self.seq_len + i
            node_feats = self.features_norm[t_idx] if hasattr(self, "features_norm") else self.features[t_idx]
            graph = Data(x=node_feats, edge_index=self.edge_index, edge_attr=self.edge_attr)
            graphs.append(graph)

        targets = self.targets_norm if hasattr(self, "targets_norm") else self.targets
        feats = self.features_norm if hasattr(self, "features_norm") else self.features
        target = targets[t, self.main_node_idx, :]  # main-node-only target
        context = feats[t - 1, self.main_node_idx, :]  # main-node-only context

        if hasattr(self, "precomputed_retrieval") and self.precomputed_retrieval is not None:
            return graphs, target, context, self.precomputed_retrieval[idx]
        return graphs, target, context


def collate_temporal_graphs(batch: List[Tuple]):
    has_retrieval = len(batch[0]) == 4
    seq_len = len(batch[0][0])
    timestep_graphs = [[] for _ in range(seq_len)]
    targets = []
    contexts = []
    retrievals = [] if has_retrieval else None

    for sample in batch:
        graphs = sample[0]
        targets.append(sample[1])
        contexts.append(sample[2])
        for t, g in enumerate(graphs):
            timestep_graphs[t].append(g)
        if has_retrieval:
            retrievals.append(sample[3])

    batched_graphs = [Batch.from_data_list(graphs) for graphs in timestep_graphs]
    targets = torch.stack(targets)
    contexts = torch.stack(contexts)
    if has_retrieval:
        retrievals = torch.stack(retrievals)
        return batched_graphs, targets, contexts, retrievals
    return batched_graphs, targets, contexts


def create_temporal_dataloader(
    df: pd.DataFrame,
    feature_cols: List[str],
    seq_len: int = 6,
    batch_size: int = 32,
    stats: Optional[dict] = None,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TemporalGraphDataset(
        df=df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stats=stats,
        node_names=NODE_NAMES,
        main_node_name=MAIN_NODE_NAME,
        edge_index=build_star_edge_index(NODE_NAMES),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_temporal_graphs,
    )


def crosscheck_temporal_loader():
    """
    Lightweight crosscheck for shape and contract validation.
    """
    print("=" * 60)
    print("CROSSCHECK: Temporal DataLoader (5-node star, main-node target)")
    print("=" * 60)
    dates = pd.date_range("2024-01-01", periods=48, freq="H")
    rows = []
    for date in dates:
        for i, node in enumerate(NODE_NAMES):
            rows.append(
                {
                    "date": date,
                    "node": node,
                    "temperature_2m": float(i + 1),
                    "relative_humidity_2m": 70.0 + i,
                    "dewpoint_2m": 20.0 + i,
                    "surface_pressure": 900.0 + i,
                    "wind_speed_10m": 2.0 + i,
                    "wind_direction_10m": 180.0,
                    "cloud_cover": 50.0,
                    "precipitation_lag1": 0.1 * i,
                    "elevation": 100.0 * (i + 1),
                    "precipitation": 0.2 * i,
                }
            )
    df = pd.DataFrame(rows)
    dataset = TemporalGraphDataset(
        df=df,
        feature_cols=[
            "temperature_2m",
            "relative_humidity_2m",
            "dewpoint_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "cloud_cover",
            "precipitation_lag1",
            "elevation",
        ],
        seq_len=6,
    )
    graphs, target, context = dataset[0]
    print(f"Graphs: {len(graphs)} timesteps")
    print(f"Graph[0].x shape: {graphs[0].x.shape}")
    print(f"Graph[0].edge_index shape: {graphs[0].edge_index.shape}")
    print(f"Target shape (main-node-only): {target.shape}")
    print(f"Context shape (main-node-only): {context.shape}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    crosscheck_temporal_loader()
