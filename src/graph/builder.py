import numpy as np
import torch
from torch_geometric.data import Data

from src.config import MAIN_NODE_NAME, NODE_NAMES, build_star_edge_index, get_node_index_map


class PangrangoGraphBuilder:
    """
    Canonical graph builder for 5-node star topology.
    """

    def __init__(self, nodes_meta):
        self.nodes_meta = nodes_meta
        self.node_names = list(nodes_meta["name"]) if "name" in nodes_meta.columns else list(NODE_NAMES)
        self.node_to_idx = get_node_index_map(self.node_names)
        if MAIN_NODE_NAME not in self.node_to_idx:
            raise ValueError(f"Main node '{MAIN_NODE_NAME}' not found in nodes_meta")
        self.main_idx = self.node_to_idx[MAIN_NODE_NAME]
        self.num_nodes = len(self.node_names)
        self.edge_index, self.edge_attr = self._build_topology()

    def _build_topology(self):
        edge_index = build_star_edge_index(self.node_names)
        coords = self.nodes_meta.set_index("name")[["lat", "lon"]]
        distances = []
        for src_idx, dst_idx in edge_index.t().tolist():
            src_name = self.node_names[src_idx]
            dst_name = self.node_names[dst_idx]
            src = coords.loc[src_name].to_numpy(dtype=float)
            dst = coords.loc[dst_name].to_numpy(dtype=float)
            distances.append(float(np.linalg.norm(dst - src)))
        edge_attr = torch.tensor(distances, dtype=torch.float32).unsqueeze(1)
        return edge_index, edge_attr

    def build_dynamic_edges(self, wind_speed, wind_direction):
        """
        Keep star skeleton and derive directed weights from source-node wind speed.
        """
        weights = []
        for src_idx, _ in self.edge_index.t().tolist():
            weights.append(float(wind_speed[src_idx]))
        return self.edge_index, torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    def build_snapshot(self, feature_matrix, target=None, wind_speed=None, wind_dir=None):
        x = torch.tensor(feature_matrix, dtype=torch.float32)
        y = torch.tensor(target, dtype=torch.float32) if target is not None else None
        if wind_speed is not None and wind_dir is not None:
            edge_index, edge_attr = self.build_dynamic_edges(wind_speed, wind_dir)
        else:
            edge_index, edge_attr = self.edge_index, self.edge_attr
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def create_temporal_graphs(df, sequence_length=6):
    """
    Legacy stub intentionally kept for compatibility.
    """
    raise NotImplementedError("Use src.data.temporal_loader.TemporalGraphDataset for sequence construction.")

