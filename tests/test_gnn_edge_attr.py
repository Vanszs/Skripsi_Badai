"""Unit tests for edge_attr behavior in SpatioTemporalGNN."""
import unittest

import torch
from torch_geometric.data import Data

from src.config import NODE_NAMES, build_star_edge_attr, build_star_edge_index
from src.models.gnn import SpatioTemporalGNN


class GNNEdgeAttrTest(unittest.TestCase):
    def _make_graphs(self, edge_attr_value, node_features=9, hidden_dim=8):
        edge_index = build_star_edge_index(NODE_NAMES)
        edge_attr = torch.full((edge_index.shape[1], 1), edge_attr_value, dtype=torch.float32)
        x = torch.randn(5, node_features)
        # batch assignment: all 5 nodes belong to the same graph sample.
        batch = torch.zeros(5, dtype=torch.long)
        graphs = [Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch) for _ in range(6)]
        return graphs

    def test_gnn_output_shape(self):
        gnn = SpatioTemporalGNN(
            node_features=9, hidden_dim=8, output_dim=8, seq_len=6
        )
        graphs = self._make_graphs(0.25)
        out = gnn(graphs)
        self.assertEqual(out.shape, (1, 8))

    def test_edge_attr_is_constant(self):
        """
        On the canonical ERA5 0.25-deg star grid every edge has the same
        distance, so edge_attr must be constant across all edges.
        """
        edge_attr = build_star_edge_attr(NODE_NAMES)
        self.assertTrue((edge_attr == edge_attr[0]).all())
        self.assertAlmostEqual(float(edge_attr[0].item()), 0.25, places=5)
