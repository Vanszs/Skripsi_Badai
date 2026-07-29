"""Unit tests for src.inference.create_inference_graphs shape contract."""
import unittest

import torch

from src.config import NODE_NAMES
from src.inference import create_inference_graphs


class InferenceGraphsTest(unittest.TestCase):
    def _config(self):
        return {
            "seq_len": 6,
            "node_names": NODE_NAMES,
            "num_nodes": 5,
            "main_node_name": "MAIN",
            "graph_dim": 8,
            "feature_cols": ["f"],
        }

    def test_create_inference_graphs_correct_shape(self):
        cfg = self._config()
        cond = torch.randn(6, 5, 9)  # [seq_len, nodes, features]
        graphs = create_inference_graphs(cond, cfg, device="cpu")
        self.assertEqual(len(graphs), 6)
        self.assertEqual(graphs[0].x.shape, (5, 9))
        self.assertTrue(hasattr(graphs[0], "edge_attr"))
        self.assertEqual(graphs[0].edge_attr.shape, (8, 1))

    def test_create_inference_graphs_rejects_wrong_shape(self):
        cfg = self._config()
        cond = torch.randn(6, 9)  # missing node dimension
        with self.assertRaisesRegex(ValueError, "condition_sequence must be 3-dimensional"):
            create_inference_graphs(cond, cfg, device="cpu")

    def test_create_inference_graphs_rejects_wrong_node_count(self):
        cfg = self._config()
        cond = torch.randn(6, 3, 9)  # 3 nodes instead of 5
        with self.assertRaisesRegex(ValueError, "Condition node dimension mismatch"):
            create_inference_graphs(cond, cfg, device="cpu")
