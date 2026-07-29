import os
import unittest

import torch


class CheckpointContractTest(unittest.TestCase):
    def test_diffusion_checkpoint_metadata_contract(self):
        path = "models/diffusion_chkpt.pth"
        self.assertTrue(os.path.exists(path), f"Missing checkpoint: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
        required = [
            "node_names",
            "node_roles",
            "node_coordinates",
            "main_node_identifier",
            "main_node_name",
            "graph_topology",
            "num_nodes",
            "target_node_policy",
            "context_policy",
        ]
        missing = [k for k in required if k not in cfg]
        self.assertFalse(missing, f"Missing metadata keys: {missing}")
        self.assertEqual(cfg["graph_topology"], "star")
        self.assertEqual(cfg["num_nodes"], 5)
        self.assertEqual(cfg["target_node_policy"], "main_node_only")

        # FIX #1: retrieval values are outcomes (num_targets each), so
        # retrieval_dim must equal num_targets * k_neighbors.
        num_targets = cfg.get("num_targets", len(cfg.get("target_cols", [])))
        self.assertEqual(cfg["retrieval_dim"], num_targets * cfg["k_neighbors"])
