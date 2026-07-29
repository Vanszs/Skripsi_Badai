import numpy as np
import pandas as pd
import unittest

from src.inference import create_inference_graphs, load_model_and_stats, run_inference_real


class InferenceSmokeTest(unittest.TestCase):
    def test_inference_smoke_main_node_output(self):
        model, stats, retrieval_db = load_model_and_stats("models/diffusion_chkpt.pth")
        cfg = model.config
        feature_cols = cfg["feature_cols"]
        node_names = cfg["node_names"]
        node_to_idx = {n: i for i, n in enumerate(node_names)}
        seq_len = cfg["seq_len"]

        df = pd.read_parquet(cfg["data_path"])
        df["date"] = pd.to_datetime(df["date"])
        timestamps = sorted(df["date"].unique())[:seq_len]
        arr = np.zeros((seq_len, len(node_names), len(feature_cols)), dtype=np.float32)
        for t_i, ts in enumerate(timestamps):
            g = df[df["date"] == ts]
            for _, row in g.iterrows():
                arr[t_i, node_to_idx[row["node"]], :] = row[feature_cols].values.astype(np.float32)

        arr_norm = (arr - stats["c_mean"].numpy()) / (stats["c_std"].numpy() + 1e-5)

        # FIX #6: inference graphs must carry edge_attr (per-edge distance).
        import torch
        cond_seq = torch.tensor(arr_norm, dtype=torch.float32)
        graphs = create_inference_graphs(cond_seq, cfg, device="cpu")
        self.assertTrue(hasattr(graphs[0], "edge_attr"))
        self.assertIsNotNone(graphs[0].edge_attr)
        self.assertEqual(graphs[0].edge_attr.shape[-1], 1)

        # FIX #1: retrieved analogs must be outcomes shaped [*, k, num_targets].
        num_targets = cfg.get("num_targets", 3)
        main_idx = node_names.index(cfg["main_node_name"])
        context_last = cond_seq[-1, main_idx, :].unsqueeze(0)
        retrieved = retrieval_db.query(context_last.numpy(), k=cfg["k_neighbors"])
        self.assertEqual(tuple(retrieved.shape), (1, cfg["k_neighbors"], num_targets))

        out = run_inference_real(arr_norm, model, stats, retrieval_db, num_samples=4, device="cpu")

        self.assertEqual(out["main_node_name"], "MAIN")
        self.assertEqual(out["target_node_policy"], "main_node_only")
        self.assertEqual(out["raw"].shape, (4, 3))
        self.assertTrue(np.isfinite(out["raw"]).all())
