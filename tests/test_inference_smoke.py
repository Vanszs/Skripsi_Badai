import numpy as np
import pandas as pd
import unittest

from src.inference import load_model_and_stats, run_inference_real


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
        out = run_inference_real(arr_norm, model, stats, retrieval_db, num_samples=4, device="cpu")

        self.assertEqual(out["main_node_name"], "MAIN")
        self.assertEqual(out["target_node_policy"], "main_node_only")
        self.assertEqual(out["raw"].shape, (4, 3))
        self.assertTrue(np.isfinite(out["raw"]).all())
