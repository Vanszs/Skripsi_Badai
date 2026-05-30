"""
Robust precipitation-only crosscheck on weekly windows.

Compares:
- Rain-specialized ST-Graph + Retrieval-Diffusion model
- Persistence baseline

Windows:
- driest week
- median-rain week
- wettest week
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (  # noqa: E402
    FINAL_FEATURE_COLS,
    FINAL_TARGET_COLS,
    MAIN_NODE_NAME,
    NODE_NAMES,
    harmonize_weather_columns,
    validate_feature_schema,
    validate_feature_values,
)
from src.data.ingest import CANONICAL_DATA_PATH  # noqa: E402
from src.inference import create_inference_graphs, load_model_and_stats  # noqa: E402
from src.train import compute_stats_from_training, temporal_split  # noqa: E402


WINDOW_HOURS = 24 * 7
EVENT_THRESHOLDS = (0.1, 1.0)


@dataclass
class WeekWindow:
    name: str
    start_idx: int
    end_idx: int
    precip_sum: float


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return float("nan")
    aa = a - a.mean()
    bb = b - b.mean()
    denom = float(np.sqrt(np.sum(aa * aa)) * np.sqrt(np.sum(bb * bb)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def _event_metrics(actual: np.ndarray, pred: np.ndarray, threshold: float) -> Dict[str, float]:
    act = actual >= threshold
    pdn = pred >= threshold
    tp = float(np.logical_and(act, pdn).sum())
    fp = float(np.logical_and(~act, pdn).sum())
    fn = float(np.logical_and(act, ~pdn).sum())
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    csi = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    return {"pod": pod, "far": far, "csi": csi}


def _pick_week_windows(main_df: pd.DataFrame) -> List[WeekWindow]:
    starts = list(range(0, max(0, len(main_df) - WINDOW_HOURS + 1), WINDOW_HOURS))
    candidates = []
    for s in starts:
        e = s + WINDOW_HOURS
        if e > len(main_df):
            continue
        precip_sum = float(main_df.iloc[s:e]["precipitation"].sum())
        candidates.append((s, e, precip_sum))
    if len(candidates) < 3:
        raise ValueError("Not enough weekly windows to build robust crosscheck.")

    candidates_sorted = sorted(candidates, key=lambda x: x[2])
    dry = candidates_sorted[0]
    wet = candidates_sorted[-1]
    median_precip = np.median([x[2] for x in candidates_sorted])
    med = min(candidates_sorted, key=lambda x: abs(x[2] - median_precip))

    picked = [
        WeekWindow("driest_week", dry[0], dry[1], dry[2]),
        WeekWindow("median_week", med[0], med[1], med[2]),
        WeekWindow("wettest_week", wet[0], wet[1], wet[2]),
    ]
    return picked


def _build_test_frames(data_path: str):
    df = pd.read_parquet(data_path)
    df = harmonize_weather_columns(df)
    validate_feature_schema(df, FINAL_FEATURE_COLS, FINAL_TARGET_COLS)
    validate_feature_values(df, FINAL_FEATURE_COLS)

    ckpt = torch.load("models/diffusion_chkpt.pth", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    feature_cols = cfg.get("feature_cols", FINAL_FEATURE_COLS)
    train_end = cfg.get("train_end", "2018-12-31")
    val_end = cfg.get("val_end", "2021-12-31")

    train_df, _, test_df = temporal_split(df, train_end, val_end)
    stats = compute_stats_from_training(train_df, feature_cols, main_node_name=MAIN_NODE_NAME)
    test_df = test_df.copy()
    test_df["date"] = pd.to_datetime(test_df["date"])
    if test_df["date"].dt.tz is not None:
        test_df["date"] = test_df["date"].dt.tz_localize(None)

    main_df = (
        test_df[test_df["node"] == MAIN_NODE_NAME]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )
    per_node_data = {}
    for node in NODE_NAMES:
        ndf = (
            test_df[test_df["node"] == node]
            .copy()
            .sort_values("date")
            .reset_index(drop=True)
        )
        if ndf.empty:
            raise ValueError(f"Test split missing node '{node}'")
        per_node_data[node] = ndf
    return main_df, per_node_data, feature_cols, stats


def _get_per_node_sequence(
    per_node_data: Dict[str, pd.DataFrame],
    idx: int,
    seq_len: int,
    feature_cols: List[str],
    stats: Dict[str, torch.Tensor],
) -> torch.Tensor:
    c_mean = stats["c_mean"].numpy()
    c_std = stats["c_std"].numpy()
    sequences = []
    for t in range(idx - seq_len, idx):
        node_feats = []
        for node in NODE_NAMES:
            ndf = per_node_data[node]
            row = ndf.iloc[t]
            feat = np.array([row[c] if c in row.index else 0.0 for c in feature_cols], dtype=np.float32)
            node_feats.append(feat)
        sequences.append(np.stack(node_feats))
    seq = np.stack(sequences)
    seq_norm = (seq - c_mean) / (c_std + 1e-5)
    return torch.tensor(seq_norm, dtype=torch.float32)


def run_crosscheck(
    data_path: str,
    num_ensemble: int,
    num_inference_steps: int,
    device: str,
):
    os.makedirs("result_test/nowcasting_hourly_week", exist_ok=True)

    model_wrapper, stats_ckpt, retrieval_db = load_model_and_stats("models/diffusion_chkpt.pth")
    model_wrapper.to(device)
    model_wrapper.eval()
    config = model_wrapper.config
    st_gnn = model_wrapper.st_gnn
    forecaster = model_wrapper.forecaster

    main_df, per_node_data, feature_cols, stats = _build_test_frames(data_path)
    c_mean = stats["c_mean"].numpy()
    c_std = stats["c_std"].numpy()
    t_mean_t = stats["t_mean"].to(device)
    t_std_t = stats["t_std"].to(device)
    seq_len = int(config["seq_len"])

    features_raw = main_df[feature_cols].values.astype(np.float32)
    features_norm = (features_raw - c_mean) / (c_std + 1e-5)
    timestamps = pd.to_datetime(main_df["date"]).reset_index(drop=True)
    actual_precip = main_df["precipitation"].values.astype(np.float32)

    windows = _pick_week_windows(main_df)
    summary_rows = []

    for window in windows:
        start_idx = window.start_idx
        end_idx = window.end_idx
        eval_start = max(start_idx + seq_len, seq_len)
        eval_range = range(eval_start, end_idx)

        rows = []
        model_preds = []
        baseline_preds = []
        actuals = []

        for idx in eval_range:
            target = float(actual_precip[idx])
            baseline = float(actual_precip[idx - 1])

            main_ctx_seq = torch.tensor(features_norm[idx - seq_len : idx], dtype=torch.float32, device=device)
            context_last = main_ctx_seq[-1].unsqueeze(0)
            per_node_seq = _get_per_node_sequence(per_node_data, idx, seq_len, feature_cols, stats).to(device)
            graphs = create_inference_graphs(per_node_seq, config, device=device)

            with torch.no_grad():
                graph_emb = st_gnn(graphs)
                retrieved = retrieval_db.query(context_last.cpu().numpy(), k=config["k_neighbors"]).to(device)
                samples = forecaster.sample_fast(
                    condition=context_last,
                    retrieved=retrieved,
                    graph_emb=graph_emb,
                    num_samples=num_ensemble,
                    num_inference_steps=num_inference_steps,
                )
                samples_denorm = samples * t_std_t + t_mean_t
                samples_denorm[:, 0] = torch.clamp(
                    torch.expm1(torch.clamp(samples_denorm[:, 0], max=20.0)),
                    min=0.0,
                )

                rain_cfg = config.get("rain_specialization", {})
                if rain_cfg.get("enabled", False):
                    wet_prob = forecaster.model.compute_wet_probability(
                        context=context_last,
                        retrieved=retrieved,
                        graph_emb=graph_emb,
                    )
                    wet_threshold = float(rain_cfg.get("wet_probability_threshold", 0.5))
                    if float(wet_prob.squeeze().item()) < wet_threshold:
                        samples_denorm[:, 0] = 0.0

                pred = float(torch.median(samples_denorm[:, 0]).item())

            rows.append(
                {
                    "timestamp": timestamps.iloc[idx],
                    "actual_precipitation": target,
                    "pred_model_precipitation": pred,
                    "pred_persistence_precipitation": baseline,
                }
            )
            actuals.append(target)
            model_preds.append(pred)
            baseline_preds.append(baseline)

        week_df = pd.DataFrame(rows)
        week_df.to_csv(
            f"result_test/nowcasting_hourly_week/{window.name}_actual_vs_pred.csv",
            index=False,
        )

        actual_arr = np.array(actuals, dtype=np.float32)
        model_arr = np.array(model_preds, dtype=np.float32)
        base_arr = np.array(baseline_preds, dtype=np.float32)

        model_rmse = float(np.sqrt(np.mean((model_arr - actual_arr) ** 2)))
        base_rmse = float(np.sqrt(np.mean((base_arr - actual_arr) ** 2)))
        model_mae = float(np.mean(np.abs(model_arr - actual_arr)))
        base_mae = float(np.mean(np.abs(base_arr - actual_arr)))
        model_corr = _corr(model_arr, actual_arr)
        base_corr = _corr(base_arr, actual_arr)

        row = {
            "window": window.name,
            "precip_sum_mm": float(window.precip_sum),
            "n_samples": int(len(actual_arr)),
            "model_rmse": model_rmse,
            "persistence_rmse": base_rmse,
            "rmse_delta_model_minus_persistence": model_rmse - base_rmse,
            "model_mae": model_mae,
            "persistence_mae": base_mae,
            "mae_delta_model_minus_persistence": model_mae - base_mae,
            "model_corr": model_corr,
            "persistence_corr": base_corr,
            "corr_delta_model_minus_persistence": model_corr - base_corr,
        }
        for thr in EVENT_THRESHOLDS:
            model_ev = _event_metrics(actual_arr, model_arr, threshold=thr)
            base_ev = _event_metrics(actual_arr, base_arr, threshold=thr)
            key = str(thr).replace(".", "p")
            row[f"model_csi_thr_{key}"] = model_ev["csi"]
            row[f"persistence_csi_thr_{key}"] = base_ev["csi"]
            row[f"csi_delta_thr_{key}"] = model_ev["csi"] - base_ev["csi"]
            row[f"model_pod_thr_{key}"] = model_ev["pod"]
            row[f"persistence_pod_thr_{key}"] = base_ev["pod"]
            row[f"model_far_thr_{key}"] = model_ev["far"]
            row[f"persistence_far_thr_{key}"] = base_ev["far"]
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = "result_test/nowcasting_hourly_week/weekly_crosscheck_rain_specialized.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved robust weekly crosscheck to: {summary_path}")
    print(summary_df)

    # Backward-compatible alias for quick inspection.
    median_row = summary_df[summary_df["window"] == "median_week"]
    if not median_row.empty:
        median_csv = "result_test/nowcasting_hourly_week/actual_vs_pred_hourly_1week.csv"
        src_csv = "result_test/nowcasting_hourly_week/median_week_actual_vs_pred.csv"
        pd.read_csv(src_csv).to_csv(median_csv, index=False)
        print(f"Updated 1-week alias CSV: {median_csv}")

    # Persist run metadata.
    meta = {
        "data_path": data_path,
        "checkpoint_path": "models/diffusion_chkpt.pth",
        "num_ensemble": int(num_ensemble),
        "num_inference_steps": int(num_inference_steps),
        "device": str(device),
        "windows": [w.__dict__ for w in windows],
        "target": "precipitation",
        "comparison": "model_vs_persistence",
    }
    with open(
        "result_test/nowcasting_hourly_week/weekly_crosscheck_rain_specialized_meta.json",
        "w",
        encoding="utf-8",
    ) as f:
        import json

        json.dump(meta, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CANONICAL_DATA_PATH)
    parser.add_argument("--num-ensemble", type=int, default=20)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_crosscheck(
        data_path=args.data_path,
        num_ensemble=args.num_ensemble,
        num_inference_steps=args.num_inference_steps,
        device=args.device,
    )
