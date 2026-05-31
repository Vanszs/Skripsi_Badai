"""
Reproducible weekly one-step nowcasting crosscheck (3 variables).

Single source for the weekly artifacts under result_test/nowcasting_hourly_week/.
- Variables: precipitation, wind_speed_10m, relative_humidity_2m
- Models compared: full RA-Diffusion model, persistence, MLP baseline
- Windows: driest / median / wettest week (chosen on test split)
- Protocol: one-step hourly with actual update (no recursive closed loop)

Deterministic: fixed seeds; outputs CSV + JSON only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
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
    PRECIP_PHYSICAL_MAX_MM,
    harmonize_weather_columns,
    validate_feature_schema,
    validate_feature_values,
)
from src.data.ingest import CANONICAL_DATA_PATH  # noqa: E402
from src.inference import create_inference_graphs, load_model_and_stats  # noqa: E402
from src.models.mlp_baseline import MLPBaseline  # noqa: E402
from src.train import compute_stats_from_training, temporal_split  # noqa: E402


WINDOW_HOURS = 24 * 7
EVENT_THRESHOLDS = (0.1, 1.0)
VAR_NAMES = ["precipitation", "wind_speed_10m", "relative_humidity_2m"]
SEED = 1234
OUT_DIR = "result_test/nowcasting_hourly_week"


@dataclass
class WeekWindow:
    name: str
    start_idx: int
    end_idx: int
    precip_sum: float


def _seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    return [
        WeekWindow("driest_week", dry[0], dry[1], dry[2]),
        WeekWindow("median_week", med[0], med[1], med[2]),
        WeekWindow("wettest_week", wet[0], wet[1], wet[2]),
    ]


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


def _get_per_node_sequence(per_node_data, idx, seq_len, feature_cols, stats) -> torch.Tensor:
    c_mean = stats["c_mean"].numpy()
    c_std = stats["c_std"].numpy()
    sequences = []
    for t in range(idx - seq_len, idx):
        node_feats = []
        for node in NODE_NAMES:
            row = per_node_data[node].iloc[t]
            feat = np.array([row[c] if c in row.index else 0.0 for c in feature_cols], dtype=np.float32)
            node_feats.append(feat)
        sequences.append(np.stack(node_feats))
    seq = np.stack(sequences)
    seq_norm = (seq - c_mean) / (c_std + 1e-5)
    return torch.tensor(seq_norm, dtype=torch.float32)


def _load_mlp(device):
    path = "models/mlp_baseline_chkpt.pth"
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = MLPBaseline(
        input_dim=cfg["input_dim"], hidden_dim=cfg["hidden_dim"], num_targets=cfg["num_targets"]
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_crosscheck(data_path: str, num_ensemble: int, num_inference_steps: int, device: str):
    _seed_everything()
    os.makedirs(OUT_DIR, exist_ok=True)

    model_wrapper, _, retrieval_db = load_model_and_stats("models/diffusion_chkpt.pth")
    model_wrapper.to(device)
    model_wrapper.eval()
    config = model_wrapper.config
    st_gnn = model_wrapper.st_gnn
    forecaster = model_wrapper.forecaster
    mlp = _load_mlp(device)

    main_df, per_node_data, feature_cols, stats = _build_test_frames(data_path)
    c_mean = stats["c_mean"].numpy()
    c_std = stats["c_std"].numpy()
    t_mean = stats["t_mean"].numpy()
    t_std = stats["t_std"].numpy()
    t_mean_t = stats["t_mean"].to(device)
    t_std_t = stats["t_std"].to(device)
    seq_len = int(config["seq_len"])

    features_raw = main_df[feature_cols].values.astype(np.float32)
    features_norm = (features_raw - c_mean) / (c_std + 1e-5)
    timestamps = pd.to_datetime(main_df["date"]).reset_index(drop=True)
    targets_raw = main_df[VAR_NAMES].values.astype(np.float32)

    rain_cfg = config.get("rain_specialization", {})
    rain_enabled = bool(rain_cfg.get("enabled", False))
    wet_threshold = float(rain_cfg.get("wet_probability_threshold", 0.5))

    windows = _pick_week_windows(main_df)
    summary_rows = []
    metrics_payload: Dict[str, Dict] = {}

    for window in windows:
        eval_start = max(window.start_idx + seq_len, seq_len)
        eval_range = list(range(eval_start, window.end_idx))

        actuals = np.zeros((len(eval_range), 3), dtype=np.float32)
        model_preds = np.zeros((len(eval_range), 3), dtype=np.float32)
        persist_preds = np.zeros((len(eval_range), 3), dtype=np.float32)
        mlp_preds = np.full((len(eval_range), 3), np.nan, dtype=np.float32)
        wet_probs = np.full(len(eval_range), np.nan, dtype=np.float32)

        for j, idx in enumerate(eval_range):
            actuals[j] = targets_raw[idx]
            persist_preds[j] = targets_raw[idx - 1]

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
                sd = samples * t_std_t + t_mean_t
                sd[:, 0] = torch.clamp(torch.expm1(torch.clamp(sd[:, 0], max=20.0)), min=0.0, max=PRECIP_PHYSICAL_MAX_MM)
                sd[:, 1] = torch.clamp(sd[:, 1], min=0.0)  # wind speed >= 0
                sd[:, 2] = torch.clamp(sd[:, 2], min=0.0, max=100.0)
                model_preds[j] = torch.median(sd, dim=0).values.cpu().numpy()

                if rain_enabled:
                    wet_prob = forecaster.model.compute_wet_probability(
                        context=context_last, retrieved=retrieved, graph_emb=graph_emb
                    )
                    wet_probs[j] = float(wet_prob.squeeze().item())

                if mlp is not None:
                    x = features_norm[idx - seq_len : idx].flatten()
                    x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
                    out = mlp(x_t).cpu().numpy()[0] * t_std + t_mean
                    out[0] = np.clip(np.expm1(min(out[0], 20.0)), 0.0, PRECIP_PHYSICAL_MAX_MM)
                    out[1] = np.clip(out[1], 0.0, None)  # wind speed >= 0
                    out[2] = np.clip(out[2], 0.0, 100.0)
                    mlp_preds[j] = out

        # FIX #5: rain gate with anti-degenerate guard (applied per window, post-loop).
        gated = False
        if rain_enabled and np.isfinite(wet_probs).any():
            if float(np.nanmax(wet_probs)) < wet_threshold:
                print(
                    f"  [rain-gate guard] {window.name}: all wet_prob < {wet_threshold:.3f} "
                    f"(max={np.nanmax(wet_probs):.3f}); gate skipped to avoid all-dry collapse."
                )
            else:
                mask = wet_probs < wet_threshold
                model_preds[mask, 0] = 0.0
                gated = True

        week_df = pd.DataFrame(
            {
                "timestamp": [timestamps.iloc[i] for i in eval_range],
                "actual_precipitation": actuals[:, 0],
                "actual_wind_speed_10m": actuals[:, 1],
                "actual_relative_humidity_2m": actuals[:, 2],
                "pred_model_precipitation": model_preds[:, 0],
                "pred_model_wind_speed_10m": model_preds[:, 1],
                "pred_model_relative_humidity_2m": model_preds[:, 2],
                "pred_persistence_precipitation": persist_preds[:, 0],
                "pred_persistence_wind_speed_10m": persist_preds[:, 1],
                "pred_persistence_relative_humidity_2m": persist_preds[:, 2],
                "pred_mlp_precipitation": mlp_preds[:, 0],
                "pred_mlp_wind_speed_10m": mlp_preds[:, 1],
                "pred_mlp_relative_humidity_2m": mlp_preds[:, 2],
                "wet_probability": wet_probs,
            }
        )
        week_df.to_csv(os.path.join(OUT_DIR, f"{window.name}_actual_vs_pred.csv"), index=False)

        window_metrics = {"n_samples": int(len(eval_range)), "rain_gated": bool(gated)}
        row = {"window": window.name, "precip_sum_mm": float(window.precip_sum), "n_samples": int(len(eval_range))}
        for vi, var in enumerate(VAR_NAMES):
            act = actuals[:, vi]
            preds_by_model = {"model": model_preds[:, vi], "persistence": persist_preds[:, vi]}
            if mlp is not None:
                preds_by_model["mlp"] = mlp_preds[:, vi]
            var_metrics = {}
            for mname, pr in preds_by_model.items():
                rmse = float(np.sqrt(np.mean((pr - act) ** 2)))
                mae = float(np.mean(np.abs(pr - act)))
                corr = _corr(pr, act)
                var_metrics[mname] = {"rmse": rmse, "mae": mae, "corr": corr}
                row[f"{mname}_{var}_rmse"] = rmse
                row[f"{mname}_{var}_mae"] = mae
                row[f"{mname}_{var}_corr"] = corr
            if var == "precipitation":
                for thr in EVENT_THRESHOLDS:
                    key = str(thr).replace(".", "p")
                    for mname, pr in preds_by_model.items():
                        ev = _event_metrics(act, pr, threshold=thr)
                        var_metrics[mname][f"csi_thr_{key}"] = ev["csi"]
                        var_metrics[mname][f"pod_thr_{key}"] = ev["pod"]
                        var_metrics[mname][f"far_thr_{key}"] = ev["far"]
                        row[f"{mname}_csi_thr_{key}"] = ev["csi"]
            window_metrics[var] = var_metrics
        metrics_payload[window.name] = window_metrics
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "weekly_crosscheck_3var.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved weekly crosscheck summary: {summary_path}")

    meta = {
        "data_path": data_path,
        "checkpoint_path": "models/diffusion_chkpt.pth",
        "seed": SEED,
        "num_ensemble": int(num_ensemble),
        "num_inference_steps": int(num_inference_steps),
        "device": str(device),
        "variables": VAR_NAMES,
        "models_compared": ["model", "persistence"] + (["mlp"] if mlp is not None else []),
        "protocol": "one_step_hourly_with_actual_update",
        "rain_specialization_enabled": rain_enabled,
        "wet_probability_threshold": wet_threshold,
        "windows": [asdict(w) for w in windows],
        "metrics": metrics_payload,
    }
    with open(os.path.join(OUT_DIR, "weekly_crosscheck_3var_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"Saved metadata + metrics JSON: {os.path.join(OUT_DIR, 'weekly_crosscheck_3var_meta.json')}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CANONICAL_DATA_PATH)
    parser.add_argument("--num-ensemble", type=int, default=30)
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
