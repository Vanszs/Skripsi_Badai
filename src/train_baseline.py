"""
MLP baseline training aligned with canonical main-node-only target policy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (  # noqa: E402
    CONTEXT_POLICY,
    FINAL_FEATURE_COLS,
    FINAL_TARGET_COLS,
    MAIN_NODE_NAME,
    NODE_COORDINATES,
    OPEN_METEO_MODEL,
    TARGET_NODE_POLICY,
    get_checkpoint_node_metadata,
    harmonize_weather_columns,
    validate_feature_schema,
    validate_feature_values,
)
from src.data.ingest import CANONICAL_DATA_PATH, fetch_era5_data  # noqa: E402
from src.models.mlp_baseline import MLPBaseline  # noqa: E402
from src.train import compute_stats_from_training, temporal_split  # noqa: E402

matplotlib.use("Agg")


class MLPDataset(torch.utils.data.Dataset):
    TARGET_COLS = FINAL_TARGET_COLS

    def __init__(self, df, feature_cols, seq_len=6, stats=None, main_node_name=MAIN_NODE_NAME):
        self.feature_cols = list(feature_cols)
        self.seq_len = seq_len
        self.stats = stats
        self.target_cols = self.TARGET_COLS
        self.main_node_name = main_node_name

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        main_df = df[df["node"] == self.main_node_name].copy().sort_values("date").reset_index(drop=True)
        if main_df.empty:
            raise ValueError(f"No rows found for main node '{self.main_node_name}'")

        self.features = main_df[self.feature_cols].values.astype(np.float32)
        self.targets_raw = main_df[self.target_cols].values.astype(np.float32)

        if stats is not None:
            c_mean = stats["c_mean"].numpy()[: len(self.feature_cols)]
            c_std = stats["c_std"].numpy()[: len(self.feature_cols)]
            self.features = (self.features - c_mean) / (c_std + 1e-5)

            t_mean = stats["t_mean"].numpy()
            t_std = stats["t_std"].numpy()
            self.targets = self.targets_raw.copy()
            self.targets[:, 0] = np.log1p(self.targets_raw[:, 0])
            self.targets = (self.targets - t_mean) / (t_std + 1e-5)
        else:
            self.targets = self.targets_raw.copy()

        self.n_samples = len(self.features) - seq_len

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len].flatten()
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_mlp_baseline(
    data_path: str = CANONICAL_DATA_PATH,
    seq_len: int = 6,
    batch_size: int = 256,
    epochs: int = 100,
    hidden_dim: int = 128,
    lr: float = 1e-3,
    patience: int = 10,
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
):
    print("=" * 70)
    print("MLP BASELINE (MAIN-NODE-ONLY)")
    print("=" * 70)
    print(f"Main node: {MAIN_NODE_NAME} @ {NODE_COORDINATES[MAIN_NODE_NAME]}")
    print(f"Target policy: {TARGET_NODE_POLICY} | Context policy: {CONTEXT_POLICY}")
    print(f"Model mode: {OPEN_METEO_MODEL}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Running ingestion...")
        fetch_era5_data(start_year=2005, end_year=2025)
    df = pd.read_parquet(data_path)
    df = harmonize_weather_columns(df)
    validate_feature_schema(df, FINAL_FEATURE_COLS, FINAL_TARGET_COLS)
    validate_feature_values(df, FINAL_FEATURE_COLS)

    feature_cols = list(FINAL_FEATURE_COLS)
    train_df, val_df, test_df = temporal_split(df, train_end, val_end)
    stats = compute_stats_from_training(train_df, feature_cols, main_node_name=MAIN_NODE_NAME)

    train_dataset = MLPDataset(train_df, feature_cols, seq_len, stats, MAIN_NODE_NAME)
    val_dataset = MLPDataset(val_df, feature_cols, seq_len, stats, MAIN_NODE_NAME)
    test_dataset = MLPDataset(test_df, feature_cols, seq_len, stats, MAIN_NODE_NAME)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = seq_len * len(feature_cols)
    num_targets = len(FINAL_TARGET_COLS)
    model = MLPBaseline(input_dim=input_dim, hidden_dim=hidden_dim, num_targets=num_targets).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train = epoch_loss / max(len(train_loader), 1)
        train_losses.append(avg_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()

        avg_val = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0
            metadata = get_checkpoint_node_metadata()
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "input_dim": input_dim,
                        "hidden_dim": hidden_dim,
                        "num_targets": num_targets,
                        "seq_len": seq_len,
                        "feature_cols": feature_cols,
                        "main_node_name": MAIN_NODE_NAME,
                        "main_node_identifier": NODE_COORDINATES[MAIN_NODE_NAME],
                        "target_node_policy": TARGET_NODE_POLICY,
                        "context_policy": CONTEXT_POLICY,
                        "open_meteo_model": OPEN_METEO_MODEL,
                        "data_path": data_path,
                        **metadata,
                    },
                    "stats": stats,
                },
                "models/mlp_baseline_chkpt.pth",
            )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        scheduler.step()

    checkpoint = torch.load("models/mlp_baseline_chkpt.pth", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())

    preds = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, num_targets))
    targets = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, num_targets))
    t_mean = stats["t_mean"].numpy()
    t_std = stats["t_std"].numpy()
    preds_denorm = preds * t_std + t_mean
    targets_denorm = targets * t_std + t_mean
    preds_denorm[:, 0] = np.clip(np.expm1(np.clip(preds_denorm[:, 0], a_min=None, a_max=20.0)), 0, None)
    targets_denorm[:, 0] = np.clip(np.expm1(np.clip(targets_denorm[:, 0], a_min=None, a_max=20.0)), 0, None)
    preds_denorm[:, 2] = np.clip(preds_denorm[:, 2], 0, 100)

    var_names = ["precipitation", "wind_speed", "humidity"]
    results = {}
    for i, var in enumerate(var_names):
        actual = targets_denorm[:, i]
        predicted = preds_denorm[:, i]
        rmse = float(np.sqrt(np.mean((predicted - actual) ** 2))) if len(actual) else float("nan")
        mae = float(np.mean(np.abs(predicted - actual))) if len(actual) else float("nan")
        corr = (
            float(np.corrcoef(predicted, actual)[0, 1])
            if len(actual) and np.std(actual) > 0 and np.std(predicted) > 0
            else float("nan")
        )
        results[var] = {"rmse": rmse, "mae": mae, "correlation": corr}

    os.makedirs("results/baseline_results", exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)
    with open("results/baseline_results/baseline_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {
                    "main_node_name": MAIN_NODE_NAME,
                    "main_node_identifier": NODE_COORDINATES[MAIN_NODE_NAME],
                    "target_node_policy": TARGET_NODE_POLICY,
                    "context_policy": CONTEXT_POLICY,
                    "graph_topology": "star",
                    "model_mode": OPEN_METEO_MODEL,
                },
                "metrics": results,
            },
            f,
            indent=2,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(train_losses, label="Train")
    axes[0].set_title("MLP Train Loss")
    axes[0].grid(True)
    axes[1].plot(val_losses, label="Val", color="orange")
    axes[1].set_title("MLP Val Loss")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("results/training_logs/baseline_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Baseline training complete.")
    print(results)
    return results


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CANONICAL_DATA_PATH)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--val-end", type=str, default="2021-12-31")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_mlp_baseline(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        patience=args.patience,
        train_end=args.train_end,
        val_end=args.val_end,
    )
