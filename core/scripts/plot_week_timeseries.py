"""
Plot 1-week hourly timeseries (actual vs model vs persistence vs MLP) for 3 targets.
Reads the reproducible weekly CSV produced by scripts/eval_rain_robust.py.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

OUT_DIR = "result_test/nowcasting_hourly_week"
VARS = [
    ("precipitation", "Precipitation (mm/h)"),
    ("wind_speed_10m", "Wind speed 10m (m/s)"),
    ("relative_humidity_2m", "Relative humidity 2m (%)"),
]


def plot_week(window: str):
    csv = os.path.join(OUT_DIR, f"{window}_actual_vs_pred.csv")
    df = pd.read_csv(csv, parse_dates=["timestamp"])
    t = df["timestamp"]
    fig, axes = plt.subplots(3, 1, figsize=(15, 11), sharex=True)
    for ax, (var, label) in zip(axes, VARS):
        ax.plot(t, df[f"actual_{var}"], color="#000000", lw=1.6, label="Actual")
        ax.plot(t, df[f"pred_model_{var}"], color="#d62728", lw=1.2, label="Full model")
        ax.plot(t, df[f"pred_persistence_{var}"], color="#1f77b4", lw=1.0, ls="--", alpha=0.8, label="Persistence")
        if f"pred_mlp_{var}" in df.columns and df[f"pred_mlp_{var}"].notna().any():
            ax.plot(t, df[f"pred_mlp_{var}"], color="#2ca02c", lw=1.0, ls=":", alpha=0.8, label="MLP")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_title(f"One-week hourly nowcasting — {window} (MAIN node, one-step + actual update)")
    axes[-1].set_xlabel("Time")
    fig.autofmt_xdate()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"timeseries_1week_{window}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}  ({len(df)} hours)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--window", default="median_week",
                   choices=["driest_week", "median_week", "wettest_week"])
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if a.all:
        for w in ["driest_week", "median_week", "wettest_week"]:
            plot_week(w)
    else:
        plot_week(a.window)
