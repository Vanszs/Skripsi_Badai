"""
Training pipeline for canonical 5-node star spatio-temporal graph.
Target policy: main-node-only.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from typing import Dict, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import (  # noqa: E402
    FINAL_FEATURE_COLS,
    FINAL_TARGET_COLS,
    MAIN_NODE_NAME,
    NODE_COORDINATES,
    NODE_NAMES,
    OPEN_METEO_MODEL,
    STAR_EDGE_COUNT,
    TARGET_NODE_POLICY,
    CONTEXT_POLICY,
    build_star_edge_index,
    get_checkpoint_node_metadata,
    harmonize_weather_columns,
    validate_feature_schema,
    validate_feature_values,
)
from src.data.ingest import CANONICAL_DATA_PATH, fetch_era5_data  # noqa: E402
from src.data.temporal_loader import TemporalGraphDataset, collate_temporal_graphs  # noqa: E402
from src.models.diffusion import ConditionalDiffusionModel, RainForecaster  # noqa: E402
from src.models.gnn import SpatioTemporalGNN  # noqa: E402
from src.retrieval.base import RetrievalDatabase  # noqa: E402

matplotlib.use("Agg")


def temporal_split(df, train_end="2018-12-31", val_end="2021-12-31"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    train_df = df[df["date"] <= train_end_dt].copy()
    val_df = df[(df["date"] > train_end_dt) & (df["date"] <= val_end_dt)].copy()
    test_df = df[df["date"] > val_end_dt].copy()
    return train_df, val_df, test_df


def compute_stats_from_training(
    train_df: pd.DataFrame,
    feature_cols,
    target_col="precipitation",
    main_node_name: str = MAIN_NODE_NAME,
) -> Dict[str, torch.Tensor]:
    """
    Compute normalization stats from main node only, aligned to target policy.
    """
    _ = target_col  # compatibility
    target_cols = FINAL_TARGET_COLS
    main_df = train_df[train_df["node"] == main_node_name].copy()
    if main_df.empty:
        raise ValueError(f"No rows found for main node '{main_node_name}' in training set.")

    t_means = []
    t_stds = []
    for col in target_cols:
        values = main_df[col].values
        if col == "precipitation":
            values = np.log1p(values)
        t_means.append(float(values.mean()))
        t_stds.append(float(values.std()))

    feature_values = main_df[list(feature_cols)].values
    c_mean = feature_values.mean(axis=0)
    c_std = feature_values.std(axis=0)

    # FIX (KRITIS): some features are constant at the MAIN node (e.g. `elevation` is a
    # single static value per node), giving std=0. Dividing surrounding-node values by
    # (0 + 1e-5) produced ~1e8 GNN inputs and astronomical gradients. For any such
    # degenerate feature, fall back to ALL-training-node stats so the cross-node scale
    # is captured meaningfully (elevation varies across the 5 nodes by design).
    all_node_values = train_df[list(feature_cols)].values
    degenerate = c_std < 1e-6
    if degenerate.any():
        an_mean = all_node_values.mean(axis=0)
        an_std = all_node_values.std(axis=0)
        for i, is_deg in enumerate(degenerate):
            if is_deg:
                c_mean[i] = an_mean[i]
                c_std[i] = an_std[i] if an_std[i] > 1e-6 else 1.0

    return {
        "t_mean": torch.tensor(t_means, dtype=torch.float32),
        "t_std": torch.tensor(t_stds, dtype=torch.float32),
        "c_mean": torch.tensor(c_mean, dtype=torch.float32),
        "c_std": torch.tensor(c_std, dtype=torch.float32),
        "target_cols": target_cols,
        "stats_scope": "main_node_only_with_allnode_fallback_for_constant_features",
    }


def _build_precomputed_retrieval(
    dataset: TemporalGraphDataset,
    index,
    data: np.ndarray,
    k: int,
    strict_past: bool = False,
    exclude_self: bool = False,
    query_context_indices: np.ndarray | None = None,
    search_extra_factor: int = 8,
    chunk_size: int = 16384,
) -> torch.Tensor:
    contexts = dataset.get_all_contexts_for_valid_indices().numpy().astype(np.float32)
    n = len(contexts)
    if n == 0:
        return torch.zeros((0, k * data.shape[1]), dtype=torch.float32)

    total_refs = len(data)
    if total_refs == 0:
        raise ValueError("Retrieval data is empty. Cannot precompute retrieval contexts.")
    if strict_past and query_context_indices is None:
        raise ValueError("query_context_indices is required when strict_past=True")

    # Wide candidate pool is needed when filtering self/future neighbors.
    search_k = min(total_refs, max(k, k * max(1, int(search_extra_factor))))

    if not strict_past and not exclude_self and search_k == k:
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            query = contexts[start:end]
            _, indices = index.search(query, k)
            indices = np.clip(indices, 0, total_refs - 1)
            chunk = data[indices]
            chunks.append(torch.tensor(chunk, dtype=torch.float32))
        retrieved = torch.cat(chunks, dim=0)
        return retrieved.view(retrieved.shape[0], -1)

    rows = []
    data_dim = int(data.shape[1])
    zero_count = 0
    for i in range(n):
        query = contexts[i : i + 1]
        _, cand_idx = index.search(query, search_k)
        candidates = cand_idx[0].tolist()

        q_idx = int(query_context_indices[i]) if query_context_indices is not None else -1
        filtered = []
        for idx in candidates:
            if idx < 0 or idx >= total_refs:
                continue
            if exclude_self and idx == q_idx:
                continue
            if strict_past and idx >= q_idx:
                continue
            filtered.append(idx)
            if len(filtered) >= k:
                break

        if not filtered:
            rows.append(torch.zeros((k, data_dim), dtype=torch.float32))
            zero_count += 1
            continue

        if len(filtered) < k:
            filtered.extend([filtered[-1]] * (k - len(filtered)))
        chosen = np.array(filtered[:k], dtype=np.int64)
        rows.append(torch.tensor(data[chosen], dtype=torch.float32))

    if zero_count > 0:
        pct = 100.0 * zero_count / n
        warnings.warn(
            f"Retrieval fallback to zeros for {zero_count}/{n} samples ({pct:.2f}%). "
            "Consider increasing search_extra_factor or reducing strict_past/exclude_self constraints.",
            RuntimeWarning,
            stacklevel=2,
        )

    retrieved = torch.stack(rows, dim=0)
    return retrieved.view(retrieved.shape[0], -1)


def _tensor_corrcoef(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute Pearson correlation on flattened tensors.
    """
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    if x.numel() == 0 or y.numel() == 0:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x)) * torch.sqrt(torch.sum(y * y))
    denom_val = float(denom.item())
    if denom_val == 0.0:
        return float("nan")
    return float(torch.sum(x * y).item() / denom_val)


def _extract_precip_mm_from_normalized_targets(
    targets: torch.Tensor,
    precip_idx: int,
    t_mean_precip: torch.Tensor,
    t_std_precip: torch.Tensor,
) -> torch.Tensor:
    """
    Convert normalized precipitation target back to mm/h.
    """
    precip_log = targets[:, precip_idx] * t_std_precip + t_mean_precip
    precip_mm = torch.expm1(torch.clamp(precip_log, max=20.0))
    return torch.clamp(precip_mm, min=0.0)


def _binary_event_metrics(prob: np.ndarray, obs: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = prob >= threshold
    tp = float(np.logical_and(pred, obs == 1).sum())
    fp = float(np.logical_and(pred, obs == 0).sum())
    fn = float(np.logical_and(~pred, obs == 1).sum())
    denom = tp + fp + fn
    csi = tp / denom if denom > 0 else 0.0
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {"csi": csi, "pod": pod, "far": far}


def _calibrate_wet_probability_threshold(
    prob: np.ndarray,
    obs: np.ndarray,
    candidates: np.ndarray | None = None,
) -> Dict[str, float]:
    if candidates is None:
        candidates = np.linspace(0.05, 0.95, 19, dtype=np.float32)
    best = {"threshold": 0.5, "csi": -1.0, "pod": 0.0, "far": 1.0}
    for thr in candidates:
        m = _binary_event_metrics(prob, obs, float(thr))
        if m["csi"] > best["csi"]:
            best = {"threshold": float(thr), **m}
    return best


def train_pipeline(
    data_path: str = CANONICAL_DATA_PATH,
    seq_len: int = 6,
    batch_size: int = 512,
    epochs: int = 20,
    hidden_dim: int = 128,
    graph_dim: int = 64,
    k_neighbors: int = 3,
    amp: bool = False,
    train_end: str = "2018-12-31",
    val_end: str = "2021-12-31",
    start_year: int = 2005,
    end_year: int = 2025,
    grad_clip_norm: float = 1.0,
    lr_reduce_factor: float = 0.5,
    lr_patience: int = 3,
    lr_threshold: float = 1e-4,
    min_lr: float = 1e-6,
    early_stop_patience: int = 12,
    early_stop_min_delta: float = 0.0,
    rain_specialization: bool = True,
    rain_occurrence_threshold_mm: float = 0.1,
    wet_loss_weight: float = 0.7,
    cond_dropout: float = 0.15,
    seed: int = 1,
    lr: float = 1e-3,
    num_workers: int = 8,
) -> Tuple[float, Dict[str, object]]:
    print("=" * 80)
    print("TRAINING: 5-NODE STAR ST-GRAPH | MAIN-NODE-ONLY TARGET")
    print("=" * 80)
    print(f"Model mode: {OPEN_METEO_MODEL}")
    print(f"Node order: {NODE_NAMES}")
    print(f"Main node: {MAIN_NODE_NAME} @ {NODE_COORDINATES[MAIN_NODE_NAME]}")
    print(f"Target policy: {TARGET_NODE_POLICY}")
    print(f"Context policy: {CONTEXT_POLICY}")
    if grad_clip_norm and grad_clip_norm > 0:
        print(f"Grad clip norm: {grad_clip_norm}")
    else:
        print("Grad clip norm: disabled")
    print(
        "LR scheduler (ReduceLROnPlateau): "
        f"factor={lr_reduce_factor}, patience={lr_patience}, threshold={lr_threshold}, min_lr={min_lr}"
    )
    print(
        f"Early stopping: patience={early_stop_patience}, min_delta={early_stop_min_delta}"
    )
    print(
        "Rain specialization: "
        f"enabled={rain_specialization}, threshold_mm={rain_occurrence_threshold_mm}, "
        f"wet_loss_weight={wet_loss_weight}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Reproducibility: training was previously non-deterministic and could diverge
    # (val loss 0.61 on a lucky seed vs >10 on an unlucky one) due to lr * weighted-loss.
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    use_amp = bool(amp and torch.cuda.is_available())
    if torch.cuda.is_available():
        # benchmark=False so the fixed seed gives reproducible training.
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device} | seed: {seed}")

    print("\n[1/8] Loading canonical dataset...")
    if not os.path.exists(data_path):
        print(f"Canonical dataset not found at {data_path}. Running ingestion...")
        fetch_era5_data(start_year=start_year, end_year=end_year)
    df = pd.read_parquet(data_path)
    df = harmonize_weather_columns(df)
    validate_feature_schema(df, FINAL_FEATURE_COLS, FINAL_TARGET_COLS)
    validate_feature_values(df, FINAL_FEATURE_COLS)
    feature_cols = list(FINAL_FEATURE_COLS)
    print(f"Loaded: {df.shape}")

    print("\n[2/8] Temporal split...")
    train_df, val_df, test_df = temporal_split(df, train_end, val_end)
    print(f"Train rows: {len(train_df):,}")
    print(f"Val rows:   {len(val_df):,}")
    print(f"Test rows:  {len(test_df):,}")

    print("\n[3/8] Compute normalization stats (main-node-only)...")
    stats = compute_stats_from_training(train_df, feature_cols, main_node_name=MAIN_NODE_NAME)
    print(f"t_mean: {stats['t_mean'].tolist()}")
    print(f"t_std:  {stats['t_std'].tolist()}")

    print("\n[4/8] Build temporal datasets...")
    edge_index = build_star_edge_index(NODE_NAMES)
    if edge_index.shape[1] != STAR_EDGE_COUNT:
        raise ValueError(
            f"Training graph edge mismatch. Expected {STAR_EDGE_COUNT}, got {edge_index.shape[1]}"
        )
    train_dataset = TemporalGraphDataset(
        df=train_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        node_names=NODE_NAMES,
        main_node_name=MAIN_NODE_NAME,
        edge_index=edge_index,
        stats=stats,
    )
    val_dataset = TemporalGraphDataset(
        df=val_df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        node_names=NODE_NAMES,
        main_node_name=MAIN_NODE_NAME,
        edge_index=edge_index,
        stats=stats,
    )

    _nw = int(num_workers)
    _loader_kwargs = dict(collate_fn=collate_temporal_graphs, num_workers=_nw, pin_memory=True)
    if _nw > 0:
        _loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs,
    )
    print(f"Train samples: {len(train_dataset)} | batches: {len(train_loader)}")
    print(f"Val samples:   {len(val_dataset)} | batches: {len(val_loader)}")

    print("\n[5/8] Build retrieval context (main-node-only)...")
    main_train = train_df[train_df["node"] == MAIN_NODE_NAME].copy().sort_values("date")
    precip_idx = list(FINAL_TARGET_COLS).index("precipitation")
    t_mean_precip = stats["t_mean"][precip_idx].to(device)
    t_std_precip = stats["t_std"][precip_idx].to(device)
    rain_occurrence_threshold_mm = max(0.0, float(rain_occurrence_threshold_mm))

    wet_series = (main_train["precipitation"].astype(float).values >= rain_occurrence_threshold_mm).astype(np.float32)
    wet_total = float(len(wet_series))
    wet_pos = float(wet_series.sum())
    wet_neg = wet_total - wet_pos
    wet_pos_weight = 1.0
    effective_rain_specialization = bool(rain_specialization and wet_loss_weight > 0.0)
    if effective_rain_specialization:
        if wet_pos <= 0.0 or wet_neg <= 0.0:
            print(
                "Rain specialization disabled automatically: "
                "train split does not contain both wet and dry samples."
            )
            effective_rain_specialization = False
        else:
            wet_pos_weight = float(np.clip(wet_neg / max(wet_pos, 1.0), 1.0, 100.0))
    wet_criterion = (
        torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([wet_pos_weight], dtype=torch.float32, device=device)
        )
        if effective_rain_specialization
        else None
    )
    print(
        "Rain occurrence train stats: "
        f"wet={wet_pos:.0f}/{wet_total:.0f} ({(wet_pos / max(wet_total, 1.0)):.3f}), "
        f"pos_weight={wet_pos_weight:.3f}, enabled={effective_rain_specialization}"
    )

    train_features = main_train[feature_cols].values
    train_features_norm = (
        (train_features - stats["c_mean"].numpy()) / (stats["c_std"].numpy() + 1e-5)
    ).astype(np.float32)

    # Retrieval values are the NEXT-STEP outcome (target at tau+1), normalized like training targets.
    train_targets_raw = main_train[FINAL_TARGET_COLS].values.astype(np.float32)
    train_targets_g = train_targets_raw.copy()
    train_targets_g[:, precip_idx] = np.log1p(train_targets_g[:, precip_idx])
    train_targets_norm = (
        (train_targets_g - stats["t_mean"].numpy()) / (stats["t_std"].numpy() + 1e-5)
    ).astype(np.float32)

    context_dim = len(feature_cols)
    # DB key = feature at tau (drop last), value = outcome target at tau+1 (drop first).
    retrieval_keys = train_features_norm[:-1]
    retrieval_values = train_targets_norm[1:]
    retrieval_db = RetrievalDatabase(embedding_dim=context_dim)
    retrieval_db.add_items(retrieval_keys, retrieval_values)
    retrieval_index = retrieval_db.index

    # Train retrieval is restricted to strict-past neighbors to avoid temporal look-ahead.
    # Key position j == time tau; strict_past keeps j < t-1 so value time tau+1 < t.
    train_context_indices = np.array(train_dataset.valid_indices, dtype=np.int64) - 1

    train_retrieved = _build_precomputed_retrieval(
        train_dataset,
        retrieval_index,
        retrieval_values,
        k_neighbors,
        strict_past=True,
        exclude_self=True,
        query_context_indices=train_context_indices,
    )
    val_retrieved = _build_precomputed_retrieval(
        val_dataset,
        retrieval_index,
        retrieval_values,
        k_neighbors,
        strict_past=False,
        exclude_self=False,
        search_extra_factor=1,
    )
    train_dataset.set_precomputed_retrieval(train_retrieved)
    val_dataset.set_precomputed_retrieval(val_retrieved)
    print(f"Retrieval DB vectors: {len(retrieval_values):,} (key=feat[t], value=target[t+1])")

    print("\n[6/8] Initialize models...")
    num_targets = len(FINAL_TARGET_COLS)
    retrieval_dim = num_targets * k_neighbors

    st_gnn = SpatioTemporalGNN(
        node_features=context_dim,
        hidden_dim=hidden_dim // 2,
        output_dim=graph_dim,
        num_gat_heads=4,
        num_attn_heads=4,
        seq_len=seq_len,
    ).to(device)

    diff_model = ConditionalDiffusionModel(
        input_dim=num_targets,
        context_dim=context_dim,
        retrieval_dim=retrieval_dim,
        graph_dim=graph_dim,
        hidden_dim=hidden_dim,
    )
    forecaster = RainForecaster(diff_model, device=device)

    trainable_params = list(st_gnn.parameters()) + list(diff_model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_patience,
        threshold=lr_threshold,
        min_lr=min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print("\n[7/8] Train/validate...")
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_corrs = []
    val_corrs = []
    grad_norms = []
    lrs = []
    best_epoch = 0
    no_improve_epochs = 0
    best_wet_threshold = 0.5
    best_wet_metrics = {"csi": 0.0, "pod": 0.0, "far": 1.0}
    non_finite_batches_total = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        st_gnn.train()
        forecaster.model.train()
        running = 0.0
        running_noise = 0.0
        running_wet = 0.0
        running_wet_count = 0
        running_corr = 0.0
        running_corr_count = 0
        effective_train_batches = 0
        non_finite_batches_epoch = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batched_graphs, targets, contexts, retrieved in pbar:
            batched_graphs = [g.to(device, non_blocking=True) for g in batched_graphs]
            targets = targets.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            retrieved = retrieved.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                graph_emb = st_gnn(batched_graphs)
                # FIX #16: conditioning dropout so the model is robust to ablation
                # (graph_emb / retrieved set to zero), making the ablation table meaningful.
                if cond_dropout > 0.0:
                    bsz = targets.shape[0]
                    g_keep = (torch.rand(bsz, 1, device=device) >= cond_dropout).float()
                    r_keep = (torch.rand(bsz, 1, device=device) >= cond_dropout).float()
                    graph_emb = graph_emb * g_keep
                    retrieved = retrieved * r_keep
                noise = torch.randn_like(targets)
                timesteps = torch.randint(0, 1000, (targets.shape[0],), device=device).long()
                noisy_target = forecaster.scheduler.add_noise(targets, noise, timesteps)
                if effective_rain_specialization:
                    noise_pred, wet_logit = forecaster.model(
                        noisy_target,
                        timesteps,
                        contexts,
                        retrieved,
                        graph_emb,
                        return_wet_logit=True,
                    )
                else:
                    noise_pred = forecaster.model(noisy_target, timesteps, contexts, retrieved, graph_emb)
                    wet_logit = None

            # Keep loss computation in FP32 for numerical stability under AMP.
            noise_loss = RainForecaster.weighted_noise_loss(
                noise_pred.float(), noise.float(), targets.float()
            )
            if effective_rain_specialization:
                precip_mm = _extract_precip_mm_from_normalized_targets(
                    targets.float(), precip_idx, t_mean_precip, t_std_precip
                )
                wet_labels = (precip_mm >= rain_occurrence_threshold_mm).float().unsqueeze(1)
                wet_loss = wet_criterion(wet_logit.float(), wet_labels)
                loss = noise_loss + (float(wet_loss_weight) * wet_loss)
            else:
                wet_loss = None
                loss = noise_loss

            if not torch.isfinite(loss):
                non_finite_batches_epoch += 1
                non_finite_batches_total += 1
                optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="nan", corr="nan")
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if grad_clip_norm and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
                grad_norms.append(float(grad_norm.detach().item()))
            scaler.step(optimizer)
            scaler.update()
            effective_train_batches += 1
            running += loss.item()
            running_noise += float(noise_loss.item())
            if wet_loss is not None:
                running_wet += float(wet_loss.item())
                running_wet_count += 1
            batch_corr = _tensor_corrcoef(noise_pred, noise)
            if np.isfinite(batch_corr):
                running_corr += batch_corr
                running_corr_count += 1
                corr_txt = f"{batch_corr:.4f}"
            else:
                corr_txt = "nan"
            wet_txt = f"{wet_loss.item():.4f}" if wet_loss is not None else "na"
            pbar.set_postfix(loss=f"{loss.item():.4f}", noise=f"{noise_loss.item():.4f}", wet=wet_txt, corr=corr_txt)

        avg_train = running / max(effective_train_batches, 1)
        avg_train_noise = running_noise / max(effective_train_batches, 1)
        avg_train_wet = (
            running_wet / running_wet_count if running_wet_count > 0 else float("nan")
        )
        train_losses.append(avg_train)
        avg_train_corr = (
            running_corr / running_corr_count if running_corr_count > 0 else float("nan")
        )
        train_corrs.append(avg_train_corr)

        st_gnn.eval()
        forecaster.model.eval()
        val_running = 0.0
        val_noise_running = 0.0
        val_wet_running = 0.0
        val_wet_count = 0
        val_corr_running = 0.0
        val_corr_count = 0
        val_effective_batches = 0
        val_wet_probs = []
        val_wet_obs = []
        with torch.no_grad():
            for batched_graphs, targets, contexts, retrieved in val_loader:
                batched_graphs = [g.to(device, non_blocking=True) for g in batched_graphs]
                targets = targets.to(device, non_blocking=True)
                contexts = contexts.to(device, non_blocking=True)
                retrieved = retrieved.to(device, non_blocking=True)
                graph_emb = st_gnn(batched_graphs)
                noise = torch.randn_like(targets)
                timesteps = torch.randint(0, 1000, (targets.shape[0],), device=device).long()
                noisy_target = forecaster.scheduler.add_noise(targets, noise, timesteps)
                if effective_rain_specialization:
                    noise_pred, wet_logit = forecaster.model(
                        noisy_target,
                        timesteps,
                        contexts,
                        retrieved,
                        graph_emb,
                        return_wet_logit=True,
                    )
                else:
                    noise_pred = forecaster.model(noisy_target, timesteps, contexts, retrieved, graph_emb)
                    wet_logit = None

                val_noise_loss = RainForecaster.weighted_noise_loss(
                    noise_pred.float(), noise.float(), targets.float()
                )
                if effective_rain_specialization:
                    precip_mm = _extract_precip_mm_from_normalized_targets(
                        targets.float(), precip_idx, t_mean_precip, t_std_precip
                    )
                    wet_labels = (precip_mm >= rain_occurrence_threshold_mm).float().unsqueeze(1)
                    val_wet_loss = wet_criterion(wet_logit.float(), wet_labels)
                    val_loss = val_noise_loss + (float(wet_loss_weight) * val_wet_loss)
                    val_wet_probs.append(torch.sigmoid(wet_logit.float()).squeeze(1).cpu().numpy())
                    val_wet_obs.append(wet_labels.squeeze(1).cpu().numpy())
                    val_wet_running += float(val_wet_loss.item())
                    val_wet_count += 1
                else:
                    val_wet_loss = None
                    val_loss = val_noise_loss

                if not torch.isfinite(val_loss):
                    continue

                val_effective_batches += 1
                val_running += val_loss.item()
                val_noise_running += float(val_noise_loss.item())
                batch_corr = _tensor_corrcoef(noise_pred, noise)
                if np.isfinite(batch_corr):
                    val_corr_running += batch_corr
                    val_corr_count += 1

        avg_val = val_running / max(val_effective_batches, 1)
        avg_val_noise = val_noise_running / max(val_effective_batches, 1)
        avg_val_wet = val_wet_running / val_wet_count if val_wet_count > 0 else float("nan")
        if effective_rain_specialization and val_wet_probs:
            wet_prob_np = np.concatenate(val_wet_probs).astype(np.float32)
            wet_obs_np = np.concatenate(val_wet_obs).astype(np.int32)
            wet_calib = _calibrate_wet_probability_threshold(wet_prob_np, wet_obs_np)
            val_wet_threshold = float(wet_calib["threshold"])
            val_wet_csi = float(wet_calib["csi"])
            val_wet_pod = float(wet_calib["pod"])
            val_wet_far = float(wet_calib["far"])
        else:
            val_wet_threshold = 0.5
            val_wet_csi = float("nan")
            val_wet_pod = float("nan")
            val_wet_far = float("nan")

        val_losses.append(avg_val)
        avg_val_corr = (
            val_corr_running / val_corr_count if val_corr_count > 0 else float("nan")
        )
        val_corrs.append(avg_val_corr)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train: {avg_train:.4f} (noise={avg_train_noise:.4f}, wet={avg_train_wet:.4f}, corr={avg_train_corr:.4f}) | "
            f"Val: {avg_val:.4f} (noise={avg_val_noise:.4f}, wet={avg_val_wet:.4f}, corr={avg_val_corr:.4f})"
        )
        if effective_rain_specialization:
            print(
                "  Rain occurrence (val): "
                f"thr={val_wet_threshold:.2f}, CSI={val_wet_csi:.4f}, POD={val_wet_pod:.4f}, FAR={val_wet_far:.4f}"
            )
        if non_finite_batches_epoch > 0:
            print(f"  Warning: skipped non-finite train batches this epoch = {non_finite_batches_epoch}")
        scheduler.step(avg_val)
        current_lr = float(optimizer.param_groups[0]["lr"])
        lrs.append(current_lr)
        print(f"  LR: {current_lr:.6g}")

        improved = avg_val < (best_val_loss - max(0.0, float(early_stop_min_delta)))
        if improved:
            best_val_loss = avg_val
            best_epoch = epoch + 1
            no_improve_epochs = 0
            metadata = get_checkpoint_node_metadata()
            checkpoint = {
                "diffusion_state": forecaster.model.state_dict(),
                "st_gnn_state": st_gnn.state_dict(),
                "stats": stats,
                "config": {
                    "context_dim": context_dim,
                    "retrieval_dim": retrieval_dim,
                    "graph_dim": graph_dim,
                    "hidden_dim": hidden_dim,
                    "k_neighbors": k_neighbors,
                    "amp_enabled": use_amp,
                    "seq_len": seq_len,
                    "feature_cols": feature_cols,
                    "target_cols": FINAL_TARGET_COLS,
                    "num_targets": num_targets,
                    "cond_dropout": float(cond_dropout),
                    "seed": int(seed),
                    "loss_type": (
                        "weighted_noise_mse_plus_wet_bce"
                        if effective_rain_specialization
                        else "weighted_noise_mse"
                    ),
                    "train_noise_corr": None if not np.isfinite(avg_train_corr) else float(avg_train_corr),
                    "val_noise_corr": None if not np.isfinite(avg_val_corr) else float(avg_val_corr),
                    "train_noise_loss": float(avg_train_noise),
                    "val_noise_loss": float(avg_val_noise),
                    "train_wet_loss": None if not np.isfinite(avg_train_wet) else float(avg_train_wet),
                    "val_wet_loss": None if not np.isfinite(avg_val_wet) else float(avg_val_wet),
                    "grad_clip_norm": float(grad_clip_norm),
                    "lr_scheduler": "ReduceLROnPlateau",
                    "lr_reduce_factor": float(lr_reduce_factor),
                    "lr_patience": int(lr_patience),
                    "lr_threshold": float(lr_threshold),
                    "min_lr": float(min_lr),
                    "current_lr": current_lr,
                    "best_epoch": int(best_epoch),
                    "early_stop_patience": int(early_stop_patience),
                    "early_stop_min_delta": float(early_stop_min_delta),
                    "train_end": train_end,
                    "val_end": val_end,
                    "data_path": data_path,
                    "open_meteo_model": OPEN_METEO_MODEL,
                    "main_node_identifier": NODE_COORDINATES[MAIN_NODE_NAME],
                    "rain_specialization": {
                        "enabled": bool(effective_rain_specialization),
                        "rain_occurrence_threshold_mm": float(rain_occurrence_threshold_mm),
                        "wet_loss_weight": float(wet_loss_weight),
                        "wet_pos_weight": float(wet_pos_weight),
                        "wet_probability_threshold": float(val_wet_threshold),
                        "calibration_metric": "csi",
                        "calibration_split": "validation",
                    },
                    **metadata,
                },
            }
            torch.save(checkpoint, "models/diffusion_chkpt.pth")
            best_wet_threshold = float(val_wet_threshold)
            best_wet_metrics = {
                "csi": None if not np.isfinite(val_wet_csi) else float(val_wet_csi),
                "pod": None if not np.isfinite(val_wet_pod) else float(val_wet_pod),
                "far": None if not np.isfinite(val_wet_far) else float(val_wet_far),
            }
        else:
            no_improve_epochs += 1
            if early_stop_patience > 0 and no_improve_epochs >= early_stop_patience:
                print(
                    f"Early stopping triggered at epoch {epoch+1} "
                    f"(best epoch: {best_epoch}, best val: {best_val_loss:.4f})."
                )
                break

    if non_finite_batches_total > 0:
        print(f"Total non-finite train batches skipped: {non_finite_batches_total}")

    print("\n[8/8] Save training curves...")
    os.makedirs("results/training_logs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    axes[0].set_title("Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[1].plot(range(1, len(val_losses) + 1), val_losses, label="Val", color="r")
    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig("results/training_logs/training_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "best_val_loss": float(best_val_loss),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "node_names": NODE_NAMES,
        "main_node": MAIN_NODE_NAME,
        "target_policy": TARGET_NODE_POLICY,
        "context_policy": CONTEXT_POLICY,
        "graph_topology": "star",
        "open_meteo_model": OPEN_METEO_MODEL,
        "best_epoch": int(best_epoch),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "lr_scheduler": "ReduceLROnPlateau",
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "grad_clip_norm": float(grad_clip_norm),
        "mean_grad_norm": None if not grad_norms else float(np.mean(grad_norms)),
        "max_grad_norm": None if not grad_norms else float(np.max(grad_norms)),
        "last_train_noise_corr": None if not train_corrs else (None if not np.isfinite(train_corrs[-1]) else float(train_corrs[-1])),
        "last_val_noise_corr": None if not val_corrs else (None if not np.isfinite(val_corrs[-1]) else float(val_corrs[-1])),
        "rain_specialization_enabled": bool(effective_rain_specialization),
        "rain_occurrence_threshold_mm": float(rain_occurrence_threshold_mm),
        "wet_loss_weight": float(wet_loss_weight),
        "wet_pos_weight": float(wet_pos_weight),
        "wet_probability_threshold": float(best_wet_threshold),
        "wet_val_metrics_at_best": best_wet_metrics,
        "non_finite_batches_skipped": int(non_finite_batches_total),
    }
    print("Training complete.")
    print(summary)
    return best_val_loss, summary


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default=CANONICAL_DATA_PATH)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument("--k-neighbors", type=int, default=3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--val-end", type=str, default="2021-12-31")
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--lr-reduce-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=3)
    parser.add_argument("--lr-threshold", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--disable-rain-specialization", action="store_true")
    parser.add_argument("--rain-occurrence-threshold-mm", type=float, default=0.1)
    parser.add_argument("--wet-loss-weight", type=float, default=0.7)
    parser.add_argument("--cond-dropout", type=float, default=0.15,
                        help="Per-sample dropout prob for graph/retrieval conditioning (robust ablation).")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed for reproducible training (1 is empirically stable).")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers (parallel graph batching; 0=main process).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_pipeline(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        graph_dim=args.graph_dim,
        k_neighbors=args.k_neighbors,
        amp=args.amp,
        train_end=args.train_end,
        val_end=args.val_end,
        start_year=args.start_year,
        end_year=args.end_year,
        grad_clip_norm=args.grad_clip_norm,
        lr_reduce_factor=args.lr_reduce_factor,
        lr_patience=args.lr_patience,
        lr_threshold=args.lr_threshold,
        min_lr=args.min_lr,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        rain_specialization=not args.disable_rain_specialization,
        rain_occurrence_threshold_mm=args.rain_occurrence_threshold_mm,
        wet_loss_weight=args.wet_loss_weight,
        cond_dropout=args.cond_dropout,
        seed=args.seed,
        lr=args.lr,
        num_workers=args.num_workers,
    )
