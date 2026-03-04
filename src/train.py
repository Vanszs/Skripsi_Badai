"""
Complete Training Pipeline with FULL Spatio-Temporal Graph Conditioning

CRITICAL: Implements STRICT TEMPORAL SPLIT to prevent data leakage!
- Training:   2005-2018 (14 years)
- Validation: 2019-2021 (3 years)
- Test:       2022-2025 (4 years)

This script implements the COMPLETE thesis pipeline:
1. Load ERA5 data with all features
2. TEMPORAL SPLIT (no random shuffle!)
3. Create Temporal Graph Sequences (sliding window)
4. Build Graph structure with SpatialGNN (GAT)
5. Apply TemporalAttention for sequence modeling
6. Condition Diffusion Model on graph embeddings
7. Save checkpoint with all configs

This fully satisfies the thesis title:
"Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.utils.data
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.ingest import fetch_era5_data, PANGRANGO_NODES
from src.data.temporal_loader import TemporalGraphDataset, collate_temporal_graphs
from src.models.diffusion import ConditionalDiffusionModel, RainForecaster
from src.models.gnn import SpatioTemporalGNN
from src.retrieval.base import RetrievalDatabase

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ==============================================================================
# TEMPORAL SPLIT FUNCTION - CRITICAL FOR PREVENTING DATA LEAKAGE
# ==============================================================================
def temporal_split(df, train_end='2018-12-31', val_end='2021-12-31'):
    """
    Split DataFrame berdasarkan waktu, BUKAN random.
    
    PENTING: Random shuffle pada time series menyebabkan data leakage!
    Model akan "melihat" masa depan saat training → evaluasi tidak valid.
    
    Args:
        df: DataFrame dengan kolom 'date'
        train_end: Tanggal terakhir training (inclusive)
        val_end: Tanggal terakhir validation (inclusive)
    
    Returns:
        train_df, val_df, test_df
    
    Split yang digunakan:
        Training:   2005-01-01 s/d 2018-12-31 (14 tahun, 67%)
        Validation: 2019-01-01 s/d 2021-12-31 (3 tahun, 14%)
        Test:       2022-01-01 s/d 2025-12-31 (4 tahun, 19%)
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # Remove timezone if present (to avoid tz-naive vs tz-aware comparison)
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    
    train_end_dt = pd.to_datetime(train_end)
    val_end_dt = pd.to_datetime(val_end)
    
    train_mask = df['date'] <= train_end_dt
    val_mask = (df['date'] > train_end_dt) & (df['date'] <= val_end_dt)
    test_mask = df['date'] > val_end_dt
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    return train_df, val_df, test_df


def compute_stats_from_training(train_df, feature_cols, target_col='precipitation'):
    """
    Compute normalization stats ONLY from training data.
    
    CRITICAL: Stats harus dari training set saja untuk mencegah data leakage!
    Validation dan test set dinormalisasi dengan stats yang sama dari training.
    
    Now supports MULTI-OUTPUT: 4 target variables.
    
    Args:
        train_df: DataFrame training SAJA
        feature_cols: List of feature column names
        target_col: Not used anymore, kept for compatibility
    
    Returns:
        dict with t_mean, t_std (both [4] tensors), c_mean, c_std
    """
    # MULTI-OUTPUT: 4 target variables
    TARGET_COLS = ['precipitation', 'wind_speed_10m', 'relative_humidity_2m']  # 3 vars, excluded temperature
    
    # Compute stats per target variable
    t_means = []
    t_stds = []
    
    for i, col in enumerate(TARGET_COLS):
        values = train_df[col].values
        
        if col == 'precipitation':
            # Log transform for precipitation only
            values_transformed = np.log1p(values)
            mean_val = values_transformed.mean()
            std_val = values_transformed.std()
            # Apply T_STD_MULTIPLIER only to precipitation
            T_STD_MULTIPLIER = 5.0
            std_val = std_val * T_STD_MULTIPLIER
        else:
            # No log transform for temp, wind, humidity
            mean_val = values.mean()
            std_val = values.std()
        
        t_means.append(mean_val)
        t_stds.append(std_val)
    
    t_mean = torch.tensor(t_means, dtype=torch.float32)  # [4]
    t_std = torch.tensor(t_stds, dtype=torch.float32)    # [4]
    
    # Feature stats (unchanged)
    feature_values = train_df[feature_cols].values
    c_mean = torch.tensor(feature_values.mean(axis=0), dtype=torch.float32)
    c_std = torch.tensor(feature_values.std(axis=0), dtype=torch.float32)
    
    print(f"[Multi-Output Stats] Target variables: {TARGET_COLS}")
    print(f"   t_mean: {t_mean.tolist()}")
    print(f"   t_std:  {t_std.tolist()}")
    
    return {
        't_mean': t_mean,
        't_std': t_std,
        'c_mean': c_mean,
        'c_std': c_std,
        'target_cols': TARGET_COLS  # Save for reference
    }


def train_pipeline():
    print("=" * 70)
    print("FULL SPATIO-TEMPORAL GRAPH CONDITIONED DIFFUSION MODEL TRAINING")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Using STRICT TEMPORAL SPLIT to prevent data leakage!")
    
    # ===================================================================
    # STEP 1: Configuration
    # ===================================================================
    SEQ_LEN = 6         # 6 timesteps in each sequence
    BATCH_SIZE = 512    # Maximized for RTX 3050 4GB
    EPOCHS = 20         # Training epochs
    HIDDEN_DIM = 128    # Hidden dimension
    GRAPH_DIM = 64      # Graph embedding dimension
    K_NEIGHBORS = 3     # FAISS neighbors
    
    # Temporal split boundaries
    TRAIN_END = '2018-12-31'
    VAL_END = '2021-12-31'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"\n[Config]")
    print(f"   Device: {device}")
    print(f"   Sequence Length: {SEQ_LEN}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Hidden Dim: {HIDDEN_DIM}")
    print(f"   Train Period: 2005-01-01 to {TRAIN_END}")
    print(f"   Val Period: {TRAIN_END[:-2]}01 to {VAL_END}")
    print(f"   Test Period: {VAL_END[:-2]}01 to 2025-12-31")
    
    # Check for AMP availability
    use_amp = torch.cuda.is_available()
    print(f"   Mixed Precision (AMP): {'Enabled' if use_amp else 'Disabled'}")
    
    # ===================================================================
    # STEP 2: Load Data
    # ===================================================================
    print("\n[1/8] Loading Data...")
    
    data_path = 'data/raw/pangrango_era5_2005_2025.parquet'
    try:
        df = pd.read_parquet(data_path)
        print(f"   Loaded from {data_path}")
        print(f"   Total Shape: {df.shape}")
    except FileNotFoundError:
        print("   Data not found. Running ingestion first...")
        df = fetch_era5_data()
    
    # Feature columns (use available ones)
    all_feature_cols = [
        'temperature_2m',
        'relative_humidity_2m', 
        'dewpoint_2m',
        'surface_pressure',
        'wind_speed_10m',
        'wind_direction_10m',
        'cloud_cover',
        'precipitation_lag1',
        'elevation',
    ]
    
    feature_cols = [c for c in all_feature_cols if c in df.columns]
    print(f"   Using {len(feature_cols)} features: {feature_cols}")
    
    # ===================================================================
    # STEP 3: TEMPORAL SPLIT - CRITICAL!
    # ===================================================================
    print("\n[2/8] Applying TEMPORAL SPLIT (No Random Shuffle!)...")
    
    train_df, val_df, test_df = temporal_split(df, TRAIN_END, VAL_END)
    
    print(f"   Training:   {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test:       {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Verify no overlap
    train_max = train_df['date'].max()
    val_min = val_df['date'].min()
    print(f"   ✓ Train ends: {train_max}")
    print(f"   ✓ Val starts: {val_min}")
    print(f"   ✓ Gap verified: {val_min > train_max}")
    
    # ===================================================================
    # STEP 4: Compute Normalization Stats FROM TRAINING ONLY
    # ===================================================================
    print("\n[3/8] Computing Normalization Stats (from TRAINING only)...")
    
    stats = compute_stats_from_training(train_df, feature_cols)
    print(f"   Target means: {stats['t_mean'].tolist()}")
    print(f"   Target stds:  {stats['t_std'].tolist()}")
    print(f"   ⚠️  These stats computed ONLY from training data!")
    
    # ===================================================================
    # STEP 5: Create Temporal Graph Datasets
    # ===================================================================
    print("\n[4/8] Creating Temporal Graph Datasets...")
    
    # Training dataset
    train_dataset = TemporalGraphDataset(
        df=train_df,
        feature_cols=feature_cols,
        seq_len=SEQ_LEN,
        stats=stats  # Stats from training
    )
    
    # Validation dataset (using training stats!)
    val_dataset = TemporalGraphDataset(
        df=val_df,
        feature_cols=feature_cols,
        seq_len=SEQ_LEN,
        stats=stats  # SAME stats from training
    )
    
    # On Windows, num_workers > 0 with custom collate + multiprocessing overhead is slow
    num_workers = 0
    print(f"   DataLoader Workers: {num_workers} (Windows optimized)")

    # DataLoaders with drop_last for consistent batch sizes
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_temporal_graphs,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_temporal_graphs,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Training batches/epoch: {len(train_loader)}")
    
    # ===================================================================
    # STEP 6: Build Retrieval Database & PRE-COMPUTE all queries
    # ===================================================================
    print("\n[5/8] Building Retrieval Database (from TRAINING only)...")
    
    CONTEXT_DIM = len(feature_cols)
    
    # Get training features and normalize with training stats
    train_features = train_df[feature_cols].values
    train_features_norm = (train_features - stats['c_mean'].numpy()) / (stats['c_std'].numpy() + 1e-5)
    train_features_norm = train_features_norm.astype(np.float32)
    
    # Build FAST IVF index for pre-computation (approximate but ~100x faster)
    import faiss
    nlist = 256  # number of Voronoi cells
    quantizer = faiss.IndexFlatL2(CONTEXT_DIM)
    fast_index = faiss.IndexIVFFlat(quantizer, CONTEXT_DIM, nlist, faiss.METRIC_L2)
    fast_index.train(train_features_norm)
    fast_index.add(train_features_norm)
    fast_index.nprobe = 16  # search 16 cells (speed/accuracy tradeoff)
    print(f"   Built IVF index: {fast_index.ntotal:,} items, {nlist} cells")
    
    # Store the data for retrieval lookup
    stored_data = train_features_norm.copy()
    
    # PRE-COMPUTE all FAISS queries (eliminates GPU<->CPU sync during training)
    print("   Pre-computing retrieval for all train/val samples...")
    
    def precompute_retrieval_fast(dataset, index, data, k, chunk_size=16384):
        """Extract contexts and batch-query fast IVF index."""
        valid_indices = dataset.valid_indices
        t_minus_1 = np.array(valid_indices) - 1
        if hasattr(dataset, 'features_norm'):
            all_contexts = dataset.features_norm[t_minus_1].mean(dim=1)
        else:
            all_contexts = dataset.features[t_minus_1].mean(dim=1)
        
        contexts_np = all_contexts.numpy().astype(np.float32)
        n = len(contexts_np)
        
        all_retrieved = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = contexts_np[start:end]
            _, indices = index.search(chunk, k)
            # Clamp invalid indices
            indices = np.clip(indices, 0, len(data) - 1)
            retrieved_chunk = data[indices]  # [chunk, k, dim]
            all_retrieved.append(torch.tensor(retrieved_chunk, dtype=torch.float32))
        
        retrieved = torch.cat(all_retrieved, dim=0)
        if retrieved.ndim == 3:
            retrieved = retrieved.view(retrieved.shape[0], -1)
        return retrieved
    
    train_retrieved = precompute_retrieval_fast(train_dataset, fast_index, stored_data, K_NEIGHBORS)
    train_dataset.set_precomputed_retrieval(train_retrieved)
    print(f"   Train retrieval: {train_retrieved.shape}")
    
    val_retrieved = precompute_retrieval_fast(val_dataset, fast_index, stored_data, K_NEIGHBORS)
    val_dataset.set_precomputed_retrieval(val_retrieved)
    print(f"   Val retrieval: {val_retrieved.shape}")
    print(f"   FAISS queries eliminated from training loop!")
    
    # Also keep a small retrieval_db for inference compatibility
    retrieval_db = RetrievalDatabase(embedding_dim=CONTEXT_DIM)
    retrieval_db.add_items(train_features_norm, train_features_norm)
    
    # ===================================================================
    # STEP 7: Initialize Models
    # ===================================================================
    print("\n[6/8] Initializing Models...")
    
    NUM_NODES = len(PANGRANGO_NODES)
    RETRIEVAL_DIM = CONTEXT_DIM * K_NEIGHBORS
    
    # Spatio-Temporal GNN (GAT + TemporalAttention)
    st_gnn = SpatioTemporalGNN(
        node_features=CONTEXT_DIM,
        hidden_dim=HIDDEN_DIM // 2,
        output_dim=GRAPH_DIM,
        num_gat_heads=4,
        num_attn_heads=4,
        seq_len=SEQ_LEN
    ).to(device)
    
    print(f"   SpatioTemporalGNN: {sum(p.numel() for p in st_gnn.parameters()):,} params")
    
    # Conditional Diffusion Model - MULTI-OUTPUT (4 targets)
    NUM_TARGETS = 3  # precipitation, wind, humidity (excluded temperature)
    diff_model = ConditionalDiffusionModel(
        input_dim=NUM_TARGETS,  # MULTI-OUTPUT
        context_dim=CONTEXT_DIM,
        retrieval_dim=RETRIEVAL_DIM,
        graph_dim=GRAPH_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    forecaster = RainForecaster(diff_model, device=device)
    print(f"   DiffusionModel: {sum(p.numel() for p in diff_model.parameters()):,} params")
    
    # Combined optimizer
    all_params = list(st_gnn.parameters()) + list(diff_model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    # ===================================================================
    # STEP 8: Training Loop with Validation
    # ===================================================================
    print("\n[7/8] Training with Validation...")
    print(f"   Epochs: {EPOCHS}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        st_gnn.train()
        forecaster.model.train()
        
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batched_graphs, targets, contexts, retrieved in progress_bar:
            # Move to device (non_blocking for async CPU->GPU)
            batched_graphs = [g.to(device, non_blocking=True) for g in batched_graphs]
            targets = targets.to(device, non_blocking=True)
            contexts = contexts.to(device, non_blocking=True)
            retrieved = retrieved.to(device, non_blocking=True)
            
            # AMP Context
            with torch.amp.autocast('cuda', enabled=use_amp):
                # 1. Get Spatio-Temporal Graph Embedding
                graph_emb = st_gnn(batched_graphs)
                
                # 2. Retrieved neighbors already pre-computed (no FAISS call!)
                
                # 3. Diffusion training step
                noise = torch.randn_like(targets)
                timesteps = torch.randint(0, 1000, (targets.shape[0],), device=device).long()
                
                noisy_target = forecaster.scheduler.add_noise(targets, noise, timesteps)
                
                noise_pred = forecaster.model(
                    noisy_target, 
                    timesteps, 
                    contexts, 
                    retrieved,
                    graph_emb
                )
                
                # Weighted MSE Loss Implementation
                error = (noise_pred - noise) ** 2
                
                # Weighting scheme for extreme events
                weights = torch.ones_like(error)
                weights[targets.abs() > 1.0] = 5.0
                weights[targets.abs() > 3.0] = 10.0
                
                loss = (error * weights).mean()
            
            # Scaler Step
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- Validation Phase ---
        st_gnn.eval()
        forecaster.model.eval()
        
        val_loss = 0
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
                
                noise_pred = forecaster.model(
                    noisy_target, timesteps, contexts, retrieved, graph_emb
                )
                
                val_loss += forecaster.criterion(noise_pred, noise).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"   Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   ✓ New best validation loss! Saving checkpoint...")
            
            os.makedirs("models", exist_ok=True)
            checkpoint = {
                'diffusion_state': forecaster.model.state_dict(),
                'st_gnn_state': st_gnn.state_dict(),
                'stats': stats,
                'config': {
                    'context_dim': CONTEXT_DIM,
                    'retrieval_dim': RETRIEVAL_DIM,
                    'graph_dim': GRAPH_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'k_neighbors': K_NEIGHBORS,
                    'seq_len': SEQ_LEN,
                    'num_nodes': NUM_NODES,
                    'feature_cols': feature_cols,
                    'train_end': TRAIN_END,
                    'val_end': VAL_END,
                    # MULTI-OUTPUT config
                    'num_targets': NUM_TARGETS,
                    'target_cols': ['precipitation', 'wind_speed_10m', 'relative_humidity_2m']
                }
            }
            torch.save(checkpoint, "models/diffusion_chkpt.pth")
    
    # ===================================================================
    # STEP 9: Save Loss Curves
    # ===================================================================
    os.makedirs("results/training_logs", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss')
    axes[0].set_title('Diffusion Model - Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    axes[1].set_title('Diffusion Model - Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/training_logs/training_loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save combined loss curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Train Loss')
    ax.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    ax.set_title('Diffusion Model - Training & Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("results/training_logs/validation_loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ===================================================================
    # STEP 10: Final Summary
    # ===================================================================
    print("\n[8/8] Training Complete!")
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE WITH PROPER TEMPORAL SPLIT!")
    print("=" * 70)
    print(f"   Best Validation Loss: {best_val_loss:.4f}")
    print(f"   Checkpoint: models/diffusion_chkpt.pth")
    print(f"\n   TEMPORAL SPLIT VERIFIED:")
    print(f"   ├── Training:   2005-2018 ({len(train_df):,} rows)")
    print(f"   ├── Validation: 2019-2021 ({len(val_df):,} rows)")
    print(f"   └── Test:       2022-2025 ({len(test_df):,} rows, NOT USED in training)")
    print(f"\n   ⚠️  Stats & Retrieval: computed from TRAINING only")
    print("=" * 70)


if __name__ == "__main__":
    train_pipeline()
