"""
Complete Training Pipeline with FULL Spatio-Temporal Graph Conditioning

This script implements the COMPLETE thesis pipeline:
1. Load ERA5 data with all features
2. Create Temporal Graph Sequences (sliding window)
3. Build Graph structure with SpatialGNN (GAT)
4. Apply TemporalAttention for sequence modeling
5. Condition Diffusion Model on graph embeddings
6. Save checkpoint with all configs

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

from src.data.ingest import fetch_era5_data, SITARO_NODES
from src.data.temporal_loader import TemporalGraphDataset, collate_temporal_graphs
from src.models.diffusion import ConditionalDiffusionModel, RainForecaster
from src.models.gnn import SpatioTemporalGNN
from src.retrieval.base import RetrievalDatabase


def train_pipeline():
    print("=" * 70)
    print("FULL SPATIO-TEMPORAL GRAPH CONDITIONED DIFFUSION MODEL TRAINING")
    print("=" * 70)
    
    # ===================================================================
    # STEP 1: Configuration
    # ===================================================================
    SEQ_LEN = 6         # 6 timesteps in each sequence
    BATCH_SIZE = 32     # Batch size
    EPOCHS = 10         # Training epochs
    HIDDEN_DIM = 128    # Hidden dimension
    GRAPH_DIM = 64      # Graph embedding dimension
    K_NEIGHBORS = 3     # FAISS neighbors
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Config]")
    print(f"   Device: {device}")
    print(f"   Sequence Length: {SEQ_LEN}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    # ===================================================================
    # STEP 2: Load Data
    # ===================================================================
    print("\n[1/7] Loading Data...")
    
    data_path = 'data/raw/sitaro_era5_2005_2025.parquet'
    try:
        df = pd.read_parquet(data_path)
        print(f"   Loaded from {data_path}")
        print(f"   Shape: {df.shape}")
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
        'cloudcover',
        'precipitation_lag1',
        'precipitation_lag3',
        'elevation',
    ]
    
    feature_cols = [c for c in all_feature_cols if c in df.columns]
    print(f"   Using {len(feature_cols)} features: {feature_cols}")
    
    # ===================================================================
    # STEP 3: Compute Normalization Stats
    # ===================================================================
    print("\n[2/7] Computing Normalization Stats...")
    
    # Target stats
    target_series = df['precipitation'].values
    target_log = np.log1p(target_series)
    t_mean = torch.tensor(target_log.mean(), dtype=torch.float32)
    t_std = torch.tensor(target_log.std(), dtype=torch.float32)
    
    # Context stats
    context_data = df[feature_cols].values
    c_mean = torch.tensor(context_data.mean(axis=0), dtype=torch.float32)
    c_std = torch.tensor(context_data.std(axis=0), dtype=torch.float32)
    
    stats = {'t_mean': t_mean, 't_std': t_std, 'c_mean': c_mean, 'c_std': c_std}
    print(f"   Target mean (log): {t_mean:.4f}, std: {t_std:.4f}")
    
    # ===================================================================
    # STEP 4: Create Temporal Graph Dataset
    # ===================================================================
    print("\n[3/7] Creating Temporal Graph Dataset...")
    
    dataset = TemporalGraphDataset(
        df=df,
        feature_cols=feature_cols,
        seq_len=SEQ_LEN,
        stats=stats
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_temporal_graphs
    )
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Batches per epoch: {len(dataloader)}")
    
    # ===================================================================
    # STEP 5: Build Retrieval Database
    # ===================================================================
    print("\n[4/7] Building Retrieval Database (FAISS)...")
    
    CONTEXT_DIM = len(feature_cols)
    
    # Normalize context for retrieval
    context_norm = (context_data - c_mean.numpy()) / (c_std.numpy() + 1e-5)
    
    retrieval_db = RetrievalDatabase(embedding_dim=CONTEXT_DIM)
    retrieval_db.add_items(context_norm, context_norm)
    print(f"   Indexed {len(context_norm)} historical states")
    
    # ===================================================================
    # STEP 6: Initialize Models
    # ===================================================================
    print("\n[5/7] Initializing Models...")
    
    NUM_NODES = len(SITARO_NODES)
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
    
    print(f"   SpatioTemporalGNN: {sum(p.numel() for p in st_gnn.parameters())} params")
    
    # Conditional Diffusion Model
    diff_model = ConditionalDiffusionModel(
        input_dim=1,
        context_dim=CONTEXT_DIM,
        retrieval_dim=RETRIEVAL_DIM,
        graph_dim=GRAPH_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    forecaster = RainForecaster(diff_model, device=device)
    print(f"   DiffusionModel: {sum(p.numel() for p in diff_model.parameters())} params")
    
    # Combined optimizer
    all_params = list(st_gnn.parameters()) + list(diff_model.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-3, weight_decay=1e-4)
    
    # ===================================================================
    # STEP 7: Training Loop
    # ===================================================================
    print("\n[6/7] Training...")
    print(f"   Epochs: {EPOCHS}")
    
    for epoch in range(EPOCHS):
        st_gnn.train()
        forecaster.model.train()
        
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batched_graphs, targets, contexts in progress_bar:
            # Move to device
            batched_graphs = [g.to(device) for g in batched_graphs]
            targets = targets.to(device)
            contexts = contexts.to(device)
            
            # 1. Get Spatio-Temporal Graph Embedding
            graph_emb = st_gnn(batched_graphs)  # [B, GRAPH_DIM]
            
            # 2. Get Retrieval (using context features)
            with torch.no_grad():
                context_np = contexts.cpu().numpy()
                retrieved = retrieval_db.query(context_np, k=K_NEIGHBORS)
                retrieved = retrieved.to(device)
            
            # 3. Diffusion training step (manual to include graph_emb)
            optimizer.zero_grad()
            
            noise = torch.randn_like(targets).to(device)
            timesteps = torch.randint(0, 1000, (targets.shape[0],), device=device).long()
            
            noisy_target = forecaster.scheduler.add_noise(targets, noise, timesteps)
            
            noise_pred = forecaster.model(
                noisy_target, 
                timesteps, 
                contexts, 
                retrieved,
                graph_emb  # FULL Spatio-Temporal Graph Conditioning!
            )
            
            loss = forecaster.criterion(noise_pred, noise)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"   Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")
    
    # ===================================================================
    # STEP 8: Save Checkpoint
    # ===================================================================
    print("\n[7/7] Saving Checkpoint...")
    
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
            'feature_cols': feature_cols
        }
    }
    
    torch.save(checkpoint, "models/diffusion_chkpt.pth")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("   Checkpoint: models/diffusion_chkpt.pth")
    print("   Components saved:")
    print("     - ConditionalDiffusionModel (with graph conditioning)")
    print("     - SpatioTemporalGNN (GAT + TemporalAttention)")
    print("=" * 70)


if __name__ == "__main__":
    train_pipeline()
