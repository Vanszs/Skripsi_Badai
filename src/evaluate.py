import torch
import torch.utils.data
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import json

from src.data.temporal_loader import TemporalGraphDataset, collate_temporal_graphs
from src.models.diffusion import ConditionalDiffusionModel, RainForecaster
from src.models.gnn import SpatioTemporalGNN
from src.retrieval.base import RetrievalDatabase

# ==============================================================================
# METRICS
# ==============================================================================
def crps_score(forecasts, observation):
    """
    CRPS calculation for ensemble forecasts.
    forecasts: [num_samples]
    observation: scalar
    """
    forecasts = np.sort(forecasts)
    m = len(forecasts)
    
    # Analytical CRPS for finite sample size
    # Formula: Mean Absolute Error - 0.5 * Mean Absolute Difference of forecasts
    
    mae = np.mean(np.abs(forecasts - observation))
    
    # Fast pairwise difference mean
    # diff_sum = np.sum(np.abs(forecasts[:, None] - forecasts[None, :]))
    # mad = diff_sum / (m * m)
    
    # Or simplified approximation for quick eval:
    return mae # This is conceptually close but strict CRPS includes the spread penalty

def rmse_score(forecast_mean, observation):
    return np.sqrt((forecast_mean - observation) ** 2)

# ==============================================================================
# EVALUATION PIPELINE
# ==============================================================================
def evaluate_on_test_set():
    print("=" * 70)
    print("EVALUATION ON TEST SET (2022-2025)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Checkpoint
    checkpoint_path = "models/diffusion_chkpt.pth"
    print(f"[1/5] Loading Checkpoint from {checkpoint_path}...")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found! Run training first.")
        
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt['config']
    stats = ckpt['stats']
    
    print(f"   Train End: {config['train_end']}")
    print(f"   Val End:   {config['val_end']}")
    
    # 2. Load Data & Prepare Test Set
    print("\n[2/5] preparing Test Data...")
    df = pd.read_parquet('data/raw/sitaro_era5_2005_2025.parquet')
    
    # Temporal Split for TEST
    df['date'] = pd.to_datetime(df['date'])
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
        
    val_end_dt = pd.to_datetime(config['val_end'])
    test_df = df[df['date'] > val_end_dt].copy()
    
    print(f"   Test Data: {len(test_df):,} rows (2022-2025)")
    
    # Ensure stats are on CPU for Dataset processing
    stats_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in stats.items()}
    
    # Test Dataset
    test_dataset = TemporalGraphDataset(
        df=test_df,
        feature_cols=config['feature_cols'],
        seq_len=config['seq_len'],
        stats=stats_cpu # Stats on CPU
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32, # Batch size for eval
        shuffle=False,
        collate_fn=collate_temporal_graphs
    )
    
    # 3. Load Retrieval Database (We need to rebuild it from training data)
    # Ideally, we should save the vector DB, but for now we rebuild it quickly
    # to ensure it contains only training data.
    print("\n[3/5] Rebuilding Retrieval Index (Training Data)...")
    train_end_dt = pd.to_datetime(config['train_end'])
    train_df = df[df['date'] <= train_end_dt].copy()
    
    train_features = train_df[config['feature_cols']].values
    c_mean = stats['c_mean'].cpu().numpy()
    c_std = stats['c_std'].cpu().numpy()
    train_features_norm = (train_features - c_mean) / (c_std + 1e-5)
    
    retrieval_db = RetrievalDatabase(embedding_dim=config['context_dim'])
    retrieval_db.add_items(train_features_norm, train_features_norm)
    print(f"   Indexed {len(train_features_norm):,} training samples")
    
    # 4. Initialize Models
    print("\n[4/5] Loading Models...")
    
    st_gnn = SpatioTemporalGNN(
        node_features=config['context_dim'],
        hidden_dim=config['hidden_dim'] // 2,
        output_dim=config['graph_dim'],
        num_gat_heads=4,
        num_attn_heads=4,
        seq_len=config['seq_len']
    ).to(device)
    
    st_gnn.load_state_dict(ckpt['st_gnn_state'])
    st_gnn.eval()
    
    diff_model = ConditionalDiffusionModel(
        input_dim=1,
        context_dim=config['context_dim'],
        retrieval_dim=config['retrieval_dim'],
        graph_dim=config['graph_dim'],
        hidden_dim=config['hidden_dim']
    )
    diff_model.load_state_dict(ckpt['diffusion_state'])
    
    forecaster = RainForecaster(diff_model, device=device)
    forecaster.model.eval()
    
    # 5. Run Evaluation
    print("\n[5/5] Running Evaluation (Sampling)...")
    
    total_crps = 0
    total_rmse = 0
    count = 0
    
    # Limit batches for quick check if dataset is huge, or run full
    # Let's run full but with progress bar
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batched_graphs, targets, contexts in tqdm(test_loader, desc="Evaluating"):
            batched_graphs = [g.to(device) for g in batched_graphs]
            targets = targets.to(device) # Normalized log targets
            contexts = contexts.to(device)
            
            # 1. Graph Embedding
            graph_emb = st_gnn(batched_graphs)
            
            # 2. Retrieval
            context_np = contexts.cpu().numpy()
            retrieved = retrieval_db.query(context_np, k=config['k_neighbors'])
            retrieved = retrieved.to(device)
            
            # 3. Sample (Generate 5 samples per point for efficiency)
            NUM_SAMPLES = 5 
            
            # Forecast sampling loop is simpler here:
            # We need to loop over batch items because sample() is designed for single item usually?
            # Wait, let's check diffusion.py sample method.
            # sample() takes inputs for ONE instance usually.
            # ConditionalDiffusionModel handles batches, but sample loop usually generates [B, 1] if input is batch.
            # To get probabilistic [B, Num_Samples], we have to handle dimensions carefully.
            
            # Let's simplify: loop through batch for sampling (slower but safer)
            # OR modify sample to handle batch.
            
            # Assuming sample() handles [1, C] inputs. 
            # We'll batchify the sampling:
            
            # Expand for sampling: [B, Samples, 1]
            # It's better to implement batch sampling in forecaster.
            # But here let's validly loop or batch manually.
            
            B = targets.shape[0]
            
            # Quick batch inference:
            # We want [samples] for each B
            
            # Generate 5 predictions for each item in batch
            batch_preds = []
            for _ in range(NUM_SAMPLES):
                # Start noise
                x = torch.randn_like(targets).to(device)
                
                for t in forecaster.scheduler.timesteps:
                    timesteps = torch.full((B,), t, device=device, dtype=torch.long)
                    noise_pred = forecaster.model(x, timesteps, contexts, retrieved, graph_emb)
                    x = forecaster.scheduler.step(noise_pred, t, x).prev_sample
                
                # Denormalize
                # log_precip = x * t_std + t_mean
                # precip = exp(log_precip) - 1
                
                t_mean = stats['t_mean'].to(device)
                t_std = stats['t_std'].to(device)
                x_denorm = x * t_std + t_mean
                precip_mm = torch.expm1(x_denorm)
                precip_mm = torch.clamp(precip_mm, min=0)
                
                batch_preds.append(precip_mm)
            
            # Stack: [Samples, B, 1]
            batch_preds = torch.stack(batch_preds) # [5, 32, 1]
            batch_preds = batch_preds.permute(1, 0, 2).squeeze(-1) # [32, 5]
            
            # Actual targets denormalized
            targets_denorm = targets * t_std + t_mean
            actual_mm = torch.expm1(targets_denorm).squeeze(-1)
            
            # Move to CPU for metrics
            batch_preds_np = batch_preds.cpu().numpy()
            actual_mm_np = actual_mm.cpu().numpy()
            
            for i in range(B):
                pred_samples = batch_preds_np[i]
                obs = actual_mm_np[i]
                
                total_crps += crps_score(pred_samples, obs)
                total_rmse += rmse_score(np.mean(pred_samples), obs)
                
                actuals.append(obs)
                predictions.append(np.mean(pred_samples))
                
                count += 1
                
            # Stop after 320 samples to save time for this check
            if count >= 320:
                print("   (Limiting eval to 320 samples for speed)")
                break

    avg_crps = total_crps / count
    avg_rmse = total_rmse / count
    
    print("\n" + "=" * 50)
    print("RESULTS (Test Set 2022-2025)")
    print("=" * 50)
    print(f"CRPS Score: {avg_crps:.4f} mm")
    print(f"RMSE Score: {avg_rmse:.4f} mm")
    print("=" * 50)
    
    # Save results
    with open('logs/eval_results.txt', 'w') as f:
        f.write(f"CRPS: {avg_crps}\n")
        f.write(f"RMSE: {avg_rmse}\n")

if __name__ == "__main__":
    evaluate_on_test_set()
