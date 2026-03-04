"""
Training Pipeline untuk MLP Baseline Model

Pipeline:
1. Load dataset ERA5 Gunung Gede-Pangrango
2. Split dataset (temporal split: Train 2005-2018, Val 2019-2021, Test 2022-2025)
3. Train MLP baseline
4. Evaluate RMSE / MAE / Correlation
5. Simpan hasil ke results/baseline_results/

Tujuan: Membandingkan model sederhana (MLP) dengan model utama
(Diffusion + Retrieval + GNN).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import torch.nn as nn
import torch.utils.data
import pandas as pd
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.mlp_baseline import MLPBaseline
from src.train import temporal_split, compute_stats_from_training


class MLPDataset(torch.utils.data.Dataset):
    """
    Dataset untuk MLP baseline.
    Mengambil sliding window features dan target multi-output.
    """
    
    TARGET_COLS = ['precipitation', 'wind_speed_10m', 'relative_humidity_2m']
    
    def __init__(self, df, feature_cols, seq_len=6, stats=None):
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.stats = stats
        self.target_cols = self.TARGET_COLS
        
        # Pivot per-node: rata-rata semua node per timestep
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Average across nodes per timestamp
        grouped = df.groupby('date').agg({
            **{col: 'mean' for col in feature_cols if col in df.columns},
            **{col: 'mean' for col in self.TARGET_COLS if col in df.columns}
        }).sort_index().reset_index()
        
        available_features = [c for c in feature_cols if c in grouped.columns]
        self.features = grouped[available_features].values.astype(np.float32)
        
        # Multi-output targets
        targets_raw = []
        for col in self.TARGET_COLS:
            if col in grouped.columns:
                vals = grouped[col].values.astype(np.float32)
            else:
                vals = np.zeros(len(grouped), dtype=np.float32)
            targets_raw.append(vals)
        self.targets_raw = np.stack(targets_raw, axis=1)  # [T, num_targets]
        
        # Normalize
        if stats is not None:
            c_mean = stats['c_mean'].numpy()[:len(available_features)]
            c_std = stats['c_std'].numpy()[:len(available_features)]
            self.features = (self.features - c_mean) / (c_std + 1e-5)
            
            t_mean = stats['t_mean'].numpy()
            t_std = stats['t_std'].numpy()
            
            # Transform targets: log1p untuk precipitation
            self.targets = self.targets_raw.copy()
            self.targets[:, 0] = np.log1p(self.targets_raw[:, 0])
            
            # Normalize
            self.targets = (self.targets - t_mean) / (t_std + 1e-5)
        else:
            self.targets = self.targets_raw.copy()
        
        self.n_samples = len(self.features) - seq_len
    
    def __len__(self):
        return max(0, self.n_samples)
    
    def __getitem__(self, idx):
        # Flatten sliding window features
        x = self.features[idx:idx + self.seq_len].flatten()
        y = self.targets[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def train_mlp_baseline():
    print("=" * 70)
    print("MLP BASELINE MODEL TRAINING")
    print("=" * 70)
    
    # ===================================================================
    # Configuration
    # ===================================================================
    SEQ_LEN = 6
    BATCH_SIZE = 256
    EPOCHS = 50
    HIDDEN_DIM = 128
    LR = 1e-3
    
    TRAIN_END = '2018-12-31'
    VAL_END = '2021-12-31'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ===================================================================
    # Load Data
    # ===================================================================
    print("\n[1/5] Loading Data...")
    data_path = 'data/raw/pangrango_era5_2005_2025.parquet'
    df = pd.read_parquet(data_path)
    print(f"   Loaded: {df.shape}")
    
    feature_cols = [
        'temperature_2m', 'relative_humidity_2m', 'dewpoint_2m',
        'surface_pressure', 'wind_speed_10m', 'wind_direction_10m',
        'cloud_cover', 'precipitation_lag1', 'elevation'
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"   Features: {len(feature_cols)}")
    
    # ===================================================================
    # Temporal Split
    # ===================================================================
    print("\n[2/5] Temporal Split...")
    train_df, val_df, test_df = temporal_split(df, TRAIN_END, VAL_END)
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Val:   {len(val_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")
    
    # Compute stats from training only
    stats = compute_stats_from_training(train_df, feature_cols)
    
    # ===================================================================
    # Create Datasets
    # ===================================================================
    print("\n[3/5] Creating Datasets...")
    train_dataset = MLPDataset(train_df, feature_cols, SEQ_LEN, stats)
    val_dataset = MLPDataset(val_df, feature_cols, SEQ_LEN, stats)
    test_dataset = MLPDataset(test_df, feature_cols, SEQ_LEN, stats)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples:   {len(val_dataset)}")
    print(f"   Test samples:  {len(test_dataset)}")
    
    # ===================================================================
    # Initialize Model
    # ===================================================================
    INPUT_DIM = SEQ_LEN * len(feature_cols)
    NUM_TARGETS = 3
    
    model = MLPBaseline(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_targets=NUM_TARGETS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    print(f"\n[4/5] Training MLP Baseline ({sum(p.numel() for p in model.parameters()):,} params)...")
    
    # ===================================================================
    # Training Loop
    # ===================================================================
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0
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
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += criterion(pred, y).item()
        
        avg_val = val_loss / max(len(val_loader), 1)
        val_losses.append(avg_val)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'config': {
                    'input_dim': INPUT_DIM,
                    'hidden_dim': HIDDEN_DIM,
                    'num_targets': NUM_TARGETS,
                    'seq_len': SEQ_LEN,
                    'feature_cols': feature_cols
                },
                'stats': stats
            }, "models/mlp_baseline_chkpt.pth")
    
    # ===================================================================
    # Evaluate on Test Set
    # ===================================================================
    print("\n[5/5] Evaluating on Test Set (2022-2025)...")
    
    # Load best model
    checkpoint = torch.load("models/mlp_baseline_chkpt.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Denormalize
    t_mean = stats['t_mean'].numpy()
    t_std = stats['t_std'].numpy()
    
    preds_denorm = preds * t_std + t_mean
    targets_denorm = targets * t_std + t_mean
    
    # Inverse log1p for precipitation
    preds_denorm[:, 0] = np.expm1(preds_denorm[:, 0])
    targets_denorm[:, 0] = np.expm1(targets_denorm[:, 0])
    
    # Clamp
    preds_denorm[:, 0] = np.clip(preds_denorm[:, 0], 0, None)
    preds_denorm[:, 2] = np.clip(preds_denorm[:, 2], 0, 100)
    
    # Compute metrics
    var_names = ['precipitation', 'wind_speed', 'humidity']
    results = {}
    
    print(f"\n{'Variable':<15} {'RMSE':<10} {'MAE':<10} {'Corr':<10}")
    print("-" * 45)
    
    for i, var in enumerate(var_names):
        actual = targets_denorm[:, i]
        predicted = preds_denorm[:, i]
        
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        mae = np.mean(np.abs(predicted - actual))
        
        if np.std(actual) > 0 and np.std(predicted) > 0:
            corr = np.corrcoef(predicted, actual)[0, 1]
        else:
            corr = 0.0
        
        results[var] = {
            'rmse': float(rmse),
            'mae': float(mae),
            'correlation': float(corr)
        }
        
        print(f"{var:<15} {rmse:<10.4f} {mae:<10.4f} {corr:<10.4f}")
    
    # ===================================================================
    # Save Results
    # ===================================================================
    os.makedirs("results/baseline_results", exist_ok=True)
    os.makedirs("results/training_logs", exist_ok=True)
    
    # Save metrics
    with open("results/baseline_results/baseline_metrics.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].set_title('MLP Baseline - Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(val_losses, label='Validation Loss', color='orange')
    axes[1].set_title('MLP Baseline - Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/training_logs/baseline_loss_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print("MLP BASELINE TRAINING COMPLETE!")
    print(f"{'='*70}")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Results: results/baseline_results/baseline_metrics.json")
    print(f"   Loss Curve: results/training_logs/baseline_loss_curve.png")
    
    return results


if __name__ == "__main__":
    train_mlp_baseline()
