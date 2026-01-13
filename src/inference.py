
import os
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch

from src.models.diffusion import ConditionalDiffusionModel, RainForecaster
from src.models.gnn import SpatioTemporalGNN
from src.retrieval.base import RetrievalDatabase

def create_inference_graphs(condition_sequence, config, num_nodes=3, device='cpu'):
    """
    Create graph sequence for inference.
    """
    seq_len = config['seq_len']
    # Build edge index (fully connected)
    sources, targets = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)
    edge_index = torch.tensor([sources, targets], dtype=torch.long, device=device)
    
    # Create graph sequence
    graphs_sequence = []
    for t in range(seq_len):
        node_features = condition_sequence[t].unsqueeze(0).repeat(num_nodes, 1)
        graph = Data(x=node_features, edge_index=edge_index)
        batch = Batch.from_data_list([graph])
        graphs_sequence.append(batch.to(device))
    
    return graphs_sequence

class InferenceModelWrapper:
    """Wrapper to hold both GNN and Diffusion model for easy passing around."""
    def __init__(self, st_gnn, forecaster, config):
        self.st_gnn = st_gnn
        self.forecaster = forecaster
        self.config = config
        self.device = 'cpu'
    
    def eval(self):
        self.st_gnn.eval()
        self.forecaster.model.eval()
        
    def to(self, device):
        self.device = device
        self.st_gnn.to(device)
        self.forecaster.model.to(device)
        self.forecaster.device = device  # Update forecaster's device attribute
        return self

def load_model_and_stats(checkpoint_path="models/diffusion_chkpt.pth"):
    """
    Load trained models and statistics.
    Returns: (InferenceModelWrapper, stats, retrieval_db)
    """
    if not os.path.exists(checkpoint_path):
        # Fallback to absolute path if needed, or raise error
        if os.path.exists(os.path.join("..", checkpoint_path)):
            checkpoint_path = os.path.join("..", checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    stats = checkpoint['stats']
    config = checkpoint['config']
    
    # Initialize GNN
    st_gnn = SpatioTemporalGNN(
        node_features=config['context_dim'],
        hidden_dim=config['hidden_dim'] // 2,
        output_dim=config['graph_dim'],
        num_gat_heads=4,
        num_attn_heads=4,
        seq_len=config['seq_len']
    )
    if 'st_gnn_state' in checkpoint:
        st_gnn.load_state_dict(checkpoint['st_gnn_state'])
    else:
        print("Warning: st_gnn_state not found in checkpoint. GNN might be uninitialized.")
    
    # Initialize Diffusion
    diff_model = ConditionalDiffusionModel(
        input_dim=1,
        context_dim=config['context_dim'],
        retrieval_dim=config['retrieval_dim'],
        graph_dim=config['graph_dim'],
        hidden_dim=config['hidden_dim']
    )
    diff_model.load_state_dict(checkpoint['diffusion_state'])
    
    forecaster = RainForecaster(diff_model)
    model_wrapper = InferenceModelWrapper(st_gnn, forecaster, config)
    
    # Rebuild Retrieval DB
    # Note: In a production env, this should be loaded from a file, but here we rebuild from raw data
    print("Rebuilding retrieval database from Training data...")
    try:
        # Try finding data in common locations
        possible_paths = [
            'data/raw/sitaro_era5_2005_2025.parquet',
            '../data/raw/sitaro_era5_2005_2025.parquet',
            os.path.join(os.path.dirname(checkpoint_path), '../data/raw/sitaro_era5_2005_2025.parquet')
        ]
        data_path = None
        for p in possible_paths:
            if os.path.exists(p):
                data_path = p
                break
        
        if data_path:
            df = pd.read_parquet(data_path)
            # Filter for training data only to avoid leakage
            df['date'] = pd.to_datetime(df['date'])
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            # Using config train_end if available, else default
            train_end = config.get('train_end', '2018-12-31')
            train_df = df[df['date'] <= pd.to_datetime(train_end)].copy()
            
            feature_cols = config.get('feature_cols', ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'wind_speed_10m', 'wind_direction_10m'])
            
            train_features = train_df[feature_cols].values
            c_mean = stats['c_mean'].numpy()
            c_std = stats['c_std'].numpy()
            train_features_norm = (train_features - c_mean) / (c_std + 1e-5)
            
            retrieval_db = RetrievalDatabase(embedding_dim=len(feature_cols))
            retrieval_db.add_items(train_features_norm, train_features_norm)
            print(f"Retrieval index rebuilt with {len(train_features_norm)} vectors.")
        else:
            print("Warning: Data file not found. Retrieval DB will be empty.")
            retrieval_db = RetrievalDatabase(embedding_dim=config['context_dim'])
            
    except Exception as e:
        print(f"Error rebuilding retrieval DB: {e}")
        retrieval_db = RetrievalDatabase(embedding_dim=config['context_dim'])

    return model_wrapper, stats, retrieval_db

def run_inference_real(features_norm, model_wrapper, stats, retrieval_db, num_samples=50, device='cpu'):
    """
    Run inference on a single sequence of normalized features.
    
    Args:
        features_norm: tensor [seq_len, features] or [1, seq_len, features]
        model_wrapper: InferenceModelWrapper
        stats: dictionary of stats
        retrieval_db: RetrievalDatabase
        num_samples: int
    """
    if isinstance(device, str):
        device = torch.device(device)
        
    model_wrapper.to(device)
    model_wrapper.eval()
    
    # Ensure input is tensor and move to device
    if not torch.is_tensor(features_norm):
        features_norm = torch.tensor(features_norm, dtype=torch.float32)
    features_norm = features_norm.to(device)
    
    # Handle batch dim
    if features_norm.dim() == 2:
        features_norm = features_norm.unsqueeze(0) # [1, seq_len, feat]
        
    # Check sequence length
    seq_len = features_norm.shape[1]
    cfg_seq_len = model_wrapper.config['seq_len']
    
    # Pad if necessary (simple repeat padding if short, or slice if long)
    if seq_len < cfg_seq_len:
        # Repeat last frame
        last_frame = features_norm[:, -1:, :]
        repeats = cfg_seq_len - seq_len
        features_norm = torch.cat([features_norm, last_frame.repeat(1, repeats, 1)], dim=1)
    elif seq_len > cfg_seq_len:
        features_norm = features_norm[:, -cfg_seq_len:, :]
        
    # Access inner models
    st_gnn = model_wrapper.st_gnn
    forecaster = model_wrapper.forecaster
    config = model_wrapper.config
    
    with torch.no_grad():
        # 1. Graph Embedding
        # Need to create graph objects on the fly
        condition_seq = features_norm[0] # [seq_len, feat] - already on device
        graphs_sequence = create_inference_graphs(condition_seq, config, device=device)
        
        graph_emb = st_gnn(graphs_sequence) # [1, graph_dim]
        
        # 2. Retrieval
        context_last = condition_seq[-1].unsqueeze(0) # [1, feat]
        context_np = context_last.cpu().numpy()
        retrieved = retrieval_db.query(context_np, k=config['k_neighbors'])
        retrieved = retrieved.to(device) # [1, k, feat]
        
        # 3. Sampling
        # Forecaster sample() expects [B, ...]
        # We want multiple samples for this single input.
        # We can replicate inputs to batch size = num_samples for parallel sampling
        
        # Replicate conditioning - everything must be on device
        context_batch = context_last.to(device)
        retrieved_batch = retrieved.to(device)
        graph_emb_batch = graph_emb.to(device)
        
        # Sample
        samples = forecaster.sample(
            condition=context_batch,
            retrieved=retrieved_batch,
            graph_emb=graph_emb_batch,
            num_samples=num_samples
        )
        # samples is [num_samples, 1]
        
        # 4. Denormalize
        t_mean = stats['t_mean'].to(device)
        t_std = stats['t_std'].to(device)
        
        samples_log = samples * t_std + t_mean
        samples_mm = torch.expm1(samples_log)
        samples_mm = torch.clamp(samples_mm, min=0.0)
        
        return samples_mm.cpu().numpy().flatten()
