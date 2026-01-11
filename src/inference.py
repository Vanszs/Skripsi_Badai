"""
Inference Pipeline for FULL Spatio-Temporal Graph Conditioned Nowcasting

This script demonstrates:
1. Loading trained SpatioTemporalGNN + DiffusionModel
2. Creating graph sequences for inference
3. Running probabilistic inference with all conditioning
4. Generating uncertainty quantification
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data, Batch

from src.models.diffusion import ConditionalDiffusionModel, RainForecaster
from src.models.gnn import SpatioTemporalGNN
from src.retrieval.base import RetrievalDatabase


def load_models(checkpoint_path="models/diffusion_chkpt.pth"):
    """
    Load trained SpatioTemporalGNN and DiffusionModel.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    stats = checkpoint['stats']
    config = checkpoint['config']
    
    print(f"   Config: seq_len={config['seq_len']}, graph_dim={config['graph_dim']}")
    
    # Initialize SpatioTemporalGNN
    st_gnn = SpatioTemporalGNN(
        node_features=config['context_dim'],
        hidden_dim=config['hidden_dim'] // 2,
        output_dim=config['graph_dim'],
        num_gat_heads=4,
        num_attn_heads=4,
        seq_len=config['seq_len']
    )
    st_gnn.load_state_dict(checkpoint['st_gnn_state'])
    st_gnn.eval()
    print("   SpatioTemporalGNN loaded.")
    
    # Initialize Diffusion Model
    diff_model = ConditionalDiffusionModel(
        input_dim=1,
        context_dim=config['context_dim'],
        retrieval_dim=config['retrieval_dim'],
        graph_dim=config['graph_dim'],
        hidden_dim=config['hidden_dim']
    )
    diff_model.load_state_dict(checkpoint['diffusion_state'])
    print("   DiffusionModel loaded.")
    
    # Rebuild retrieval database
    print("   Rebuilding retrieval database...")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'raw', 'sitaro_era5_2005_2025.parquet')
    
    if not os.path.exists(data_path):
        data_path = 'data/raw/sitaro_era5_2005_2025.parquet'
    
    df = pd.read_parquet(data_path)
    feature_cols = config['feature_cols']
    available_cols = [c for c in feature_cols if c in df.columns]
    
    context_data = df[available_cols].values
    c_mean = stats['c_mean'].numpy()
    c_std = stats['c_std'].numpy()
    context_norm = (context_data - c_mean) / (c_std + 1e-5)
    
    retrieval_db = RetrievalDatabase(embedding_dim=len(available_cols))
    retrieval_db.add_items(context_norm, context_norm)
    print(f"   Retrieval index ready ({len(context_norm)} items).")
    
    return st_gnn, diff_model, stats, config, retrieval_db


def create_inference_graphs(condition_sequence, config, num_nodes=3):
    """
    Create graph sequence for inference.
    
    Args:
        condition_sequence: Tensor [seq_len, features] - sequence of weather conditions
        config: Model config
        num_nodes: Number of graph nodes (3 for Sitaro)
    
    Returns:
        List of PyG Batch objects (each batch has 1 sample)
    """
    seq_len = config['seq_len']
    
    # Build edge index (fully connected)
    sources, targets = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sources.append(i)
                targets.append(j)
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    
    # Create graph sequence
    graphs_sequence = []
    for t in range(seq_len):
        # Replicate condition for each node (simplified - in practice each node has own features)
        node_features = condition_sequence[t].unsqueeze(0).repeat(num_nodes, 1)
        
        graph = Data(x=node_features, edge_index=edge_index)
        batch = Batch.from_data_list([graph])  # Single sample batch
        graphs_sequence.append(batch)
    
    return graphs_sequence


def run_inference(condition_sequence, st_gnn, diff_model, stats, config, retrieval_db,
                  num_samples=50, device='cpu'):
    """
    Run probabilistic inference with FULL Spatio-Temporal conditioning.
    
    Args:
        condition_sequence: Tensor [seq_len, features] - normalized weather sequence
        st_gnn: Trained SpatioTemporalGNN
        diff_model: Trained DiffusionModel
        stats: Normalization stats
        config: Model config
        retrieval_db: FAISS database
        num_samples: Number of probabilistic samples
        device: 'cpu' or 'cuda'
    
    Returns:
        np.ndarray: Predicted rainfall samples in mm
    """
    st_gnn.to(device)
    diff_model.to(device)
    
    forecaster = RainForecaster(diff_model, device=device)
    
    # 1. Create graph sequence
    graphs_sequence = create_inference_graphs(condition_sequence, config)
    graphs_sequence = [g.to(device) for g in graphs_sequence]
    
    # 2. Get Spatio-Temporal Graph Embedding
    with torch.no_grad():
        graph_emb = st_gnn(graphs_sequence)  # [1, graph_dim]
    
    # 3. Get retrieval context (use last timestep)
    context_last = condition_sequence[-1].unsqueeze(0)  # [1, features]
    retrieved = retrieval_db.query(context_last.numpy(), k=config['k_neighbors'])
    
    # 4. Generate samples
    samples = forecaster.sample(
        condition=context_last.to(device),
        retrieved=retrieved.to(device),
        graph_emb=graph_emb.to(device),
        num_samples=num_samples
    )
    
    # 5. Denormalize
    t_mean = stats['t_mean']
    t_std = stats['t_std']
    
    samples_log = samples * t_std + t_mean
    samples_mm = torch.expm1(samples_log)
    samples_mm = torch.clamp(samples_mm, min=0.0)
    
    return samples_mm.cpu().numpy().flatten()


def analyze_predictions(samples, thresholds=[50, 100, 150]):
    """
    Analyze probabilistic predictions.
    """
    results = {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples)),
        'min': float(np.min(samples)),
        'max': float(np.max(samples)),
        'p10': float(np.percentile(samples, 10)),
        'p50': float(np.percentile(samples, 50)),
        'p90': float(np.percentile(samples, 90)),
    }
    
    for t in thresholds:
        results[f'P(>{t}mm)'] = float(np.mean(samples > t))
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("FULL SPATIO-TEMPORAL PROBABILISTIC RAIN NOWCASTING INFERENCE")
    print("=" * 70)
    
    try:
        st_gnn, diff_model, stats, config, retrieval_db = load_models()
        
        # Create mock condition sequence (seq_len timesteps)
        seq_len = config['seq_len']
        context_dim = config['context_dim']
        
        mock_sequence = torch.randn(seq_len, context_dim)
        
        print(f"\nRunning inference with {seq_len}-step sequence...")
        predictions = run_inference(
            condition_sequence=mock_sequence,
            st_gnn=st_gnn,
            diff_model=diff_model,
            stats=stats,
            config=config,
            retrieval_db=retrieval_db,
            num_samples=50
        )
        
        print("\n" + "=" * 70)
        print("PREDICTION RESULTS (50 probabilistic samples)")
        print("=" * 70)
        
        analysis = analyze_predictions(predictions)
        for key, value in analysis.items():
            if 'P(' in key:
                print(f"   {key}: {value*100:.1f}%")
            else:
                print(f"   {key}: {value:.2f} mm")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("   Please run 'python src/train.py' first to train the model.")
