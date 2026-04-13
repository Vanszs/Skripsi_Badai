"""
Temporal DataLoader for Spatio-Temporal Graph Sequences

This module creates sliding window sequences for temporal modeling.
Each sample contains a sequence of graph snapshots across time.

Author: Skripsi Implementation
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional

from src.config import FINAL_TARGET_COLS


class TemporalGraphDataset(Dataset):
    """
    Dataset that creates sliding window sequences of graphs.
    
    Each sample: (graph_sequence, target, retrieval_context)
    - graph_sequence: List[Data] of length seq_len
    - target: [4] tensor with (precipitation, temperature, wind_speed, humidity)
    - retrieval_context: flattened features for FAISS query
    """
    
    # Multi-output target columns
    TARGET_COLS = FINAL_TARGET_COLS
    
    def __init__(self, 
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_cols: List[str] = None,  # Now multi-output
                 seq_len: int = 6,
                 node_names: List[str] = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur'],
                 edge_index: Optional[torch.Tensor] = None,
                 stats: Optional[dict] = None):
        """
        Args:
            df: DataFrame with columns [date, node, features..., target]
            feature_cols: List of feature column names
            target_col: Name of target column
            seq_len: Number of timesteps in each sequence
            node_names: List of node identifiers
            edge_index: Static edge index for graph [2, num_edges]
            stats: Normalization stats {'t_mean', 't_std', 'c_mean', 'c_std'}
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_cols = target_cols if target_cols else self.TARGET_COLS
        self.num_targets = len(self.target_cols)
        self.seq_len = seq_len
        self.node_names = node_names
        self.num_nodes = len(node_names)
        
        # Build edge_index if not provided (fully connected)
        if edge_index is None:
            self.edge_index = self._build_fully_connected_edges()
        else:
            self.edge_index = edge_index
            
        self.stats = stats
        
        # Pivot data: rows=timestamps, columns=nodes
        self._prepare_data()
        
    def _build_fully_connected_edges(self) -> torch.Tensor:
        """Create fully connected graph edges."""
        sources, targets = [], []
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    sources.append(i)
                    targets.append(j)
        return torch.tensor([sources, targets], dtype=torch.long)
    
    def _prepare_data(self):
        """Pivot DataFrame and create normalized tensors."""
        # Sort by date and node
        self.df = self.df.sort_values(['date', 'node']).reset_index(drop=True)
        
        # Get unique timestamps
        self.timestamps = self.df['date'].unique()
        self.num_timestamps = len(self.timestamps)
        
        # ===== VECTORIZED 3D TENSOR CREATION =====
        # Create node-to-index mapping for correct ordering
        node_to_idx = {name: i for i, name in enumerate(self.node_names)}
        self.df['_node_idx'] = self.df['node'].map(node_to_idx)
        
        # Filter out rows with unmapped nodes
        valid_df = self.df.dropna(subset=['_node_idx']).copy()
        valid_df['_node_idx'] = valid_df['_node_idx'].astype(int)
        
        # Create timestamp-to-index mapping
        ts_sorted = np.sort(self.timestamps)
        self.timestamps = ts_sorted
        ts_to_idx = {ts: i for i, ts in enumerate(ts_sorted)}
        valid_df['_ts_idx'] = valid_df['date'].map(ts_to_idx)
        
        # Pre-allocate arrays
        num_features = len(self.feature_cols)
        feature_data = np.zeros((self.num_timestamps, self.num_nodes, num_features), dtype=np.float32)
        target_data = np.zeros((self.num_timestamps, self.num_nodes, self.num_targets), dtype=np.float32)
        
        # Vectorized fill using advanced indexing
        ts_indices = valid_df['_ts_idx'].values
        node_indices = valid_df['_node_idx'].values
        feature_data[ts_indices, node_indices] = valid_df[self.feature_cols].values.astype(np.float32)
        
        for k, col in enumerate(self.target_cols):
            if col in valid_df.columns:
                target_data[ts_indices, node_indices, k] = valid_df[col].values.astype(np.float32)
        
        # Clean up temporary columns
        self.df.drop(columns=['_node_idx'], inplace=True, errors='ignore')
        
        # Convert to tensors
        self.features = torch.tensor(feature_data, dtype=torch.float32)
        # Shape: [Timestamps, Nodes, Features]
        
        self.targets = torch.tensor(target_data, dtype=torch.float32)
        # Shape: [Timestamps, Nodes, num_targets] for multi-output
        
        # Apply normalization if stats provided
        if self.stats:
            self._normalize()
            
        # Valid indices (accounting for sequence length)
        self.valid_indices = list(range(self.seq_len, self.num_timestamps))
        
        print(f"[TemporalGraphDataset] Prepared:")
        print(f"   Timestamps: {self.num_timestamps}")
        print(f"   Nodes: {self.num_nodes}")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Targets: {self.num_targets} ({self.target_cols})")
        print(f"   Valid samples: {len(self.valid_indices)}")
        
    def _normalize(self):
        """Apply appropriate transform and z-score normalization for multi-output."""
        # targets shape: [Timestamps, Nodes, num_targets]
        # Apply log1p only to precipitation (index 0), not to other variables
        self.targets_transformed = self.targets.clone()
        self.targets_transformed[:, :, 0] = torch.log1p(self.targets[:, :, 0])  # precip
        # Other variables (temp, wind, humidity) - no log transform
        
        # Get stats - now should be [num_targets] tensors
        t_mean = self.stats.get('t_mean', self.targets_transformed.mean(dim=(0, 1)))
        t_std = self.stats.get('t_std', self.targets_transformed.std(dim=(0, 1)))
        c_mean = self.stats.get('c_mean', self.features.mean(dim=(0, 1)))
        c_std = self.stats.get('c_std', self.features.std(dim=(0, 1)))
        
        # Normalize each target variable
        self.targets_norm = (self.targets_transformed - t_mean) / (t_std + 1e-5)
        self.features_norm = (self.features - c_mean) / (c_std + 1e-5)
        
    def set_precomputed_retrieval(self, retrieved_tensor: torch.Tensor):
        """
        Store pre-computed FAISS retrieval results.
        This eliminates the need for FAISS queries during training.
        
        Args:
            retrieved_tensor: [num_samples, k*data_dim] or [num_samples, k, data_dim]
        """
        if retrieved_tensor.ndim == 3:
            # Flatten k neighbors: [N, k, dim] -> [N, k*dim]
            retrieved_tensor = retrieved_tensor.view(retrieved_tensor.shape[0], -1)
        self.precomputed_retrieval = retrieved_tensor
        print(f"   Pre-computed retrieval set: {retrieved_tensor.shape}")

    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int):
        """
        Returns:
            graphs: List of PyG Data objects for the sequence
            target: [num_targets] tensor (mean over nodes)
            context: Flattened features for retrieval [Features]
            retrieved: (optional) pre-computed retrieval [k*data_dim]
        """
        # Get the actual timestamp index
        t = self.valid_indices[idx]
        
        # Build graph sequence: [t-seq_len, t-seq_len+1, ..., t-1]
        graphs = []
        for i in range(self.seq_len):
            t_idx = t - self.seq_len + i
            
            # Node features at this timestep
            if hasattr(self, 'features_norm'):
                node_feats = self.features_norm[t_idx]  # [Nodes, Features]
            else:
                node_feats = self.features[t_idx]
            
            # Create PyG Data object
            graph = Data(
                x=node_feats,
                edge_index=self.edge_index
            )
            graphs.append(graph)
        
        # Target: multi-output at time t (mean over nodes)
        # Shape: [num_targets]
        if hasattr(self, 'targets_norm'):
            target = self.targets_norm[t].mean(dim=0)  # [num_targets]
        else:
            target = self.targets[t].mean(dim=0)
            
        # Context for retrieval: use last timestep features (flattened)
        if hasattr(self, 'features_norm'):
            context = self.features_norm[t-1].mean(dim=0)  # [Features]
        else:
            context = self.features[t-1].mean(dim=0)
        
        # Return pre-computed retrieval if available
        if hasattr(self, 'precomputed_retrieval') and self.precomputed_retrieval is not None:
            retrieved = self.precomputed_retrieval[idx]
            return graphs, target, context, retrieved
            
        return graphs, target, context


def collate_temporal_graphs(batch: List[Tuple]):
    """
    Custom collate function for batching temporal graph sequences.
    Supports both 3-tuple (graphs, target, context) and 4-tuple 
    (graphs, target, context, retrieved) formats.
    
    Returns:
        batched_graphs: List[Batch] of length seq_len
        targets: [Batch_Size, num_targets]
        contexts: [Batch_Size, Features]
        retrieved: [Batch_Size, k*data_dim] (optional, only if pre-computed)
    """
    has_retrieval = len(batch[0]) == 4
    seq_len = len(batch[0][0])
    
    # Initialize lists for each timestep
    timestep_graphs = [[] for _ in range(seq_len)]
    targets = []
    contexts = []
    retrievals = [] if has_retrieval else None
    
    for sample in batch:
        graphs = sample[0]
        targets.append(sample[1])
        contexts.append(sample[2])
        for t, g in enumerate(graphs):
            timestep_graphs[t].append(g)
        if has_retrieval:
            retrievals.append(sample[3])
    
    # Batch graphs per timestep
    batched_graphs = [Batch.from_data_list(graphs) for graphs in timestep_graphs]
    
    # Stack targets and contexts
    targets = torch.stack(targets)
    contexts = torch.stack(contexts)
    
    if has_retrieval:
        retrievals = torch.stack(retrievals)
        return batched_graphs, targets, contexts, retrievals
    
    return batched_graphs, targets, contexts


def create_temporal_dataloader(df: pd.DataFrame,
                               feature_cols: List[str],
                               seq_len: int = 6,
                               batch_size: int = 32,
                               stats: Optional[dict] = None,
                               shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for temporal graph sequences.
    
    Args:
        df: Raw DataFrame from ingest.py
        feature_cols: Feature column names
        seq_len: Sequence length
        batch_size: Batch size
        stats: Normalization stats
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader with custom collate function
    """
    dataset = TemporalGraphDataset(
        df=df,
        feature_cols=feature_cols,
        seq_len=seq_len,
        stats=stats
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_temporal_graphs
    )


# ============================================
# CROSSCHECK FUNCTION
# ============================================
def crosscheck_temporal_loader():
    """
    Crosscheck function to verify the temporal loader works correctly.
    Run this after creating the module.
    """
    print("=" * 60)
    print("CROSSCHECK: Temporal DataLoader")
    print("=" * 60)
    
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    nodes = ['Puncak', 'Lereng_Cibodas', 'Hilir_Cianjur']
    
    data = []
    for date in dates:
        for node in nodes:
            data.append({
                'date': date,
                'node': node,
                'temperature_2m': np.random.randn(),
                'relative_humidity_2m': np.random.randn(),
                'surface_pressure': np.random.randn(),
                'wind_speed_10m': np.random.randn(),
                'wind_direction_10m': np.random.randn(),
                'precipitation': max(0, np.random.randn() * 5)
            })
    
    df = pd.DataFrame(data)
    feature_cols = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 
                    'wind_speed_10m', 'wind_direction_10m']
    
    # Create dataset
    dataset = TemporalGraphDataset(
        df=df,
        feature_cols=feature_cols,
        seq_len=6
    )
    
    print(f"\n✅ Dataset created with {len(dataset)} samples")
    
    # Get one sample
    graphs, target, context = dataset[0]
    
    print(f"\n✅ Single sample structure:")
    print(f"   Graphs: {len(graphs)} timesteps")
    print(f"   Graph[0].x shape: {graphs[0].x.shape}")
    print(f"   Graph[0].edge_index shape: {graphs[0].edge_index.shape}")
    print(f"   Target shape: {target.shape}")
    print(f"   Context shape: {context.shape}")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_temporal_graphs
    )
    
    batched_graphs, targets, contexts = next(iter(loader))
    
    print(f"\n✅ Batched structure:")
    print(f"   Batched graphs: {len(batched_graphs)} timesteps")
    print(f"   Batched graph[0].x shape: {batched_graphs[0].x.shape}")
    print(f"   Targets shape: {targets.shape}")
    print(f"   Contexts shape: {contexts.shape}")
    
    print("\n" + "=" * 60)
    print("✅ CROSSCHECK PASSED: Temporal DataLoader working correctly!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    crosscheck_temporal_loader()
