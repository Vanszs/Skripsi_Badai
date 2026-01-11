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


class TemporalGraphDataset(Dataset):
    """
    Dataset that creates sliding window sequences of graphs.
    
    Each sample: (graph_sequence, target, retrieval_context)
    - graph_sequence: List[Data] of length seq_len
    - target: precipitation value to predict
    - retrieval_context: flattened features for FAISS query
    """
    
    def __init__(self, 
                 df: pd.DataFrame,
                 feature_cols: List[str],
                 target_col: str = 'precipitation',
                 seq_len: int = 6,
                 node_names: List[str] = ['Siau', 'Tagulandang', 'Biaro'],
                 edge_index: Optional[torch.Tensor] = None,
                 stats: Optional[dict] = None):
        """
        Args:
            df: DataFrame with columns [date, node_id, features..., target]
            feature_cols: List of feature column names
            target_col: Name of target column
            seq_len: Number of timesteps in each sequence
            node_names: List of node identifiers
            edge_index: Static edge index for graph [2, num_edges]
            stats: Normalization stats {'t_mean', 't_std', 'c_mean', 'c_std'}
        """
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
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
        self.df = self.df.sort_values(['date', 'node_id']).reset_index(drop=True)
        
        # Get unique timestamps
        self.timestamps = self.df['date'].unique()
        self.num_timestamps = len(self.timestamps)
        
        # Create 3D tensor: [Timestamps, Nodes, Features]
        feature_data = []
        target_data = []
        
        for ts in self.timestamps:
            ts_df = self.df[self.df['date'] == ts]
            
            # Ensure correct node order
            node_features = []
            node_targets = []
            for node in self.node_names:
                node_row = ts_df[ts_df['node_id'] == node]
                if len(node_row) > 0:
                    features = node_row[self.feature_cols].values[0]
                    target = node_row[self.target_col].values[0]
                else:
                    # Missing node: use zeros
                    features = np.zeros(len(self.feature_cols))
                    target = 0.0
                node_features.append(features)
                node_targets.append(target)
            
            feature_data.append(node_features)
            target_data.append(node_targets)
        
        # Convert to tensors
        self.features = torch.tensor(np.array(feature_data), dtype=torch.float32)
        # Shape: [Timestamps, Nodes, Features]
        
        self.targets = torch.tensor(np.array(target_data), dtype=torch.float32)
        # Shape: [Timestamps, Nodes]
        
        # Apply normalization if stats provided
        if self.stats:
            self._normalize()
            
        # Valid indices (accounting for sequence length)
        self.valid_indices = list(range(self.seq_len, self.num_timestamps))
        
        print(f"[TemporalGraphDataset] Prepared:")
        print(f"   Timestamps: {self.num_timestamps}")
        print(f"   Nodes: {self.num_nodes}")
        print(f"   Features: {len(self.feature_cols)}")
        print(f"   Valid samples: {len(self.valid_indices)}")
        
    def _normalize(self):
        """Apply log transform and z-score normalization."""
        # Log transform targets
        self.targets_log = torch.log1p(self.targets)
        
        t_mean = self.stats.get('t_mean', self.targets_log.mean())
        t_std = self.stats.get('t_std', self.targets_log.std())
        c_mean = self.stats.get('c_mean', self.features.mean(dim=(0, 1)))
        c_std = self.stats.get('c_std', self.features.std(dim=(0, 1)))
        
        self.targets_norm = (self.targets_log - t_mean) / (t_std + 1e-5)
        self.features_norm = (self.features - c_mean) / (c_std + 1e-5)
        
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[List[Data], torch.Tensor, torch.Tensor]:
        """
        Returns:
            graphs: List of PyG Data objects for the sequence
            target: Target precipitation [Nodes] or [1] (mean over nodes)
            context: Flattened features for retrieval [Features]
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
        
        # Target: precipitation at time t (mean over nodes for simplicity)
        if hasattr(self, 'targets_norm'):
            target = self.targets_norm[t].mean()  # Scalar
        else:
            target = self.targets[t].mean()
            
        # Context for retrieval: use last timestep features (flattened)
        if hasattr(self, 'features_norm'):
            context = self.features_norm[t-1].mean(dim=0)  # [Features]
        else:
            context = self.features[t-1].mean(dim=0)
            
        return graphs, target.unsqueeze(0), context


def collate_temporal_graphs(batch: List[Tuple]) -> Tuple[List[Batch], torch.Tensor, torch.Tensor]:
    """
    Custom collate function for batching temporal graph sequences.
    
    Args:
        batch: List of (graphs, target, context) tuples
        
    Returns:
        batched_graphs: List[Batch] of length seq_len, each Batch contains all samples
        targets: [Batch_Size, 1]
        contexts: [Batch_Size, Features]
    """
    seq_len = len(batch[0][0])
    batch_size = len(batch)
    
    # Initialize lists for each timestep
    timestep_graphs = [[] for _ in range(seq_len)]
    targets = []
    contexts = []
    
    for graphs, target, context in batch:
        for t, g in enumerate(graphs):
            timestep_graphs[t].append(g)
        targets.append(target)
        contexts.append(context)
    
    # Batch graphs per timestep
    batched_graphs = [Batch.from_data_list(graphs) for graphs in timestep_graphs]
    
    # Stack targets and contexts
    targets = torch.stack(targets)
    contexts = torch.stack(contexts)
    
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
    nodes = ['Siau', 'Tagulandang', 'Biaro']
    
    data = []
    for date in dates:
        for node in nodes:
            data.append({
                'date': date,
                'node_id': node,
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
