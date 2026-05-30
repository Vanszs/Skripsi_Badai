"""
Spatio-Temporal Graph Neural Network Module.

Canonical context:
- 5-node star graph (MAIN, UP, DOWN, LEFT, RIGHT)
- Temporal sequence conditioning for diffusion forecasting
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class TemporalAttention(nn.Module):
    """
    Self-attention layer for temporal sequence modeling.
    Learns which past timesteps are most relevant for prediction.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, max_len=64, causal=True):
        super().__init__()
        self.max_len = int(max_len)
        self.causal = bool(causal)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_len, hidden_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)
        
    def forward(self, x):
        """
        x: [Batch, Seq_Len, Hidden_Dim]
        Returns: [Batch, Hidden_Dim] (last-step temporal representation)
        """
        seq_len = x.shape[1]
        if seq_len > self.max_len:
            raise ValueError(
                f"TemporalAttention sequence length {seq_len} exceeds max_len={self.max_len}"
            )

        # Add positional encoding to keep timestep order identifiable.
        x = x + self.pos_embedding[:, :seq_len, :]

        # Causal mask ensures timestep t cannot attend to future timesteps > t.
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )

        # Self-attention
        attn_out, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = self.norm(x + self.dropout(attn_out))

        # Use last timestep representation for one-step-ahead conditioning.
        return x[:, -1, :]  # [Batch, Hidden_Dim]


class SpatialGNN(nn.Module):
    """
    Graph Attention Network for spatial dependencies across Gunung Gede-Pangrango nodes.
    Uses GAT to learn weighted message passing based on node features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        
        # Two-layer GAT with edge attributes (static star distances).
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True, dropout=0.1, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.1, edge_dim=1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x: [Num_Nodes, Input_Dim]
        edge_index: [2, Num_Edges]
        edge_attr: [Num_Edges, 1] - per-edge distance weights
        batch: [Num_Nodes] - batch assignment for pooling
        """
        # Layer 1
        h = self.conv1(x, edge_index, edge_attr=edge_attr)
        h = self.relu(h)
        
        # Layer 2
        h = self.conv2(h, edge_index, edge_attr=edge_attr)
        
        # Global pooling to get graph-level representation
        if batch is not None:
            h = global_mean_pool(h, batch)  # [Batch, Output_Dim]
        
        return h


class SpatioTemporalGNN(nn.Module):
    """
    Combined Spatio-Temporal Graph Neural Network.
    
    Architecture:
    1. Spatial GNN processes each timestep's graph independently
    2. Temporal Attention aggregates across timesteps
    3. Output is a fixed-size graph embedding for conditioning diffusion
    
    This satisfies the "Spatio-Temporal Graph Conditioning" in thesis title.
    """
    def __init__(self, node_features, hidden_dim=64, output_dim=64, 
                 num_gat_heads=4, num_attn_heads=4, seq_len=6):
        super().__init__()
        
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        
        # Spatial component
        self.spatial_gnn = SpatialGNN(
            input_dim=node_features,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_heads=num_gat_heads
        )
        
        # Temporal component
        self.temporal_attn = TemporalAttention(
            hidden_dim=hidden_dim,
            num_heads=num_attn_heads,
            max_len=seq_len,
            causal=True,
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, graphs_sequence):
        """
        graphs_sequence: List of PyG Data objects, length = seq_len
                        Each Data has x=[Num_Nodes, Features], edge_index, etc.
        
        Returns: [Batch, Output_Dim] - Graph embedding for diffusion conditioning
        """
        # Process each timestep with spatial GNN
        spatial_outputs = []
        for graph in graphs_sequence:
            # graph.x: [Total_Nodes_In_Batch, Features]
            # graph.edge_index: [2, Total_Edges]
            # graph.edge_attr: [Total_Edges, 1] - per-edge distance weights
            # graph.batch: [Total_Nodes] - identifies which sample each node belongs to
            edge_attr = getattr(graph, "edge_attr", None)
            h = self.spatial_gnn(graph.x, graph.edge_index, edge_attr=edge_attr, batch=graph.batch)
            spatial_outputs.append(h)
        
        # Stack temporal: [Batch, Seq_Len, Hidden_Dim]
        temporal_input = torch.stack(spatial_outputs, dim=1)
        
        # Aggregate temporal
        output = self.temporal_attn(temporal_input)  # [Batch, Hidden_Dim]
        
        # Project to output dim
        return self.output_proj(output)  # [Batch, Output_Dim]
