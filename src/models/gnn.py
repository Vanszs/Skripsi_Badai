"""
Spatio-Temporal Graph Neural Network Module

This module implements:
1. SpatioTemporalGNN - Graph Attention for spatial dependencies
2. TemporalAttention - Self-attention for temporal sequences

Scientific Justification:
- Gunung Gede-Pangrango memiliki 3 node observasi (Puncak, Lereng, Hilir)
- Weather patterns propagate via wind (directed edges)
- Temporal patterns are crucial for nowcasting
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import math


class TemporalAttention(nn.Module):
    """
    Self-attention layer for temporal sequence modeling.
    Learns which past timesteps are most relevant for prediction.
    """
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [Batch, Seq_Len, Hidden_Dim]
        Returns: [Batch, Hidden_Dim] (aggregated temporal representation)
        """
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        
        # Aggregate: take mean over sequence (or last timestep)
        return x.mean(dim=1)  # [Batch, Hidden_Dim]


class SpatialGNN(nn.Module):
    """
    Graph Attention Network for spatial dependencies across Gunung Gede-Pangrango nodes.
    Uses GAT to learn weighted message passing based on node features.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        
        # Two-layer GAT
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True, dropout=0.1)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, heads=1, concat=False, dropout=0.1)
        
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """
        x: [Num_Nodes, Input_Dim]
        edge_index: [2, Num_Edges]
        batch: [Num_Nodes] - batch assignment for pooling
        """
        # Layer 1
        h = self.conv1(x, edge_index)
        h = self.relu(h)
        
        # Layer 2
        h = self.conv2(h, edge_index)
        
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
            num_heads=num_attn_heads
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
            # graph.batch: [Total_Nodes] - identifies which sample each node belongs to
            h = self.spatial_gnn(graph.x, graph.edge_index, batch=graph.batch)
            spatial_outputs.append(h)
        
        # Stack temporal: [Batch, Seq_Len, Hidden_Dim]
        temporal_input = torch.stack(spatial_outputs, dim=1)
        
        # Aggregate temporal
        output = self.temporal_attn(temporal_input)  # [Batch, Hidden_Dim]
        
        # Project to output dim
        return self.output_proj(output)  # [Batch, Output_Dim]


class SimpleGraphEncoder(nn.Module):
    """
    Simplified graph encoder for when full temporal sequence is not available.
    Processes a single graph snapshot and outputs embedding.
    
    Use this during inference when only current timestep is available.
    """
    def __init__(self, node_features, hidden_dim=64, output_dim=64):
        super().__init__()
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.graph_pool = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.Tanh()
        )
        
    def forward(self, node_features):
        """
        node_features: [Batch, Num_Nodes, Features] or [Num_Nodes, Features]
        Returns: [Batch, Output_Dim]
        """
        if node_features.dim() == 2:
            node_features = node_features.unsqueeze(0)
        
        # Encode nodes
        h = self.node_encoder(node_features)  # [B, N, H]
        
        # Mean pooling across nodes
        h = h.mean(dim=1)  # [B, H]
        
        return self.graph_pool(h)


# Helper function to create graph from node features
def create_pangrango_graph(node_features, edge_index, edge_attr=None):
    """
    Create a PyG Data object for Gunung Gede-Pangrango nodes.
    
    node_features: Tensor [Num_Nodes, Features]
    edge_index: Tensor [2, Num_Edges]
    edge_attr: Optional Tensor [Num_Edges, Edge_Features]
    """
    return Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
