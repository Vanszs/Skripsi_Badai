"""
MLP Baseline Model untuk Prediksi Deterministik Cuaca

Arsitektur sederhana sebagai baseline pembanding terhadap model utama
(Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning).

Target variables:
- precipitation (mm/jam)
- wind_speed (m/s)
- humidity (%)

Arsitektur:
Input → Linear → ReLU → Linear → ReLU → Linear → Output
"""

import torch
import torch.nn as nn

from src.config import FINAL_TARGET_COLS


class MLPBaseline(nn.Module):
    """
    Multi-Layer Perceptron baseline untuk prediksi deterministik cuaca.
    
    Digunakan sebagai pembanding sederhana terhadap model utama
    (Diffusion + Retrieval + GNN).
    """
    
    TARGET_COLS = FINAL_TARGET_COLS
    
    def __init__(self, input_dim, hidden_dim=128, num_targets=3):
        """
        Args:
            input_dim: Jumlah fitur input (seq_len * num_features)
            hidden_dim: Dimensi hidden layer
            num_targets: Jumlah variabel target (3: precip, wind, humidity)
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_targets = num_targets
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_targets)
        )
    
    def forward(self, x):
        """
        Args:
            x: [Batch, input_dim] - flattened input features
        
        Returns:
            [Batch, num_targets] - predicted target values (normalized)
        """
        return self.network(x)
