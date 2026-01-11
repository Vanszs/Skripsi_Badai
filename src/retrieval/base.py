import torch
import torch.nn as nn
import faiss
import numpy as np

class WeatherStateEncoder(nn.Module):
    """
    Compresses the weather state of the entire graph (Nodes x Features) 
    into a single embedding vector.
    """
    def __init__(self, input_dim, hidden_dim=64, embedding_dim=32):
        super().__init__()
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh() # Normalize to [-1, 1] roughly
        )
        
    def forward(self, x):
        # x shape: [Batch, Num_Nodes, Features]
        flat = self.flatten(x)
        return self.encoder(flat)

class RetrievalDatabase:
    """
    Wrapper for FAISS Index.
    """
    def __init__(self, embedding_dim):
        self.dimension = embedding_dim
        # L2 Distance Index
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.stored_data = [] # To store metadata (dates, raw values)
        
    def add_items(self, embeddings, data_values):
        """
        embeddings: numpy array [N, dim]
        data_values: numpy array [N, data_dim] (The actual weather values)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError("Dimension mismatch")
            
        if not self.index.is_trained:
            self.index.train(embeddings) # In case we use IVF
            
        self.index.add(embeddings)
        self.stored_data.append(data_values) # List of arrays
        
    def query(self, query_embedding, k=3):
        """
        Find k nearest neighbors.
        Returns: 
          - retrieved_values: [Batch, k, data_dim]
          - distances: [Batch, k]
        """
        distances, indices = self.index.search(query_embedding, k)
        
        # Reconstruct batch results
        # stored_data is a list of blocks, or a big array. 
        # Ideally we concat stored_data into one big array for fast lookup.
        # Ensure stored_data is a single array (check add_items usage)
        if isinstance(self.stored_data, list):
             # cache it to avoid repeated concats if possible, or just concat
             # Since we add all at once in train.py, it's likely one item in list or we just concat
             all_data = np.concatenate(self.stored_data, axis=0)
        else:
             all_data = self.stored_data
        
        batch_size = query_embedding.shape[0]
        data_dim = all_data.shape[1]
        
        # Handle indices
        # indices shape: [Batch, k]
        # We clamp -1 to 0 (or safe index) to avoid crash, then mask out later if needed
        # But for this use-case, we assume we always find neighbors.
        valid_indices = indices.copy()
        valid_indices[valid_indices == -1] = 0 # Safety
        
        # Vectorized lookup
        # all_data[indices] works directly in numpy if indices is int array
        # output shape: [Batch, k, data_dim]
        retrieved_values = all_data[valid_indices]
            
        return torch.tensor(retrieved_values, dtype=torch.float)
