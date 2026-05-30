import torch
import faiss
import numpy as np

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
        
        # Cache concatenated data to avoid repeated memory allocation
        if not hasattr(self, '_cached_data') or self._cached_data is None:
            if isinstance(self.stored_data, list) and len(self.stored_data) > 0:
                # Concatenate once and cache as float32 to save memory
                self._cached_data = np.concatenate(self.stored_data, axis=0).astype(np.float32)
            else:
                self._cached_data = np.array(self.stored_data, dtype=np.float32)
        
        all_data = self._cached_data
        
        batch_size = query_embedding.shape[0]
        data_dim = all_data.shape[1] if len(all_data.shape) > 1 else 1
        
        # Handle indices - clamp -1 to 0 for safety
        valid_indices = indices.copy()
        valid_indices[valid_indices == -1] = 0
        
        # Vectorized lookup
        retrieved_values = all_data[valid_indices]
            
        return torch.tensor(retrieved_values, dtype=torch.float32)
