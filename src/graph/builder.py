import torch
from torch_geometric.data import Data, Batch
import numpy as np

class SitaroGraphBuilder:
    def __init__(self, nodes_meta):
        """
        nodes_meta: DataFrame or list of dicts with 'lat', 'lon'
        """
        self.nodes_meta = nodes_meta
        self.num_nodes = len(nodes_meta)
        self.edge_index, self.edge_attr = self._build_topology()
        
    def _build_topology(self):
        """
        Build static edge index based on distance.
        """
        # Fully connected for small graph (3 nodes)
        sources = []
        targets = []
        distances = []
        
        coords = self.nodes_meta[['lat', 'lon']].values
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    sources.append(i)
                    targets.append(j)
                    # Euclidean distance approximation
                    dist = np.linalg.norm(coords[i] - coords[j])
                    distances.append(dist)
                    
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(distances, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr
    
    def build_dynamic_edges(self, wind_speed, wind_direction):
        """
        Build dynamic graph based on wind flow.
        wind_speed: [Num_Nodes]
        wind_direction: [Num_Nodes] (Degrees, 0=North, 90=East)
        """
        sources = []
        targets = []
        weights = []
        
        coords = self.nodes_meta[['lat', 'lon']].values
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j: continue
                
                # Vector from i to j
                d_lat = coords[j][0] - coords[i][0]
                d_lon = coords[j][1] - coords[i][1]
                
                # Angle of vector i->j (0=North, 90=East)
                angle_ij = np.degrees(np.arctan2(d_lon, d_lat))
                if angle_ij < 0: angle_ij += 360
                
                # Difference with wind direction at node i
                wind_dir_i = wind_direction[i]
                diff = abs(angle_ij - wind_dir_i)
                if diff > 180: diff = 360 - diff
                
                # If wind is blowing roughly towards j (within 45 degrees)
                if diff < 45:
                    sources.append(i)
                    targets.append(j)
                    # Weight = Wind Speed * Alignment Factor
                    alignment = np.cos(np.radians(diff))
                    w = wind_speed[i] * alignment
                    weights.append(w.item()) # Ensure float
                    
        if not sources: # Fallback to static if no wind
            return self.edge_index, self.edge_attr
            
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        
        return edge_index, edge_attr

    def build_snapshot(self, feature_matrix, target=None, wind_speed=None, wind_dir=None):
        """
        feature_matrix: [Num_Nodes, Num_Features]
        """
        x = torch.tensor(feature_matrix, dtype=torch.float)
        y = torch.tensor(target, dtype=torch.float) if target is not None else None
        
        # Dynamic Topology
        if wind_speed is not None and wind_dir is not None:
            edge_index, edge_attr = self.build_dynamic_edges(wind_speed, wind_dir)
        else:
            edge_index, edge_attr = self.edge_index, self.edge_attr
            
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

# Helper to collate time-series into graph list
def create_temporal_graphs(df, sequence_length=6):
    """
    df: Dataframe from ingest.py
    Returns: List of Data objects
    """
    pass # To be implemented in data loader
