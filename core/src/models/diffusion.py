import torch
import torch.nn as nn
import math
from diffusers import DDPMScheduler, DDIMScheduler

from src.config import FINAL_TARGET_COLS

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class ConditionalDiffusionModel(nn.Module):
    """
    Conditional Diffusion Model with:
    - Time embedding (sinusoidal)
    - Context conditioning (weather features)
    - Retrieval conditioning (FAISS neighbors)
    - Graph conditioning (Spatio-Temporal GNN output) [NEW]
    
    MULTI-OUTPUT: Predicts 3 variables:
    - precipitation (mm/jam)
        - wind_speed_10m (m/s)
    - relative_humidity_2m (%)
    
    This satisfies the thesis requirement:
    "Retrieval-Augmented Diffusion Model dengan Spatio-Temporal Graph Conditioning"
    """
    
    # Multi-output configuration
    NUM_TARGETS = 3
    TARGET_NAMES = FINAL_TARGET_COLS
    
    def __init__(self, input_dim=3, context_dim=64, retrieval_dim=32, 
                 graph_dim=64, hidden_dim=64):
        """
        Args:
            input_dim: Dimension of target (3 for multi-output)
            context_dim: Dimension of current weather features
            retrieval_dim: Dimension of retrieved historical features (k * features)
            graph_dim: Dimension of graph embedding from SpatioTemporalGNN [NEW]
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        self.input_dim = input_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        
        # History/Retrieval embedding processing
        self.retrieval_mlp = nn.Sequential(
            nn.Linear(retrieval_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition embedding processing (current weather)
        self.cond_mlp = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # [NEW] Graph embedding processing (Spatio-Temporal context)
        self.graph_mlp = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # Auxiliary rain-occurrence head (wet/dry) for zero-inflated precipitation.
        self.wet_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # MLP denoiser with skip connection
        self.down1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.SiLU())
        self.down2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.SiLU())
        
        self.mid = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.SiLU())
        
        self.up1 = nn.Sequential(nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU())
        self.out = nn.Linear(hidden_dim, input_dim)

    def _encode_condition(self, context, retrieved=None, graph_emb=None):
        """
        Build conditioning embedding without timestep.
        Used by both diffusion and rain-occurrence auxiliary head.
        """
        cond_emb = self.cond_mlp(context)

        if retrieved is not None:
            # Flatten neighbors if needed: [B, k, F] -> [B, k*F]
            if retrieved.dim() == 3:
                r_flat = retrieved.reshape(retrieved.shape[0], -1)
            else:
                r_flat = retrieved
            r_emb = self.retrieval_mlp(r_flat)
            cond_emb = cond_emb + r_emb

        if graph_emb is not None:
            g_emb = self.graph_mlp(graph_emb)
            cond_emb = cond_emb + g_emb

        return cond_emb

    def compute_wet_logit(self, context, retrieved=None, graph_emb=None):
        cond_emb = self._encode_condition(context, retrieved, graph_emb)
        return self.wet_head(cond_emb)

    def compute_wet_probability(self, context, retrieved=None, graph_emb=None):
        return torch.sigmoid(self.compute_wet_logit(context, retrieved, graph_emb))

    def forward(self, x, t, context, retrieved=None, graph_emb=None, return_wet_logit=False):
        """
        Args:
            x: Noisy target [Batch, 3] for multi-output
            t: Timestep [Batch]
            context: Current weather features [Batch, Context_Dim]
            retrieved: Retrieved historical analogs [Batch, k, Features] or [Batch, k*Features]
            graph_emb: Spatio-Temporal graph embedding [Batch, Graph_Dim] [NEW]
        
        Returns:
            Predicted noise [Batch, 3] for multi-output
        """
        # Embeddings
        t_emb = self.time_mlp(t)
        cond_emb = self._encode_condition(context, retrieved, graph_emb)
        emb = t_emb + cond_emb

        # Network
        h1 = self.down1(x) + emb
        h2 = self.down2(h1)
        
        h_mid = self.mid(h2)
        
        h_up = torch.cat([h_mid, h2], dim=-1)  # Skip connection
        output = self.up1(h_up)
        noise_pred = self.out(output)

        if return_wet_logit:
            wet_logit = self.wet_head(cond_emb)
            return noise_pred, wet_logit
        return noise_pred

class RainForecaster:
    """
    Training and inference wrapper for the Conditional Diffusion Model.
    
    Supports:
    - Context conditioning (weather features)
    - Retrieval conditioning (FAISS neighbors)
    - Graph conditioning (Spatio-Temporal GNN) [NEW]
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        # clip_sample=False: weather z-scores range [-4, +8], not [-1, 1] like images
        self.scheduler = DDPMScheduler(num_train_timesteps=1000, clip_sample=False)

    @staticmethod
    def weighted_noise_loss(noise_pred, noise, target_reference):
        """
        Weighted MSE on predicted noise.
        target_reference is the normalized target tensor used to scale loss.
        """
        error = (noise_pred - noise) ** 2
        weights = torch.ones_like(error)
        weights[target_reference.abs() > 1.0] = 5.0
        weights[target_reference.abs() > 3.0] = 10.0
        return (error * weights).mean()

    @torch.no_grad()
    def sample(self, condition, retrieved=None, graph_emb=None, num_samples=1):
        """
        Generate probabilistic predictions using reverse diffusion.
        MULTI-OUTPUT: Generates all 3 weather variables.
        
        Args:
            condition: [1, C] Current weather features
            retrieved: [1, k, F] Retrieved historical analogs
            graph_emb: [1, G] Spatio-Temporal graph embedding [NEW]
            num_samples: Number of samples to generate (for probabilistic output)
        
        Returns:
            Tensor [num_samples, 3]: Sampled values for 3 variables
        """
        self.model.eval()
        
        # Start from random noise in target dimensionality (3 variables by contract).
        num_targets = self.model.input_dim
        x = torch.randn((num_samples, num_targets)).to(self.device)
        
        # Expand conditioning to match num_samples and move to device
        cond_expanded = condition.repeat(num_samples, 1).to(self.device)
        retr_expanded = retrieved.repeat(num_samples, 1, 1).to(self.device) if retrieved is not None else None
        graph_expanded = graph_emb.repeat(num_samples, 1).to(self.device) if graph_emb is not None else None
        
        # Reverse diffusion (denoising)
        for t in self.scheduler.timesteps:
            # Create batch of timesteps
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model(
                x, 
                t_batch, 
                cond_expanded, 
                retr_expanded,
                graph_expanded  # [NEW]
            )
            
            # Denoising step
            x = self.scheduler.step(noise_pred, t, x).prev_sample
            
        return x

    @torch.no_grad()
    def sample_fast(self, condition, retrieved=None, graph_emb=None, num_samples=1, num_inference_steps=50):
        """
        Fast inference using DDIM scheduler with NaN protection.
        """
        self.model.eval()
        
        # Cache DDIM scheduler to avoid re-creating each call
        if not hasattr(self, '_ddim_cache') or self._ddim_cache[0] != num_inference_steps:
            # clip_sample=False: weather z-scores range [-4, +8], not [-1, 1] like images
            ddim = DDIMScheduler(num_train_timesteps=1000, clip_sample=False)
            ddim.set_timesteps(num_inference_steps)
            self._ddim_cache = (num_inference_steps, ddim)
        ddim = self._ddim_cache[1]
        
        num_targets = self.model.input_dim
        x = torch.randn((num_samples, num_targets)).to(self.device)
        
        cond_expanded = condition.repeat(num_samples, 1).to(self.device)
        retr_expanded = retrieved.repeat(num_samples, 1, 1).to(self.device) if retrieved is not None else None
        graph_expanded = graph_emb.repeat(num_samples, 1).to(self.device) if graph_emb is not None else None
        
        use_amp = self.device != 'cpu' and str(self.device) != 'cpu'
        
        for t in ddim.timesteps:
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                noise_pred = self.model(x, t_batch, cond_expanded, retr_expanded, graph_expanded)
            
            noise_pred = noise_pred.float()
            # Clamp noise prediction to prevent NaN from numerical instability
            noise_pred = torch.clamp(noise_pred, -10.0, 10.0)
            x = ddim.step(noise_pred, t, x).prev_sample
            # Replace any NaN with 0
            x = torch.nan_to_num(x, nan=0.0)
        
        return x

