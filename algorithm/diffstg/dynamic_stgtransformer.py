import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional

from .ugnet import TimeEmbedding

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, t_emb):
        t_emb = self.adaLN_modulation(t_emb)
        shift, scale = torch.chunk(t_emb, 2, dim=1)
        return modulate(self.norm(x), shift, scale)


class DynamicTransformerBlock(nn.Module):
    """
    A Transformer block that can handle time-varying attention bias.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.adaLN1 = AdaLN(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.adaLN2 = AdaLN(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, t_emb, attn_bias):
        """
        Args:
            x: Input tensor (B, N, D)
            t_emb: Time embedding (B, D)
            attn_bias: Attention bias (N, N) or (B, N, N) for batch-specific bias
        """
        # Self-attention block with dynamic graph attention bias
        x_modulated = self.adaLN1(x, t_emb)
        
        # Handle batch-specific attention bias
        if len(attn_bias.shape) == 3:  # (B, N, N)
            # Use different bias for each sample in batch
            B, N, D = x.shape
            attn_outputs = []
            for b in range(B):
                attn_out, _ = self.attn(
                    x_modulated[b:b+1], x_modulated[b:b+1], x_modulated[b:b+1], 
                    attn_mask=attn_bias[b]
                )
                attn_outputs.append(attn_out)
            attn_output = torch.cat(attn_outputs, dim=0)
        else:  # (N, N) - same bias for all batch samples
            attn_output, _ = self.attn(x_modulated, x_modulated, x_modulated, attn_mask=attn_bias)
        
        x = x + attn_output

        # Feed-forward block
        x_modulated = self.adaLN2(x, t_emb)
        ffn_output = self.ffn(x_modulated)
        x = x + ffn_output
        return x


class DynamicSTGTransformer(nn.Module):
    """
    Dynamic Graph Transformer with time-varying adjacency matrices.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.d_h = config.d_h
        self.F = config.F
        self.V = config.V
        self.T_p = config.T_p
        self.T_h = config.T_h
        T = self.T_p + self.T_h
        self.T_total = 2 * T
        
        # Load dynamic adjacency matrices if available
        self.use_dynamic_adj = config.get('use_dynamic_adj', False)
        self.dynamic_adj_type = config.get('dynamic_adj_type', 'mobility')  # 'mobility', 'hybrid', etc.
        
        # Model layers initialization
        self.x_proj = nn.Conv2d(self.F, self.d_h, kernel_size=(1, 1))
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_h, self.d_h), nn.SiLU(), nn.Linear(self.d_h, self.d_h),
        )
        self.spatial_embedding = nn.Embedding(self.V, self.d_h)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.T_total, self.d_h))
        
        # Use dynamic transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DynamicTransformerBlock(self.d_h, config.get('n_head', 8), config.get('dropout', 0.1))
            for _ in range(config.get('n_layers', 6))
        ])
        
        self.final_norm = nn.LayerNorm(self.d_h)
        self.out = nn.Sequential(
            nn.Conv2d(self.d_h, self.d_h, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(self.d_h, self.F, kernel_size=(1, 1))
        )
        self.temporal_out_proj = nn.Linear(self.T_total, T)

        # Initialize attention bias
        self._initialize_attention_bias(config)
        
        # Optional: Learnable graph fusion weights
        if self.use_dynamic_adj:
            self.graph_fusion_weight = nn.Parameter(torch.tensor(0.5))  # Balance static vs dynamic
    
    def _initialize_attention_bias(self, config):
        """Initialize attention bias matrices."""
        if self.use_dynamic_adj:
            # Load dynamic adjacency matrices
            try:
                import os
                data_dir = config.get('data_dir', '/home/guanghui/DiffODE/data/dataset/COVID')
                dynamic_adj_path = os.path.join(data_dir, f'dynamic_adj_{self.dynamic_adj_type}.npy')
                self.dynamic_adj_matrices = torch.from_numpy(np.load(dynamic_adj_path)).float()
                print(f"Loaded dynamic adjacency matrices: {self.dynamic_adj_matrices.shape}")
                
                # Store the number of time steps in dynamic adjacency
                self.dynamic_T = self.dynamic_adj_matrices.shape[0]
                
            except FileNotFoundError:
                print(f"Dynamic adjacency file not found, falling back to static adjacency")
                self.use_dynamic_adj = False
        
        # Fallback to static adjacency
        if not self.use_dynamic_adj:
            adj = config.A
            adj = torch.from_numpy(adj + np.eye(self.V, dtype=np.float32))
            graph_connectivity = adj.repeat_interleave(self.T_total, dim=0).repeat_interleave(self.T_total, dim=1)
            attn_bias = torch.zeros_like(graph_connectivity)
            attn_bias[graph_connectivity == 0] = -1e9
            self.register_buffer('static_attn_bias', attn_bias)
    
    def _get_attention_bias(self, batch_size: int, time_step: Optional[int] = None):
        """
        Get attention bias for current time step.
        
        Args:
            batch_size: Batch size
            time_step: Current time step (for time-aware attention)
            
        Returns:
            Attention bias tensor
        """
        if not self.use_dynamic_adj:
            return self.static_attn_bias
        
        # For dynamic graphs, we need to map the time step to our dynamic adjacency
        if time_step is not None:
            # Map diffusion time step to data time step (you may need to adjust this mapping)
            data_time_step = min(time_step % self.dynamic_T, self.dynamic_T - 1)
        else:
            # Use average of all time steps or a representative time step
            data_time_step = self.dynamic_T // 2
        
        # Get dynamic adjacency for this time step
        adj_t = self.dynamic_adj_matrices[data_time_step]  # (V, V)
        
        # Add self-loops
        adj_t = adj_t + torch.eye(self.V, device=adj_t.device, dtype=adj_t.dtype)
        
        # Expand to spatio-temporal graph
        graph_connectivity = adj_t.repeat_interleave(self.T_total, dim=0).repeat_interleave(self.T_total, dim=1)
        
        # Create attention bias
        attn_bias = torch.zeros_like(graph_connectivity)
        
        # Option 1: Hard masking (original approach)
        # attn_bias[graph_connectivity == 0] = -1e9
        
        # Option 2: Soft attention bias using edge weights
        # Normalize edge weights to [-inf, 0] range for attention
        max_weight = torch.max(graph_connectivity)
        if max_weight > 0:
            # Convert edge weights to attention bias: higher weight = less negative bias
            attn_bias = -1e9 * (1 - graph_connectivity / max_weight)
            attn_bias[graph_connectivity > 0] = torch.clamp(attn_bias[graph_connectivity > 0], min=-10, max=0)
        else:
            attn_bias[graph_connectivity == 0] = -1e9
        
        return attn_bias

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: tuple):
        x_masked, _, _ = c
        B = x.shape[0]

        x = torch.cat((x, x_masked), dim=3)
        x = self.x_proj(x)
        
        x = x.permute(0, 2, 3, 1).reshape(B, self.V * self.T_total, self.d_h)

        t_emb = TimeEmbedding(t, self.d_h)
        t_emb = self.time_mlp(t_emb)
        
        sp_emb = self.spatial_embedding.weight.unsqueeze(0).repeat(1, self.T_total, 1)
        sp_emb = sp_emb.reshape(1, self.V * self.T_total, self.d_h)
        
        tp_emb = self.temporal_embedding.repeat(1, self.V, 1)

        x = x + sp_emb + tp_emb

        # Get attention bias (potentially time-dependent)
        # For simplicity, we use a fixed time step, but you could make this time-dependent
        attn_bias = self._get_attention_bias(B, time_step=t[0].item() if len(t) > 0 else None)

        for block in self.transformer_blocks:
            x = block(x, t_emb, attn_bias)

        x = self.final_norm(x)
        x = x.reshape(B, self.V, self.T_total, self.d_h)

        # Temporal projection
        x = x.permute(0, 1, 3, 2)
        x = self.temporal_out_proj(x)
        x = x.permute(0, 2, 1, 3)

        e = self.out(x)

        return e