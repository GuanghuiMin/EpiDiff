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


class TransformerBlock(nn.Module):
    """
    A single block of the Transformer.
    It now accepts an attention bias tensor.
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
        x_modulated = self.adaLN1(x, t_emb)
        attn_output, _ = self.attn(x_modulated, x_modulated, x_modulated, attn_mask=attn_bias)
        x = x + attn_output

        x_modulated = self.adaLN2(x, t_emb)
        ffn_output = self.ffn(x_modulated)
        x = x + ffn_output
        return x


class STGTransformer(nn.Module):
    """
    A Graph Transformer model using a soft mask (attention bias)
    to incorporate graph structure.
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
        # Store T_total as a class attribute for easy access
        self.T_total = 2 * T

        # Model layers initialization
        self.x_proj = nn.Conv2d(self.F, self.d_h, kernel_size=(1, 1))
        self.time_mlp = nn.Sequential(
            nn.Linear(self.d_h, self.d_h), nn.SiLU(), nn.Linear(self.d_h, self.d_h),
        )
        self.spatial_embedding = nn.Embedding(self.V, self.d_h)
        self.temporal_embedding = nn.Parameter(torch.zeros(1, self.T_total, self.d_h))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(self.d_h, config.get('n_head', 8), config.get('dropout', 0.1))
            for _ in range(config.get('n_layers', 6))
        ])
        self.final_norm = nn.LayerNorm(self.d_h)
        self.out = nn.Sequential(
            nn.Conv2d(self.d_h, self.d_h, kernel_size=(1, 1)),
            nn.GELU(),
            nn.Conv2d(self.d_h, self.F, kernel_size=(1, 1))
        )
        self.temporal_out_proj = nn.Linear(self.T_total, T)

        # Attention Bias creation using the static graph
        adj = config.A
        adj = torch.from_numpy(adj + np.eye(self.V, dtype=np.float32))
        graph_connectivity = adj.repeat_interleave(self.T_total, dim=0).repeat_interleave(self.T_total, dim=1)
        attn_bias = torch.zeros_like(graph_connectivity)
        attn_bias[graph_connectivity == 0] = -1e9
        self.register_buffer('attn_bias', attn_bias)

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

            for block in self.transformer_blocks:
                x = block(x, t_emb, self.attn_bias)

            x = self.final_norm(x)
            x = x.reshape(B, self.V, self.T_total, self.d_h)

            x = x.permute(0, 1, 3, 2)
            x = self.temporal_out_proj(x)
            x = x.permute(0, 2, 1, 3)
            e = self.out(x)

            return e