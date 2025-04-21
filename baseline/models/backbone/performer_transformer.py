import torch
import torch.nn as nn
from performer_pytorch import SelfAttention

class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn = SelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            causal=False  # Set to True if you need autoregressive attention (not typical for vision)
        )

    def forward(self, x):
        # x shape: (batch, sequence_length, dim)
        return self.attn(x)


class PerformerTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                PerformerAttention(dim, heads, dim_head, dropout),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(mlp_dim, dim),
                    nn.Dropout(dropout)
                )
            ]))

    def forward(self, x):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x)) + x
            x = ff(norm2(x)) + x
        return x