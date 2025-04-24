import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kernel_fn=nn.ReLU()):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim),
            nn.Dropout(dropout)
        )

        self.kernel_fn = kernel_fn  # Kernel function for the Performer Attention (ReLU is common)
        
    def forward(self, x):
        batch_size, seq_len, dim = x.size()

        # Project input to query, key, value
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # Scale the queries
        q = q * self.scale

        # Compute kernelized attention scores
        qk = torch.matmul(q, k.transpose(-1, -2))

        # Apply kernel function to qk (approximate softmax attention)
        qk = self.kernel_fn(qk)

        # Compute the attention weights (no need for softmax here)
        attn = torch.matmul(qk, v)

        # Project the attention output
        out = rearrange(attn, 'b h n d -> b n (h d)')
        return self.to_out(out)
