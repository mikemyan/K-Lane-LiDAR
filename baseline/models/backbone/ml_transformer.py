import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

# implementing dropout to prevent overfitting
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# Using Gaussian kernel function here to capture long-range dependencies
def gaussian_kernel(qk, scale=1.0):
    return torch.exp(- (qk ** 2) / (2 * scale ** 2))

# implementing Performer Attention to improve computation efficiency and scalability to some extent
class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., kernel_fn=gaussian_kernel):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim_head * heads, dim),
            nn.Dropout(dropout)
        )

        self.kernel_fn = kernel_fn
        
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
