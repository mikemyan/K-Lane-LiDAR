import torch
from torch import nn
import torch.nn.functional as F

class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., nb_features=256):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.nb_features = nb_features
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(dropout)

        # Random projection matrix for FAVOR+
        self.create_projection_matrix()

    def create_projection_matrix(self):
        # Gaussian orthogonal random matrix
        self.proj = nn.Parameter(
            torch.randn(self.nb_features, self.dim_head), requires_grad=False
        )

    def kernel(self, x):
        # FAVOR+ kernel approximation (exponential kernel)
        x_proj = torch.einsum('bhnd,fd->bhnf', x, self.proj)
        return torch.exp(x_proj)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1,2), qkv)

        q = q * self.scale
        q_prime = self.kernel(q)
        k_prime = self.kernel(k)

        k_sum = k_prime.sum(dim=2)  # [b, h, d]
        D = (q_prime * k_sum.unsqueeze(2)).sum(dim=-1)  # [b, h, n]
        D_inv = 1.0 / (D + 1e-6)
        context = torch.einsum('bhmd,bhmd->bhmd', k_prime, v)
        out = torch.einsum('bhnd,bhmd->bhnm', q_prime, context)
        out = out * D_inv.unsqueeze(-1)
        out = out.sum(dim=2)
        out = out.transpose(1,2).contiguous().view(b, n, self.heads * self.dim_head)
        out = self.to_out(out)
        return self.dropout(out)

class NewPerformerTransformer(nn.Module):
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