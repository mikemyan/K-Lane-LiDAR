import torch
import torch.nn as nn
import torch.nn.functional as F

class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=256):
        super().__init__()
        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.create_projection_matrix()

    def create_projection_matrix(self):
        # Gaussian orthogonal random matrix
        self.proj = nn.Parameter(torch.randn(self.nb_features, self.dim_heads), requires_grad=False)

    def forward(self, q, k, v):
        # q, k: (batch, heads, seq, dim_heads)
        # v: (batch, heads, seq, dim_heads)
        q_prime = self.kernel(q)
        k_prime = self.kernel(k)
        D_inv = 1.0 / torch.einsum('bhnd,bhnd->bhn', q_prime, k_prime.sum(dim=2) + 1e-6)
        context = torch.einsum('bhnd,bhnd->bhnd', k_prime, v)
        out = torch.einsum('bhnd,bhnd->bhnd', q_prime, context.sum(dim=2))
        out = out * D_inv.unsqueeze(-1)
        return out

    def kernel(self, x):
        # FAVOR+ kernel approximation (relu or exp)
        x_proj = torch.einsum('bhnd,fd->bhnf', x, self.proj)
        return torch.exp(x_proj)  # or F.relu(x_proj) for relu kernel

class PerformerAttentionMinimal(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., nb_features=256):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3, bias=False)
        self.fast_attention = FastAttention(dim_head, nb_features)
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)
        q = q * self.scale
        out = self.fast_attention(q, k, v)
        out = out.transpose(1, 2).contiguous().view(b, n, -1)
        return self.dropout(self.to_out(out))


class PerformerTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                PerformerAttentionMinimal(dim, heads, dim_head, dropout),
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