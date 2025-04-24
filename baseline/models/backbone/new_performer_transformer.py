import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math

class PerformerAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim_head = dim_head
        
        # Adaptive number of features based on sequence length
        # Will be set in forward pass based on input size
        self.nb_features = None
        self.proj = None

        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=False)
        self.to_out = nn.Linear(heads * dim_head, dim)
        self.dropout = nn.Dropout(dropout)

    def create_projection_matrix(self, n):
        # Adaptive feature dimension based on sequence length
        nb_features = min(256, int(math.log(n) * 32))
        
        # Only create new projection if dimensions change
        if self.nb_features != nb_features or self.proj is None:
            self.nb_features = nb_features
            matrix = torch.randn(nb_features, self.dim_head, device=self.to_qkv.weight.device)
            q, _ = torch.qr(matrix)  # orthogonal matrix
            self.proj = nn.Parameter(q, requires_grad=False)

    def kernel_fn(self, x, proj):
        # x shape: [batch_size, num_heads, seq_len, dim_head]
        # proj shape: [nb_features, dim_head]
        
        if self.proj is None:
            raise ValueError("Projection matrix not initialized. Call create_projection_matrix first.")
        
        # Ensure proj is on the same device as x
        if proj.device != x.device:
            proj = proj.to(x.device)
        
        # More explicit computation without einsum
        # [b, h, n, d] @ [f, d].T -> [b, h, n, f]
        x_proj = torch.matmul(x, proj.t())  # Direct multiplication, no need for permute
        
        # Numerical stability
        max_val = torch.max(x_proj, dim=-1, keepdim=True)[0]
        stable_exp = torch.exp(x_proj - max_val)
        
        return stable_exp

    def process_chunk(self, chunk):
        b, n, _ = chunk.shape
        
        # Create or update projection matrix based on sequence length
        self.create_projection_matrix(n)

        # Split computation into smaller chunks for better memory efficiency
        chunk_size = min(128, n)
        total_chunks = (n + chunk_size - 1) // chunk_size
        
        outputs = []
        
        with autocast():  # Enable mixed precision
            # QKV transformation
            qkv = self.to_qkv(chunk).chunk(3, dim=-1)
            q, k, v = map(lambda t: t.view(b, n, self.heads, self.dim_head).transpose(1, 2), qkv)
            
            q = q * self.scale
            
            # Process in chunks
            for i in range(total_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, n)
                
                q_chunk = q[:, :, start_idx:end_idx]
                k_chunk = k[:, :, start_idx:end_idx]
                v_chunk = v[:, :, start_idx:end_idx]
                
                # Compute kernel approximations
                q_prime = self.kernel_fn(q_chunk, self.proj)
                k_prime = self.kernel_fn(k_chunk, self.proj)
                
                # Compute attention efficiently
                k_sum = k_prime.sum(dim=2, keepdim=True)  # [b, h, 1, f]
                D = torch.einsum('bhnf,bhf->bhn', q_prime, k_sum.squeeze(2))  # [b, h, n]
                D_inv = 1.0 / (D + 1e-4)  # Increased epsilon for stability
                
                # Compute context in chunks to save memory
                context = torch.einsum('bhnf,bhnd->bhfd', k_prime, v_chunk)  # [b, h, f, d]
                out_chunk = torch.einsum('bhnf,bhfd->bhnd', q_prime, context)  # [b, h, n, d]
                out_chunk = out_chunk * D_inv.unsqueeze(-1)
                
                outputs.append(out_chunk)

        # Combine chunks
        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(b, n, self.heads * self.dim_head)
        return self.dropout(self.to_out(out))

    def forward(self, x):
        return self.process_chunk(x)  # Process entire input as one chunk if small enough

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