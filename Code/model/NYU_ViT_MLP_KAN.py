"""
Note: This model is based on ViT, with an MLP backbone and KAN prediction head.
"""

import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .NYU_TopK import TopKPooling
from models_kan import KAN  # Import the KAN model from your Vision-KAN implementation


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    """
    Standard MLP feed-forward module.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
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


class KANHead(nn.Module):
    """
    Replaces the MLP head with a KAN-based head.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        # Define KAN for classification
        self.kan = KAN([input_dim, hidden_dim, num_classes])

    def forward(self, x):
        # Flatten the input if necessary
        x = x.reshape(-1, x.shape[-1])  # Flatten batch
        x = self.kan(x)  # Apply KAN
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))  # Standard MLP backbone
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=32, dropout=0., emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.topk = TopKPooling(in_channels=112, ratio=0.8)
        self.pos_project = nn.Linear(3, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)  # MLP backbone

        self.pool = pool
        self.to_latent = nn.Identity()

        # Replace the MLP head with a KAN head
        self.mlp_head = KANHead(input_dim=dim, hidden_dim=mlp_dim, num_classes=num_classes)

    def forward(self, x, posi):
        x, posi = self.topk(x, posi)

        b, n, _ = x.shape

        pos_project = self.pos_project(posi)
        x *= pos_project
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1)  # Mean pooling
        x = self.to_latent(x)

        return self.mlp_head(x)
