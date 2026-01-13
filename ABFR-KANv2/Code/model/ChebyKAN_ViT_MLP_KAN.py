import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
from .NYU_TopK import TopKPooling
from ChebyKANLayer import ChebyKANLayer

# ----------------------------
# Helpers
# ----------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# ----------------------------
# PreNorm
# ----------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# ----------------------------
# Attention
# ----------------------------
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

        self.to_out = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# ----------------------------
# Standard MLP FeedForward
# ----------------------------
class FeedForward(nn.Module):
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

# ----------------------------
# Transformer Encoder
# ----------------------------
class Transformer(nn.Module):
    def __init__(
        self, dim, depth, heads, dim_head, mlp_dim,
        dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

# ----------------------------
# ChebyKAN Classification Head ONLY
# ----------------------------
class ChebyKANHead(nn.Module):
    def __init__(self, dim, hidden_dim, num_classes, degree=4):
        super().__init__()
        self.layer1 = ChebyKANLayer(dim, hidden_dim, degree)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.layer2 = ChebyKANLayer(hidden_dim, num_classes, degree)

    def forward(self, x):
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.layer2(x)
        return x

# ----------------------------
# ViT with ChebyKAN Only in the Final Head
# ----------------------------
class ViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool='cls',
        channels=3,
        dim_head=32,
        dropout=0.,
        emb_dropout=0.,
        degree=4
    ):
        super().__init__()

        assert pool in {"cls", "mean"}

        self.topk = TopKPooling(in_channels=112, ratio=0.8)
        self.pos_project = nn.Linear(3, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Encoder uses ONLY STANDARD MLP
        self.transformer = Transformer(
            dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout=dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        # Classification head uses ChebyKAN
        self.mlp_head = ChebyKANHead(
            dim, mlp_dim, num_classes, degree=degree
        )

    def forward(self, x, posi):
        x, posi = self.topk(x, posi)

        pos_project = self.pos_project(posi)
        x += pos_project
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.to_latent(x)
        return self.mlp_head(x)
