import torch
from torch import nn
from einops import rearrange
from .NYU_TopK import TopKPooling
from fasterkan import FasterKAN
from .cheby_head import ChebyKANHead

# --------- helpers ----------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# --------- FasterKAN FFN in encoder ----------
class KANFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.kan = FasterKAN([dim, hidden_dim, dim])

    def forward(self, x):
        b, n, d = x.shape
        x = x.reshape(-1, d)
        x = self.kan(x)
        return x.reshape(b, n, -1)

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
                nn.Dropout(dropout),
            )
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, KANFeedForward(dim, mlp_dim)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(
        self,
        *,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=32,
        dropout=0.0,
        emb_dropout=0.0,
        cheby_degree=4,
    ):
        super().__init__()

        assert pool in {"cls", "mean"}, "pool type must be either cls or mean"

        self.topk = TopKPooling(in_channels=112, ratio=0.8)
        self.pos_project = nn.Linear(3, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # ---- ChebyKAN head instead of FasterKAN head ----
        self.mlp_head = ChebyKANHead(
            input_dim=dim,
            hidden_dim=mlp_dim,
            num_classes=num_classes,
            degree=cheby_degree,
        )

    def forward(self, x, posi):
        # Top-k pooling (same as before)
        x, posi = self.topk(x, posi)   # x: (B, N, dim_x), posi: (B, N, 3)

        # positional projection and fusion
        pos_project = self.pos_project(posi)  # (B, N, dim)
        x = x + pos_project
        x = self.dropout(x)

        # transformer encoder
        x = self.transformer(x)  # (B, N, dim)

        # pooling (you’re already using mean pool)
        x = x.mean(dim=1)        # (B, dim)

        x = self.to_latent(x)
        logits = self.mlp_head(x)  # (B, num_classes)
        return logits
