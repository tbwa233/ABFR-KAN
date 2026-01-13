import torch
import torch.nn as nn
from ChebyKANLayer import ChebyKANLayer

class ChebyKANHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, degree=4):
        super().__init__()
        self.layer1 = ChebyKANLayer(input_dim, hidden_dim, degree)
        self.norm1  = nn.LayerNorm(hidden_dim)
        self.layer2 = ChebyKANLayer(hidden_dim, num_classes, degree)

    def forward(self, x):
        # x is expected to be (B, input_dim) or anything whose last dim is input_dim.
        # For safety, we support extra leading dims by flattening.
        orig_shape = x.shape[:-1]    # e.g. (B,) or (B, N)
        feat_dim   = x.shape[-1]

        x = x.reshape(-1, feat_dim)  # (B*, input_dim)
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.layer2(x)

        # restore leading dims with num_classes at the end
        x = x.view(*orig_shape, -1)  # (..., num_classes)
        return x
