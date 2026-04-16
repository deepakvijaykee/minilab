import torch
import torch.nn as nn

from minilab.registry import register_norm


@register_norm("rmsnorm")
class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(x.dtype) * self.weight.to(x.dtype)


@register_norm("layernorm")
class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
