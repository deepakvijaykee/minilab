import torch
import torch.nn as nn

from minilab.checks import require
from minilab.registry import register_norm


@register_norm("rmsnorm")
class RMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        require(dim > 0, "RMSNorm dim must be > 0")
        require(eps > 0, "RMSNorm eps must be > 0")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms).to(x.dtype) * self.weight.to(x.dtype)


@register_norm("zero_centered_rmsnorm")
class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm parameterized around an identity scale.

    Qwen3-Next initializes this norm's learned scale at zero and applies
    `1 + weight`, so weight decay pulls the layer toward identity rather than
    toward a zero output scale.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        require(dim > 0, "ZeroCenteredRMSNorm dim must be > 0")
        require(eps > 0, "ZeroCenteredRMSNorm eps must be > 0")
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        scale = 1.0 + self.weight.to(x.dtype)
        return (x * rms).to(x.dtype) * scale


@register_norm("layernorm")
class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        require(dim > 0, "LayerNorm dim must be > 0")
        require(eps > 0, "LayerNorm eps must be > 0")
        self.norm = nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)
