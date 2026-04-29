import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.registry import register_ffn


@register_ffn("swiglu")
class SwiGLU(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        require(dim > 0, "SwiGLU dim must be > 0")
        require(hidden_dim > 0, "SwiGLU hidden_dim must be > 0")
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


@register_ffn("gelu")
class GELUFFN(nn.Module):

    def __init__(self, dim, hidden_dim):
        super().__init__()
        require(dim > 0, "GELUFFN dim must be > 0")
        require(hidden_dim > 0, "GELUFFN hidden_dim must be > 0")
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


@register_ffn("gelu_tanh")
class GELUTanhFFN(nn.Module):
    """GELU MLP using the tanh approximation used by Gemma-family configs."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        require(dim > 0, "GELUTanhFFN dim must be > 0")
        require(hidden_dim > 0, "GELUTanhFFN hidden_dim must be > 0")
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x), approximate="tanh"))


@register_ffn("geglu")
class GEGLU(nn.Module):
    """Gated GELU FFN used by T5/PaLM-family GLU ablations."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        require(dim > 0, "GEGLU dim must be > 0")
        require(hidden_dim > 0, "GEGLU hidden_dim must be > 0")
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


@register_ffn("reglu")
class ReGLU(nn.Module):
    """Gated ReLU FFN from the GLU variants family."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        require(dim > 0, "ReGLU dim must be > 0")
        require(hidden_dim > 0, "ReGLU hidden_dim must be > 0")
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.relu(self.w1(x)) * self.w2(x))
