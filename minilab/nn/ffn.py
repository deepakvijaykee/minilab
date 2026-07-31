import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require, require_finite_number
from minilab.registry import register_ffn


DEFAULT_SITU_GATE_CAP = 4.0
DEFAULT_SITU_UP_CAP = 25.0


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


def softcap(x, cap):
    """Smooth cap cap * tanh(x / cap): linear near the origin, bounded by cap."""
    return cap * torch.tanh(x / cap)


def situ_glu_activation(gate_pre, up_pre, gate_cap, up_cap):
    """SiTU-GLU hidden activation from Kimi K3: softcapped Swish gate times a
    softcapped up branch, so the elementwise product is bounded by
    gate_cap * up_cap while staying SwiGLU-like near the origin."""
    return softcap(gate_pre, gate_cap) * torch.sigmoid(gate_pre) * softcap(up_pre, up_cap)


@register_ffn("situ_glu")
class SiTUGLU(nn.Module):
    """Sigmoid Tanh Unit GLU (Kimi K3): SwiGLU with smooth caps on both factors.

    The unbounded factors of SwiGLU can overflow low-precision activations;
    SiTU-GLU applies softcap(x, cap) = cap * tanh(x / cap) to the linear part of
    the Swish gate and to the up branch independently.
    """

    def __init__(self, dim, hidden_dim, gate_cap=DEFAULT_SITU_GATE_CAP, up_cap=DEFAULT_SITU_UP_CAP):
        super().__init__()
        require(dim > 0, "SiTUGLU dim must be > 0")
        require(hidden_dim > 0, "SiTUGLU hidden_dim must be > 0")
        require_finite_number(gate_cap, "SiTUGLU gate_cap")
        require_finite_number(up_cap, "SiTUGLU up_cap")
        require(gate_cap > 0, "SiTUGLU gate_cap must be > 0")
        require(up_cap > 0, "SiTUGLU up_cap must be > 0")
        self.gate_cap = gate_cap
        self.up_cap = up_cap
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(situ_glu_activation(self.w1(x), self.w2(x), self.gate_cap, self.up_cap))


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
