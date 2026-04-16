"""AdaLN, DiffusionBlock, SinusoidalTimeEmbedding — shared by MDLM, SEDD, D3PM."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.nn.attention import MultiHeadAttention
from minilab.nn.ffn import SwiGLU


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None] * freqs[None]
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


class AdaLN(nn.Module):
    """Norm conditioned on time: scale and shift from time embedding."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionBlock(nn.Module):

    def __init__(self, dim, num_heads, ffn_hidden, dropout=0.0):
        super().__init__()
        self.norm1 = AdaLN(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = AdaLN(dim)
        self.ffn = SwiGLU(dim, ffn_hidden)

    def forward(self, x, t_emb, freqs_cis=None):
        x = x + self.attn(self.norm1(x, t_emb), freqs_cis=freqs_cis, is_causal=False)
        x = x + self.ffn(self.norm2(x, t_emb))
        return x
