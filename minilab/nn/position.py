import math

import torch
import torch.nn as nn

from minilab.registry import register_position


@register_position("rope")
class RotaryEmbedding(nn.Module):
    kind = "rotary"

    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len):
        if seq_len > self.cos_cached.size(0):
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


@register_position("alibi")
class ALiBi(nn.Module):
    """No learned parameters. Adds -slope * |distance| bias to attention, with causal mask baked in."""
    kind = "bias"

    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)
        self._build_cache(max_seq_len)

    @staticmethod
    def _get_slopes(n):
        def _pow2_slopes(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(_pow2_slopes(n))

        closest_pow2 = 2 ** math.floor(math.log2(n))
        base = _pow2_slopes(closest_pow2)
        extra = _pow2_slopes(2 * closest_pow2)
        return torch.tensor(base + extra[0::2][: n - closest_pow2])

    def _build_cache(self, seq_len):
        pos = torch.arange(seq_len)
        distance = pos[:, None] - pos[None, :]
        bias = -self.slopes[:, None, None] * distance[None].abs().float()
        bias = bias.masked_fill(distance[None] < 0, float("-inf"))
        self.register_buffer("bias_cached", bias, persistent=False)

    def forward(self, seq_len):
        if seq_len > self.bias_cached.size(1):
            self._build_cache(seq_len)
        return self.bias_cached[:, :seq_len, :seq_len]


@register_position("learned")
class LearnedPosition(nn.Module):
    kind = "additive"

    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, seq_len):
        return self.emb.weight[:seq_len]


@register_position("sinusoidal")
class SinusoidalPosition(nn.Module):
    kind = "additive"

    def __init__(self, dim, max_seq_len):
        super().__init__()
        pe = torch.zeros(max_seq_len, dim)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        return self.pe[:seq_len]
