import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.registry import register_position


@register_position("rope")
class RotaryEmbedding(nn.Module):
    kind = "rotary"

    def __init__(self, dim, max_seq_len, base=10000.0):
        super().__init__()
        require(dim > 0, "RoPE dim must be > 0")
        require(dim % 2 == 0, f"RoPE requires even dim, got {dim}")
        require(max_seq_len > 0, "RoPE max_seq_len must be > 0")
        require(base > 0, "RoPE base must be > 0")
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


@register_position("proportional_rope")
class ProportionalRotaryEmbedding(RotaryEmbedding):
    """Gemma 4 proportional RoPE.

    Frequencies use the full attention head dimension as the denominator even
    when only a partial rotary fraction is active.
    """

    def __init__(self, dim, max_seq_len, base=10000.0, partial_rotary_factor=1.0, factor=1.0):
        nn.Module.__init__(self)
        require(dim > 0, "proportional RoPE dim must be > 0")
        require(dim % 2 == 0, f"proportional RoPE requires even dim, got {dim}")
        require(max_seq_len > 0, "proportional RoPE max_seq_len must be > 0")
        require(base > 0, "proportional RoPE base must be > 0")
        require(0.0 < partial_rotary_factor <= 1.0, "partial_rotary_factor must be in (0, 1]")
        require(factor > 0, "proportional RoPE factor must be > 0")
        rope_angles = int(partial_rotary_factor * dim // 2)
        require(rope_angles > 0, "proportional RoPE must rotate at least one frequency")
        inv_freq_rotated = 1.0 / (base ** (torch.arange(0, 2 * rope_angles, 2).float() / dim))
        nope_angles = dim // 2 - rope_angles
        if nope_angles > 0:
            inv_freq = torch.cat([inv_freq_rotated, torch.zeros(nope_angles)])
        else:
            inv_freq = inv_freq_rotated
        self.register_buffer("inv_freq", inv_freq / factor)
        self._build_cache(max_seq_len)


@register_position("yarn_rope")
class YaRNRotaryEmbedding(RotaryEmbedding):
    """YaRN NTK-by-parts RoPE scaling for long-context extrapolation."""

    def __init__(
        self,
        dim,
        max_seq_len,
        base=10000.0,
        factor=4.0,
        original_max_seq_len=4096,
        beta_fast=32,
        beta_slow=1,
        attention_factor=None,
    ):
        nn.Module.__init__(self)
        require(dim > 0, "YaRN dim must be > 0")
        require(dim % 2 == 0, f"YaRN requires even dim, got {dim}")
        require(max_seq_len > 0, "YaRN max_seq_len must be > 0")
        require(base > 0, "YaRN base must be > 0")
        require(factor > 0, "YaRN factor must be > 0")
        require(original_max_seq_len > 0, "YaRN original_max_seq_len must be > 0")
        require(beta_fast > 0, "YaRN beta_fast must be > 0")
        require(beta_slow > 0, "YaRN beta_slow must be > 0")
        require(beta_fast > beta_slow, "YaRN beta_fast must be > beta_slow")
        if attention_factor is None:
            attention_factor = 1.0 if factor <= 1.0 else 0.1 * math.log(factor) + 1.0
        require(attention_factor > 0, "YaRN attention_factor must be > 0")
        self.attention_factor = attention_factor
        inv_freq = self._build_yarn_inv_freq(dim, base, factor, original_max_seq_len, beta_fast, beta_slow)
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    @staticmethod
    def _build_yarn_inv_freq(dim, base, factor, original_max_seq_len, beta_fast, beta_slow):
        def find_correction_dim(num_rotations):
            return dim * math.log(original_max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

        low = math.floor(find_correction_dim(beta_fast))
        high = math.ceil(find_correction_dim(beta_slow))
        low = max(low, 0)
        high = min(high, dim - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
        ramp = ramp.clamp(0, 1)
        extrapolation_factor = 1.0 - ramp
        pos_freqs = base ** (torch.arange(0, dim, 2).float() / dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        return inv_freq_interpolation * (1.0 - extrapolation_factor) + inv_freq_extrapolation * extrapolation_factor

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos() * self.attention_factor, persistent=False)
        self.register_buffer("sin_cached", freqs.sin() * self.attention_factor, persistent=False)


@register_position("none")
class NoPosition(nn.Module):
    kind = "none"

    def __init__(self, dim, max_seq_len):
        super().__init__()
        require(dim > 0, "NoPosition dim must be > 0")
        require(max_seq_len > 0, "NoPosition max_seq_len must be > 0")

    def forward(self, seq_len):
        return None


@register_position("alibi")
class ALiBi(nn.Module):
    """No learned parameters. Adds -slope * |distance| bias to attention, with causal mask baked in."""
    kind = "bias"

    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        require(num_heads > 0, "ALiBi num_heads must be > 0")
        require(max_seq_len > 0, "ALiBi max_seq_len must be > 0")
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
        pos = torch.arange(seq_len, device=self.slopes.device)
        distance = pos[:, None] - pos[None, :]
        bias = -self.slopes[:, None, None] * distance[None].abs().float()
        bias = bias.masked_fill(distance[None] < 0, float("-inf"))
        self.register_buffer("bias_cached", bias, persistent=False)

    def forward(self, seq_len):
        if seq_len > self.bias_cached.size(1):
            self._build_cache(seq_len)
        return self.bias_cached[:, :seq_len, :seq_len]


@register_position("t5_relative")
class T5RelativePositionBias(nn.Module):
    """T5 decoder-style logarithmic relative position buckets."""
    kind = "bias"

    def __init__(self, num_heads, max_seq_len, num_buckets=32, max_distance=128):
        super().__init__()
        require(num_heads > 0, "T5 relative bias num_heads must be > 0")
        require(max_seq_len > 0, "T5 relative bias max_seq_len must be > 0")
        require(num_buckets > 1, "T5 relative bias num_buckets must be > 1")
        require(max_distance > num_buckets // 2, "T5 relative bias max_distance must exceed exact buckets")
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        device = self.relative_attention_bias.weight.device
        context_pos = torch.arange(seq_len, device=device)
        memory_pos = torch.arange(seq_len, device=device)
        relative_position = memory_pos[None, :] - context_pos[:, None]
        buckets = self._relative_position_bucket(relative_position)
        self.register_buffer("bucket_cached", buckets, persistent=False)
        causal = memory_pos[None, :] <= context_pos[:, None]
        self.register_buffer("causal_cached", causal, persistent=False)

    def _relative_position_bucket(self, relative_position):
        n = (-relative_position).clamp(min=0)
        max_exact = self.num_buckets // 2
        is_small = n < max_exact
        n_float = n.float().clamp(min=1)
        val_if_large = max_exact + (
            torch.log(n_float / max_exact)
            / math.log(self.max_distance / max_exact)
            * (self.num_buckets - max_exact)
        ).long()
        val_if_large = val_if_large.clamp(max=self.num_buckets - 1)
        return torch.where(is_small, n, val_if_large)

    def forward(self, seq_len):
        if seq_len > self.bucket_cached.size(0):
            self._build_cache(seq_len)
        buckets = self.bucket_cached[:seq_len, :seq_len].to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(buckets).permute(2, 0, 1)
        causal = self.causal_cached[:seq_len, :seq_len].to(values.device)
        return values.masked_fill(~causal.unsqueeze(0), float("-inf"))


class _KERPLEBase(nn.Module):
    kind = "bias"

    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        require(num_heads > 0, "KERPLE num_heads must be > 0")
        require(max_seq_len > 0, "KERPLE max_seq_len must be > 0")
        init = math.log(math.expm1(1.0))
        self.log_scale = nn.Parameter(torch.full((num_heads,), init))
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        pos = torch.arange(seq_len, device=self.log_scale.device)
        distance = pos[:, None] - pos[None, :]
        self.register_buffer("distance_cached", distance.clamp(min=0).float(), persistent=False)
        self.register_buffer("causal_cached", distance >= 0, persistent=False)

    def forward(self, seq_len):
        if seq_len > self.distance_cached.size(0):
            self._build_cache(seq_len)
        distance = self.distance_cached[:seq_len, :seq_len].to(self.log_scale.device)
        bias = -self.kernel(distance)
        causal = self.causal_cached[:seq_len, :seq_len].to(bias.device)
        return bias.masked_fill(~causal.unsqueeze(0), float("-inf"))

    def kernel(self, distance):
        raise NotImplementedError


@register_position("kerple_log")
class KERPLELog(_KERPLEBase):
    """KERPLE logarithmic relative bias: -a_h log(1 + b_h |i-j|)."""

    def __init__(self, num_heads, max_seq_len):
        super().__init__(num_heads, max_seq_len)
        init = math.log(math.expm1(1.0))
        self.log_rate = nn.Parameter(torch.full((num_heads,), init))

    def kernel(self, distance):
        scale = F.softplus(self.log_scale).view(-1, 1, 1)
        rate = F.softplus(self.log_rate).view(-1, 1, 1)
        return scale * torch.log1p(rate * distance.unsqueeze(0))


@register_position("kerple_power")
class KERPLEPower(_KERPLEBase):
    """KERPLE power relative bias: -a_h |i-j|^p_h with 0 < p_h < 2."""

    def __init__(self, num_heads, max_seq_len):
        super().__init__(num_heads, max_seq_len)
        self.power_unconstrained = nn.Parameter(torch.zeros(num_heads))

    def kernel(self, distance):
        scale = F.softplus(self.log_scale).view(-1, 1, 1)
        power = (2.0 * torch.sigmoid(self.power_unconstrained)).view(-1, 1, 1)
        return scale * distance.unsqueeze(0).pow(power)


@register_position("learned")
class LearnedPosition(nn.Module):
    kind = "additive"

    def __init__(self, dim, max_seq_len):
        super().__init__()
        require(dim > 0, "learned position dim must be > 0")
        require(max_seq_len > 0, "learned position max_seq_len must be > 0")
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, seq_len):
        require(seq_len <= self.emb.num_embeddings, (
            f"learned position encoding supports at most {self.emb.num_embeddings} tokens, got {seq_len}"
        ))
        return self.emb.weight[:seq_len]


@register_position("sinusoidal")
class SinusoidalPosition(nn.Module):
    kind = "additive"

    def __init__(self, dim, max_seq_len):
        super().__init__()
        require(dim > 0, "sinusoidal position dim must be > 0")
        require(dim % 2 == 0, f"sinusoidal position requires even dim, got {dim}")
        require(max_seq_len > 0, "sinusoidal position max_seq_len must be > 0")
        pe = torch.zeros(max_seq_len, dim)
        pos = torch.arange(max_seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        require(seq_len <= self.pe.size(0), (
            f"sinusoidal position encoding supports at most {self.pe.size(0)} tokens, got {seq_len}"
        ))
        return self.pe[:seq_len]
