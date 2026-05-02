import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.attention_common import (
    _apply_tail_rotary,
    _bool_to_additive_bias,
    _local_attention_bias,
    _merge_attention_bias,
    _QKClipMixin,
    _QKNormClipMixin,
    _scale_linear_heads,
    apply_rotary_emb,
)
from minilab.nn.norm import RMSNorm
from minilab.registry import register_attention


def _append_kv_cache(k, v, past_kv):
    if past_kv is None:
        return k, v, 0
    past_k, past_v = past_kv
    require(past_k.shape[:2] == k.shape[:2], "cached keys must match batch and heads")
    require(past_v.shape[:2] == v.shape[:2], "cached values must match batch and heads")
    require(past_k.size(-1) == k.size(-1), "cached keys must match head dim")
    require(past_v.size(-1) == v.size(-1), "cached values must match head dim")
    require(past_k.size(2) == past_v.size(2), "cached keys and values must have the same sequence length")
    past_len = past_k.size(2)
    return torch.cat([past_k, k], dim=2), torch.cat([past_v, v], dim=2), past_len


def _cached_causal_bias(q_len, kv_len, past_len, device, dtype):
    require(kv_len == past_len + q_len, "cached causal attention expects contiguous KV cache")
    query_pos = torch.arange(past_len, past_len + q_len, device=device).view(q_len, 1)
    key_pos = torch.arange(kv_len, device=device).view(1, kv_len)
    return _bool_to_additive_bias(key_pos <= query_pos, dtype)


def _partial_rope_dim(head_dim, rope_fraction):
    require(0.0 < rope_fraction <= 1.0, "rope_fraction must be in (0, 1]")
    rope_dim = max(2, int(head_dim * rope_fraction))
    if rope_dim % 2 == 1:
        rope_dim -= 1
    return rope_dim


def _apply_partial_rope(q, k, freqs_cis, rope_dim):
    if freqs_cis is None:
        return q, k
    positions = torch.arange(q.size(2), device=q.device)
    return (
        _apply_tail_rotary(q, freqs_cis, positions, rope_dim=rope_dim),
        _apply_tail_rotary(k, freqs_cis, positions, rope_dim=rope_dim),
    )


class _PartialRoPEAttentionMixin:
    default_rope_fraction = 0.25

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0, rope_fraction=None):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        if rope_fraction is None:
            rope_fraction = self.default_rope_fraction
        self.rope_dim = _partial_rope_dim(self.head_dim, rope_fraction)

    def _apply_position(self, q, k, freqs_cis):
        return _apply_partial_rope(q, k, freqs_cis, self.rope_dim)


@register_attention("mha")
class MultiHeadAttention(_QKClipMixin, nn.Module):

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self._init_qk_clip(num_heads)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False, past_kv=None, return_kv=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)
        k, v, past_len = _append_kv_cache(k, v, past_kv)
        causal = is_causal and attn_bias is None
        if past_len > 0:
            require(attn_bias is None, "cached MHA does not support external attention bias")
            attn_bias = _cached_causal_bias(T, k.size(2), past_len, x.device, x.dtype)
            causal = False
        self._record_qk_clip_logits(q, k, attn_bias=attn_bias, is_causal=causal, past_len=past_len)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = self.out(out.transpose(1, 2).reshape(B, T, C))
        if return_kv:
            return out, (k, v)
        return out

    @torch.no_grad()
    def commit_qk_clip_update(self, threshold, balance=0.5):
        require(0.0 <= balance <= 1.0, "qk_clip balance must be in [0, 1]")
        gammas = self._qk_clip_gammas(threshold)
        if gammas is None:
            return
        _scale_linear_heads(self.q_proj, self.num_heads, self.head_dim, gammas.pow(balance))
        _scale_linear_heads(self.k_proj, self.num_heads, self.head_dim, gammas.pow(1.0 - balance))
        self._reset_qk_clip_stats()


@register_attention("gqa")
class GroupedQueryAttention(_QKClipMixin, nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(num_kv_heads > 0, "num_kv_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_group_size = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        kv_dim = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self._init_qk_clip(num_heads)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False, past_kv=None, return_kv=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        k, v, past_len = _append_kv_cache(k, v, past_kv)
        cache = (k, v)
        k_attn = k.repeat_interleave(self.kv_group_size, dim=1)
        v_attn = v.repeat_interleave(self.kv_group_size, dim=1)
        causal = is_causal and attn_bias is None
        if past_len > 0:
            require(attn_bias is None, "cached GQA does not support external attention bias")
            attn_bias = _cached_causal_bias(T, k.size(2), past_len, x.device, x.dtype)
            causal = False
        self._record_qk_clip_logits(q, k_attn, attn_bias=attn_bias, is_causal=causal, past_len=past_len)

        out = F.scaled_dot_product_attention(
            q, k_attn, v_attn, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = self.out(out.transpose(1, 2).reshape(B, T, C))
        if return_kv:
            return out, cache
        return out

    @torch.no_grad()
    def commit_qk_clip_update(self, threshold, balance=0.5):
        require(0.0 <= balance <= 1.0, "qk_clip balance must be in [0, 1]")
        gammas = self._qk_clip_gammas(threshold)
        if gammas is None:
            return
        _scale_linear_heads(self.q_proj, self.num_heads, self.head_dim, gammas.pow(balance))
        kv_gammas = gammas.view(self.num_kv_heads, self.kv_group_size).amin(dim=1)
        _scale_linear_heads(self.k_proj, self.num_kv_heads, self.head_dim, kv_gammas.pow(1.0 - balance))
        self._reset_qk_clip_stats()


@register_attention("mha_qknorm")
class MultiHeadQKNormAttention(_QKNormClipMixin, MultiHeadAttention):
    """MHA with per-head RMSNorm on queries and keys before RoPE/SDPA."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(dim, num_heads, dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False, past_kv=None, return_kv=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)
        k, v, past_len = _append_kv_cache(k, v, past_kv)
        causal = is_causal and attn_bias is None
        if past_len > 0:
            require(attn_bias is None, "cached QK-Norm MHA does not support external attention bias")
            attn_bias = _cached_causal_bias(T, k.size(2), past_len, x.device, x.dtype)
            causal = False
        self._record_qk_clip_logits(q, k, attn_bias=attn_bias, is_causal=causal, past_len=past_len)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = self.out(out.transpose(1, 2).reshape(B, T, C))
        if return_kv:
            return out, (k, v)
        return out


@register_attention("gqa_qknorm")
class GroupedQueryQKNormAttention(_QKNormClipMixin, GroupedQueryAttention):
    """Qwen3/Gemma/GLM-style GQA with per-head RMSNorm on Q and K."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def _project_qkv(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return self.q_norm(q), self.k_norm(k), v

    def _apply_position(self, q, k, freqs_cis):
        if freqs_cis is None:
            return q, k
        return apply_rotary_emb(q, k, *freqs_cis)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False, past_kv=None, return_kv=False):
        B, T, C = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_position(q, k, freqs_cis)
        k, v, past_len = _append_kv_cache(k, v, past_kv)
        cache = (k, v)
        k_attn = k.repeat_interleave(self.kv_group_size, dim=1)
        v_attn = v.repeat_interleave(self.kv_group_size, dim=1)
        causal = is_causal and attn_bias is None
        if past_len > 0:
            require(attn_bias is None, "cached QK-Norm GQA does not support external attention bias")
            attn_bias = _cached_causal_bias(T, k.size(2), past_len, x.device, x.dtype)
            causal = False
        self._record_qk_clip_logits(q, k_attn, attn_bias=attn_bias, is_causal=causal, past_len=past_len)
        out = F.scaled_dot_product_attention(
            q, k_attn, v_attn, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
        )
        out = self.out(out.transpose(1, 2).reshape(B, T, C))
        if return_kv:
            return out, cache
        return out


@register_attention("gated_gqa_qknorm")
class GatedGroupedQueryQKNormAttention(GroupedQueryQKNormAttention):
    """Qwen3-Next-style full attention: QK-Norm GQA with a learned output gate."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_position(q, k, freqs_cis)
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)
        self._record_qk_clip_logits(q, k, attn_bias=attn_bias, is_causal=is_causal and attn_bias is None)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out * torch.sigmoid(self.gate_proj(x)))


@register_attention("gated_gqa_qknorm_partial_rope")
class GatedGroupedQueryQKNormPartialRoPEAttention(_PartialRoPEAttentionMixin, GatedGroupedQueryQKNormAttention):
    """Qwen3-Next-style full attention with gated output and partial RoPE."""


@register_attention("gqa_qknorm_kv_tied")
class KeyValueTiedGroupedQueryQKNormAttention(GroupedQueryQKNormAttention):
    """Gemma-style global attention option where value states reuse key projection."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        self.v_proj = None
        self.v_norm = RMSNorm(self.head_dim)

    def _project_qkv(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        return self.q_norm(q), self.k_norm(k), self.v_norm(k)


@register_attention("gqa_qknorm_partial_rope")
class GroupedQueryQKNormPartialRoPEAttention(_PartialRoPEAttentionMixin, GroupedQueryQKNormAttention):
    """GLM-style GQA with QK-Norm and partial RoPE on the tail of each head."""

    default_rope_fraction = 0.5


@register_attention("sliding_window_gqa_qknorm")
class SlidingWindowGroupedQueryQKNormAttention(GroupedQueryQKNormAttention):
    """Gemma-style local GQA layer: QK-Norm plus causal sliding-window attention."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0, window_size=1024):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        require(window_size > 0, "window_size must be > 0")
        self.window_size = window_size

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_position(q, k, freqs_cis)
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)
        bias = _local_attention_bias(T, self.window_size, x.device, x.dtype, is_causal)
        bias = _merge_attention_bias(bias, attn_bias)
        self._record_qk_clip_logits(q, k, attn_bias=bias, is_causal=False)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("mqa")
class MultiQueryAttention(GroupedQueryAttention):
    """Multi-query attention: all query heads share one KV head."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(dim, num_heads, num_kv_heads=1, dropout=dropout)
