import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.norm import RMSNorm
from minilab.registry import register_attention


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(q, k, cos, sin):
    """q,k: (B,H,T,D), cos,sin: (T,D/2)."""
    cos = cos[None, None]
    sin = sin[None, None]
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return (
        q * cos + rotate_half(q) * sin,
        k * cos + rotate_half(k) * sin,
    )


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


class _QKClipMixin:
    def _init_qk_clip(self, num_heads):
        self.register_buffer("_qk_clip_max_logits", torch.zeros(num_heads), persistent=False)
        self.register_buffer("_qk_clip_seen", torch.zeros((), dtype=torch.bool), persistent=False)
        self._qk_clip_recording = False

    @torch.no_grad()
    def set_qk_clip_recording(self, enabled):
        self._qk_clip_recording = enabled
        if not enabled:
            self._reset_qk_clip_stats()

    def _record_qk_clip_logits(self, q, k):
        if not self._qk_clip_recording or not torch.is_grad_enabled():
            return
        with torch.no_grad():
            scores = torch.matmul(q.detach().float(), k.detach().float().transpose(-2, -1)) / math.sqrt(q.size(-1))
            max_logits = scores.amax(dim=(0, 2, 3)).to(self._qk_clip_max_logits.dtype)
            current = torch.where(
                self._qk_clip_seen,
                torch.maximum(self._qk_clip_max_logits, max_logits),
                max_logits,
            )
            self._qk_clip_max_logits.copy_(current)
            self._qk_clip_seen.fill_(True)

    @torch.no_grad()
    def _qk_clip_gammas(self, threshold):
        require(threshold > 0, "qk_clip threshold must be > 0")
        if not bool(self._qk_clip_seen.item()):
            return None
        max_logits = self._qk_clip_max_logits.float().clamp(min=torch.finfo(torch.float32).tiny)
        return torch.minimum(torch.ones_like(max_logits), threshold / max_logits)

    @torch.no_grad()
    def _reset_qk_clip_stats(self):
        self._qk_clip_max_logits.zero_()
        self._qk_clip_seen.zero_()


def _scale_linear_heads(linear, num_heads, head_dim, scales):
    weight = linear.weight.view(num_heads, head_dim, linear.weight.size(1))
    weight.mul_(scales.to(device=weight.device, dtype=weight.dtype).view(num_heads, 1, 1))


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

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)
        self._record_qk_clip_logits(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))

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

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)
        self._record_qk_clip_logits(q, k)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))

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
class MultiHeadQKNormAttention(MultiHeadAttention):
    """MHA with per-head RMSNorm on queries and keys before RoPE/SDPA."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(dim, num_heads, dropout)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)
        self._record_qk_clip_logits(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))

    @torch.no_grad()
    def commit_qk_clip_update(self, threshold, balance=0.5):
        require(0.0 <= balance <= 1.0, "qk_clip balance must be in [0, 1]")
        gammas = self._qk_clip_gammas(threshold)
        if gammas is None:
            return
        gamma = gammas.amin()
        self.q_norm.weight.mul_(gamma.pow(balance).to(self.q_norm.weight.device, self.q_norm.weight.dtype))
        self.k_norm.weight.mul_(gamma.pow(1.0 - balance).to(self.k_norm.weight.device, self.k_norm.weight.dtype))
        self._reset_qk_clip_stats()


@register_attention("gqa_qknorm")
class GroupedQueryQKNormAttention(GroupedQueryAttention):
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

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_position(q, k, freqs_cis)
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)
        self._record_qk_clip_logits(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))

    @torch.no_grad()
    def commit_qk_clip_update(self, threshold, balance=0.5):
        require(0.0 <= balance <= 1.0, "qk_clip balance must be in [0, 1]")
        gammas = self._qk_clip_gammas(threshold)
        if gammas is None:
            return
        gamma = gammas.amin()
        self.q_norm.weight.mul_(gamma.pow(balance).to(self.q_norm.weight.device, self.q_norm.weight.dtype))
        self.k_norm.weight.mul_(gamma.pow(1.0 - balance).to(self.k_norm.weight.device, self.k_norm.weight.dtype))
        self._reset_qk_clip_stats()


@register_attention("gated_gqa_qknorm")
class GatedGroupedQueryQKNormAttention(GroupedQueryQKNormAttention):
    """Qwen3-Next full attention: QK-Norm GQA with a learned output gate."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        self.gate_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q, k, v = self._project_qkv(x)
        q, k = self._apply_position(q, k, freqs_cis)
        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)
        self._record_qk_clip_logits(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out(out * torch.sigmoid(self.gate_proj(x)))


@register_attention("gated_gqa_qknorm_partial_rope")
class GatedGroupedQueryQKNormPartialRoPEAttention(GatedGroupedQueryQKNormAttention):
    """Qwen3-Next full attention with gated output and partial RoPE."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0, rope_fraction=0.25):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        self.rope_dim = _partial_rope_dim(self.head_dim, rope_fraction)

    def _apply_position(self, q, k, freqs_cis):
        return _apply_partial_rope(q, k, freqs_cis, self.rope_dim)


@register_attention("gqa_qknorm_kv_tied")
class KeyValueTiedGroupedQueryQKNormAttention(GroupedQueryQKNormAttention):
    """Gemma 4 MoE global attention option where value states reuse key projection."""

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
class GroupedQueryQKNormPartialRoPEAttention(GroupedQueryQKNormAttention):
    """GLM-style GQA with QK-Norm and partial RoPE on the tail of each head."""

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0, rope_fraction=0.5):
        super().__init__(dim, num_heads, num_kv_heads, dropout)
        self.rope_dim = _partial_rope_dim(self.head_dim, rope_fraction)

    def _apply_position(self, q, k, freqs_cis):
        return _apply_partial_rope(q, k, freqs_cis, self.rope_dim)


@register_attention("sliding_window_gqa_qknorm")
class SlidingWindowGroupedQueryQKNormAttention(GroupedQueryQKNormAttention):
    """Gemma 3 local GQA layer: QK-Norm plus causal sliding-window attention."""

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
        self._record_qk_clip_logits(q, k)
        bias = _local_attention_bias(T, self.window_size, x.device, x.dtype, is_causal)
        bias = _merge_attention_bias(bias, attn_bias)
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


@register_attention("iha")
class InterleavedHeadAttention(nn.Module):
    """Cross-head mixing: each pseudo Q/K is a learned linear combination of all H original Q/K."""

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
        self.q_mix = nn.Parameter(torch.eye(num_heads))
        self.k_mix = nn.Parameter(torch.eye(num_heads))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim)

        q = torch.einsum("bthd,gh->btgd", q, self.q_mix)
        k = torch.einsum("bthd,gh->btgd", k, self.k_mix)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("sliding_window")
class SlidingWindowAttention(nn.Module):
    """Local attention over a fixed token window.

    In causal mode a query attends to the previous `window_size` tokens. In
    bidirectional mode it attends to a symmetric local band.
    """

    def __init__(self, dim, num_heads, dropout=0.0, window_size=128):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(window_size > 0, "window_size must be > 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)
        bias = _local_attention_bias(T, self.window_size, x.device, x.dtype, is_causal)
        bias = _merge_attention_bias(bias, attn_bias)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("block_sparse")
class BlockSparseAttention(nn.Module):
    """BigBird/Longformer-style block sparse attention pattern.

    This is a dense reference implementation: it applies the exact sparse mask
    pattern with local, global, and deterministic random block connections, then
    uses PyTorch SDPA. It is intended for correctness and experiments, not kernel
    speedups.
    """

    def __init__(
        self,
        dim,
        num_heads,
        dropout=0.0,
        block_size=16,
        local_blocks=1,
        global_tokens=1,
        random_blocks=2,
        seed=0,
    ):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(block_size > 0, "block_size must be > 0")
        require(local_blocks >= 0, "local_blocks must be >= 0")
        require(global_tokens >= 0, "global_tokens must be >= 0")
        require(random_blocks >= 0, "random_blocks must be >= 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.block_size = block_size
        self.local_blocks = local_blocks
        self.global_tokens = global_tokens
        self.random_blocks = random_blocks
        self.seed = seed
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)
        bias = _block_sparse_attention_bias(
            T, self.block_size, self.local_blocks, self.global_tokens,
            self.random_blocks, self.seed, x.device, x.dtype, is_causal,
        )
        bias = _merge_attention_bias(bias, attn_bias)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("cosformer")
class CosFormerAttention(nn.Module):
    """cosFormer linear attention with causal cumulative-sum evaluation."""

    def __init__(self, dim, num_heads, dropout=0.0, eps=1e-6):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(eps > 0, "eps must be > 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = eps
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        require(freqs_cis is None, "cosFormer owns positional reweighting; use position='none'")
        require(attn_bias is None, "cosFormer does not consume additive attention bias; use position='none'")
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = F.relu(q) + self.eps
        k = F.relu(k) + self.eps
        q_cos, q_sin, k_cos, k_sin = _cosformer_reweight(q, k)

        if is_causal:
            kv_cos = torch.einsum("bhtd,bhte->bhtde", k_cos, v).cumsum(dim=2)
            kv_sin = torch.einsum("bhtd,bhte->bhtde", k_sin, v).cumsum(dim=2)
            k_sum_cos = k_cos.cumsum(dim=2)
            k_sum_sin = k_sin.cumsum(dim=2)
            num = (
                torch.einsum("bhtd,bhtde->bhte", q_cos, kv_cos)
                + torch.einsum("bhtd,bhtde->bhte", q_sin, kv_sin)
            )
            denom = (q_cos * k_sum_cos).sum(dim=-1) + (q_sin * k_sum_sin).sum(dim=-1)
        else:
            kv_cos = torch.einsum("bhtd,bhte->bhde", k_cos, v)
            kv_sin = torch.einsum("bhtd,bhte->bhde", k_sin, v)
            k_sum_cos = k_cos.sum(dim=2)
            k_sum_sin = k_sin.sum(dim=2)
            num = (
                torch.einsum("bhtd,bhde->bhte", q_cos, kv_cos)
                + torch.einsum("bhtd,bhde->bhte", q_sin, kv_sin)
            )
            denom = (
                torch.einsum("bhtd,bhd->bht", q_cos, k_sum_cos)
                + torch.einsum("bhtd,bhd->bht", q_sin, k_sum_sin)
            )

        out = num / denom.clamp(min=self.eps).unsqueeze(-1)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("lightning")
class LightningAttention2(nn.Module):
    """Lightning Attention-2 reference implementation.

    This follows the paper's tiled causal linear attention: intra-block
    interactions are computed directly while inter-block information flows
    through a decayed KV state. It is intentionally a clear PyTorch path rather
    than the Triton kernel used for speed in the paper.
    """

    def __init__(self, dim, num_heads, dropout=0.0, block_size=64):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(block_size > 0, "block_size must be > 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block_size = block_size
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim, bias=False)
        self.logit_decay = nn.Parameter(torch.full((num_heads,), math.log(0.95 / 0.05)))

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        require(freqs_cis is None, "Lightning Attention owns positional decay; use position='none'")
        require(attn_bias is None, "Lightning Attention does not consume additive attention bias")
        require(is_causal, "Lightning Attention-2 reference path implements causal attention")
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q / math.sqrt(self.head_dim)
        decay = torch.sigmoid(self.logit_decay).to(dtype=q.dtype, device=q.device).clamp(1e-4, 1.0 - 1e-4)
        kv_state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=q.dtype)
        chunks = []
        for start in range(0, T, self.block_size):
            end = min(start + self.block_size, T)
            q_i = q[:, :, start:end]
            k_i = k[:, :, start:end]
            v_i = v[:, :, start:end]
            L = end - start
            row = torch.arange(L, device=x.device)
            rel = row[:, None] - row[None, :]
            mask = rel >= 0
            decay_mask = decay.view(1, self.num_heads, 1, 1).pow(rel.clamp(min=0)).to(q.dtype)
            intra = torch.matmul(q_i, k_i.transpose(-2, -1)) * decay_mask
            intra = intra.masked_fill(~mask.view(1, 1, L, L), 0.0)
            out_intra = torch.matmul(intra, v_i)

            powers = decay.view(1, self.num_heads, 1).pow(torch.arange(1, L + 1, device=x.device).view(1, 1, L))
            out_inter = torch.einsum("bhld,bhde->bhle", q_i * powers.unsqueeze(-1), kv_state)
            chunks.append(out_intra + out_inter)

            block_decay = decay.view(1, self.num_heads, 1, 1).pow(L)
            update_power = decay.view(1, self.num_heads, 1).pow((L - 1 - row).view(1, 1, L))
            kv_update = torch.einsum("bhld,bhle->bhde", k_i * update_power.unsqueeze(-1), v_i)
            kv_state = block_decay * kv_state + kv_update
        out = torch.cat(chunks, dim=2)
        out = self.norm(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("gated_deltanet")
class GatedDeltaNetAttention(nn.Module):
    """Qwen3-Next Gated DeltaNet token mixer, using the recurrent reference rule."""

    def __init__(self, dim, num_heads, dropout=0.0, conv_kernel_size=4):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(conv_kernel_size > 0, "conv_kernel_size must be > 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.in_proj_qkvz = nn.Linear(dim, 4 * dim, bias=False)
        self.in_proj_ba = nn.Linear(dim, 2 * num_heads, bias=False)
        self.conv1d = nn.Conv1d(3 * dim, 3 * dim, kernel_size=conv_kernel_size, groups=3 * dim, padding=conv_kernel_size - 1, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        self.A_log = nn.Parameter(torch.empty(num_heads).uniform_(1e-3, 16.0).log_())
        self.norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        require(freqs_cis is None, "Gated DeltaNet owns recurrent position dynamics; pass no RoPE")
        require(attn_bias is None, "Gated DeltaNet does not consume additive attention bias")
        require(is_causal, "Gated DeltaNet reference path implements causal recurrence")
        B, T, C = x.shape
        q, k, v, z = self.in_proj_qkvz(x).chunk(4, dim=-1)
        b, a = self.in_proj_ba(x).chunk(2, dim=-1)
        mixed = torch.cat([q, k, v], dim=-1).transpose(1, 2)
        mixed = F.silu(self.conv1d(mixed)[:, :, :T]).transpose(1, 2)
        q, k, v = mixed.split(C, dim=-1)
        q = _l2norm(q.reshape(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        k = _l2norm(k.reshape(B, T, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        z = z.reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        beta = torch.sigmoid(b).transpose(1, 2)
        g = -self.A_log.float().exp().view(1, self.num_heads, 1) * F.softplus(
            a.float().transpose(1, 2) + self.dt_bias.float().view(1, self.num_heads, 1)
        )
        q = q / math.sqrt(self.head_dim)
        state = torch.zeros(B, self.num_heads, self.head_dim, self.head_dim, device=x.device, dtype=torch.float32)
        outs = []
        for t in range(T):
            q_t = q[:, :, t].float()
            k_t = k[:, :, t].float()
            v_t = v[:, :, t].float()
            state = state * g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta[:, :, t].float().unsqueeze(-1)
            state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            outs.append((state * q_t.unsqueeze(-1)).sum(dim=-2))
        out = torch.stack(outs, dim=2).to(x.dtype)
        out = self.norm(out) * F.silu(z)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("mla")
class MultiHeadLatentAttention(_QKClipMixin, nn.Module):
    """DeepSeek-style Multi-head Latent Attention reference implementation.

    Query and key/value projections are low-rank. KV uses a shared latent plus a
    shared RoPE key component, matching the MLA factorization used for KV-cache
    compression; this training path expands KV for SDPA rather than using the
    inference-time absorb cache.
    """

    def __init__(self, dim, num_heads, dropout=0.0, q_lora_rank=None, kv_lora_rank=None, rope_head_dim=None):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.rope_head_dim = max(2, self.head_dim // 2) if rope_head_dim is None else rope_head_dim
        if self.rope_head_dim % 2 == 1:
            self.rope_head_dim -= 1
        require(0 < self.rope_head_dim <= self.head_dim, "rope_head_dim must be in (0, head_dim]")
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.q_lora_rank = max(1, dim // 2) if q_lora_rank is None else q_lora_rank
        self.kv_lora_rank = max(1, dim // 4) if kv_lora_rank is None else kv_lora_rank
        require(self.q_lora_rank > 0, "q_lora_rank must be > 0")
        require(self.kv_lora_rank > 0, "kv_lora_rank must be > 0")
        self.dropout = dropout
        self.wq_a = nn.Linear(dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.LayerNorm(self.q_lora_rank)
        self.wq_b = nn.Linear(self.q_lora_rank, num_heads * self.head_dim, bias=False)
        self.wkv_a = nn.Linear(dim, self.kv_lora_rank + self.rope_head_dim, bias=False)
        self.kv_norm = nn.LayerNorm(self.kv_lora_rank)
        self.wkv_b = nn.Linear(self.kv_lora_rank, num_heads * (self.nope_head_dim + self.head_dim), bias=False)
        self.out = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        self._init_qk_clip(num_heads)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x))).view(B, T, self.num_heads, self.head_dim)
        q_nope, q_pe = q.split([self.nope_head_dim, self.rope_head_dim], dim=-1)

        kv_pe_latent = self.wkv_a(x)
        kv_latent, k_pe = kv_pe_latent.split([self.kv_lora_rank, self.rope_head_dim], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv_latent)).view(
            B, T, self.num_heads, self.nope_head_dim + self.head_dim
        )
        k_nope, v = kv.split([self.nope_head_dim, self.head_dim], dim=-1)
        k_pe = k_pe.unsqueeze(2).expand(B, T, self.num_heads, self.rope_head_dim)

        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        q_pe = q_pe.transpose(1, 2)
        k_pe = k_pe.transpose(1, 2)
        v = v.transpose(1, 2)
        if freqs_cis is not None:
            q_pe, k_pe = _apply_partial_rotary(q_pe, k_pe, freqs_cis, self.rope_head_dim)
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        self._record_qk_clip_logits(q, k)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim))

    @torch.no_grad()
    def commit_qk_clip_update(self, threshold, balance=0.5):
        require(0.0 <= balance <= 1.0, "qk_clip balance must be in [0, 1]")
        gammas = self._qk_clip_gammas(threshold)
        if gammas is None:
            return
        sqrt_gamma = gammas.sqrt()
        q_weight = self.wq_b.weight.view(self.num_heads, self.head_dim, self.q_lora_rank)
        q_weight[:, :self.nope_head_dim].mul_(sqrt_gamma.to(q_weight.device, q_weight.dtype).view(-1, 1, 1))
        q_weight[:, self.nope_head_dim:].mul_(gammas.to(q_weight.device, q_weight.dtype).view(-1, 1, 1))

        kv_rows = self.nope_head_dim + self.head_dim
        kv_weight = self.wkv_b.weight.view(self.num_heads, kv_rows, self.kv_lora_rank)
        kv_weight[:, :self.nope_head_dim].mul_(sqrt_gamma.to(kv_weight.device, kv_weight.dtype).view(-1, 1, 1))
        self._reset_qk_clip_stats()


class _CompressedAttentionBase(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dropout=0.0,
        compress_ratio=4,
        window_size=128,
    ):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(compress_ratio > 1, "compress_ratio must be > 1")
        require(window_size > 0, "window_size must be > 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.compress_ratio = compress_ratio
        self.window_size = window_size
        self.q_rank = max(1, dim // 4)
        self.out_groups = _output_projection_groups(num_heads)
        self.group_heads = num_heads // self.out_groups
        self.group_dim = self.group_heads * self.head_dim
        self.group_out_dim = dim // self.out_groups
        self.q_down = nn.Linear(dim, self.q_rank, bias=False)
        self.q_up = nn.Linear(self.q_rank, num_heads * self.head_dim, bias=False)
        self.local_kv = nn.Linear(dim, self.head_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.kv_norm = RMSNorm(self.head_dim)
        self.sink_logits = nn.Parameter(torch.zeros(num_heads))
        self.group_out = nn.ModuleList([
            nn.Linear(self.group_dim, self.group_out_dim, bias=False)
            for _ in range(self.out_groups)
        ])
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        # GPT relative-position modules carry their own causal mask and pass
        # is_causal=False so SDPA does not receive two masks. Compressed key/value
        # slots do not have external relative-bias entries, so keep their structural
        # compressed-memory mask causal whenever such a position bias is present.
        structural_causal = is_causal or attn_bias is not None
        q_latent = self.q_down(x)
        q = self.q_up(q_latent).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        local_kv = self.kv_norm(self.local_kv(x)).unsqueeze(1).expand(B, self.num_heads, T, self.head_dim)
        positions = torch.arange(T, device=x.device)
        if freqs_cis is not None:
            q = _apply_tail_rotary(q, freqs_cis, positions)
            local_kv = _apply_tail_rotary(local_kv, freqs_cis, positions)
        raw_bias = _local_attention_bias(T, self.window_size, x.device, x.dtype, structural_causal)
        raw_scores = torch.matmul(q, local_kv.transpose(-2, -1)) / math.sqrt(self.head_dim)
        raw_scores = raw_scores + raw_bias

        comp, comp_start, comp_end = self._compress_kv(x)
        if comp is None:
            scores = raw_scores
            values = local_kv
        else:
            comp = self.kv_norm(comp)
            comp_kv = comp.unsqueeze(1).expand(B, self.num_heads, comp.size(1), self.head_dim)
            if freqs_cis is not None:
                comp_kv = _apply_tail_rotary(comp_kv, freqs_cis, comp_end)
            comp_scores = torch.matmul(q, comp_kv.transpose(-2, -1)) / math.sqrt(self.head_dim)
            comp_bias = _compressed_attention_bias(
                T, comp_start, comp_end, self.compress_ratio, x.device, x.dtype, structural_causal
            )
            comp_scores = comp_scores + comp_bias
            comp_scores = self._filter_compressed_scores(x, q_latent, comp_bias, comp_scores)
            scores = torch.cat([raw_scores, comp_scores], dim=-1)
            values = torch.cat([local_kv, comp_kv], dim=-2)

        if attn_bias is not None:
            scores[:, :, :, :T] = scores[:, :, :, :T] + _expand_external_bias(attn_bias, B, self.num_heads, T)
        scores, values = self._append_attention_sink(scores, values)
        probs = F.softmax(scores, dim=-1)
        probs = F.dropout(probs, p=self.dropout, training=self.training)
        out = torch.matmul(probs, values)
        if freqs_cis is not None:
            out = _apply_tail_rotary(out, freqs_cis, positions, inverse=True)
        return self.out(self._project_grouped_outputs(out))

    def _compress_kv(self, x):
        raise NotImplementedError

    def _filter_compressed_scores(self, x, q_latent, comp_bias, comp_scores):
        return comp_scores

    def _append_attention_sink(self, scores, values):
        B, _, T, _ = scores.shape
        sink_scores = self.sink_logits.view(1, self.num_heads, 1, 1).expand(B, -1, T, -1)
        sink_values = torch.zeros(B, self.num_heads, 1, self.head_dim, device=values.device, dtype=values.dtype)
        return torch.cat([scores, sink_scores], dim=-1), torch.cat([values, sink_values], dim=-2)

    def _project_grouped_outputs(self, out):
        B, _, T, _ = out.shape
        out = out.transpose(1, 2).reshape(B, T, self.out_groups, self.group_dim)
        projected = [
            proj(out[:, :, group])
            for group, proj in enumerate(self.group_out)
        ]
        return torch.cat(projected, dim=-1)


def _compress_hca(entries, weights, pos_bias, compress_ratio):
    B, T, D = entries.shape
    if T < compress_ratio:
        return None, None, None
    G = math.ceil(T / compress_ratio)
    entries, weights = _pad_compressor_inputs(entries, weights, compress_ratio)
    grouped_entries = entries.view(B, G, compress_ratio, D)
    grouped_weights = weights.view(B, G, compress_ratio, D) + pos_bias.view(1, 1, compress_ratio, D)
    compressor = F.softmax(grouped_weights, dim=2)
    comp = (compressor * grouped_entries).sum(dim=2)
    comp_start = torch.arange(G, device=entries.device) * compress_ratio
    comp_end = torch.arange(1, G + 1, device=entries.device) * compress_ratio - 1
    comp_end = comp_end.clamp(max=T - 1)
    return comp, comp_start, comp_end


def _compress_csa(entries_a, entries_b, weights_a, weights_b, pos_bias_a, pos_bias_b, compress_ratio):
    B, T, D = entries_a.shape
    if T < compress_ratio:
        return None, None, None
    G = math.ceil(T / compress_ratio)
    entries_a, weights_a = _pad_compressor_inputs(entries_a, weights_a, compress_ratio)
    entries_b, weights_b = _pad_compressor_inputs(entries_b, weights_b, compress_ratio)
    entries_a = entries_a.view(B, G, compress_ratio, D)
    weights_a = weights_a.view(B, G, compress_ratio, D)
    entries_b = entries_b.view(B, G, compress_ratio, D)
    weights_b = weights_b.view(B, G, compress_ratio, D)
    zero_entries = torch.zeros_like(entries_b[:, :1])
    neg_weights = torch.full_like(weights_b[:, :1], float("-inf"))
    entries_b_prev = torch.cat([zero_entries, entries_b[:, :-1]], dim=1)
    weights_b_prev = torch.cat([neg_weights, weights_b[:, :-1]], dim=1)
    weights = torch.cat([
        weights_a + pos_bias_a.view(1, 1, compress_ratio, D),
        weights_b_prev + pos_bias_b.view(1, 1, compress_ratio, D),
    ], dim=2)
    entries = torch.cat([entries_a, entries_b_prev], dim=2)
    compressor = F.softmax(weights, dim=2)
    comp = (compressor * entries).sum(dim=2)
    comp_start = torch.arange(G, device=entries_a.device) * compress_ratio
    comp_end = torch.arange(1, G + 1, device=entries_a.device) * compress_ratio - 1
    comp_end = comp_end.clamp(max=T - 1)
    return comp, comp_start, comp_end


@register_attention("csa")
class CompressedSparseAttention(_CompressedAttentionBase):
    """DeepSeek-V4-inspired compressed sparse attention reference path."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(dim, num_heads, dropout, compress_ratio=4, window_size=128)
        self.topk_compressed = 512
        self.indexer_heads = min(8, num_heads)
        self.indexer_dim = self.head_dim
        self.ca_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.cb_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.za_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.zb_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.pos_bias_a = nn.Parameter(torch.zeros(self.compress_ratio, self.head_dim))
        self.pos_bias_b = nn.Parameter(torch.zeros(self.compress_ratio, self.head_dim))
        self.index_ca_proj = nn.Linear(dim, self.indexer_dim, bias=False)
        self.index_cb_proj = nn.Linear(dim, self.indexer_dim, bias=False)
        self.index_za_proj = nn.Linear(dim, self.indexer_dim, bias=False)
        self.index_zb_proj = nn.Linear(dim, self.indexer_dim, bias=False)
        self.index_pos_bias_a = nn.Parameter(torch.zeros(self.compress_ratio, self.indexer_dim))
        self.index_pos_bias_b = nn.Parameter(torch.zeros(self.compress_ratio, self.indexer_dim))
        self.index_q_up = nn.Linear(self.q_rank, self.indexer_heads * self.indexer_dim, bias=False)
        self.index_weight = nn.Linear(dim, self.indexer_heads, bias=False)

    def _compress_kv(self, x):
        return _compress_csa(
            self.ca_proj(x), self.cb_proj(x), self.za_proj(x), self.zb_proj(x),
            self.pos_bias_a, self.pos_bias_b, self.compress_ratio,
        )

    def _filter_compressed_scores(self, x, q_latent, comp_bias, comp_scores):
        if comp_scores.size(-1) <= self.topk_compressed:
            return comp_scores
        keep = self._topk_mask(x, q_latent, comp_bias)
        return comp_scores.masked_fill(~keep.unsqueeze(1), float("-inf"))

    def _topk_mask(self, x, q_latent, comp_bias):
        index_comp, _, _ = _compress_csa(
            self.index_ca_proj(x), self.index_cb_proj(x),
            self.index_za_proj(x), self.index_zb_proj(x),
            self.index_pos_bias_a, self.index_pos_bias_b, self.compress_ratio,
        )
        B, T, _ = x.shape
        q_index = self.index_q_up(q_latent).view(B, T, self.indexer_heads, self.indexer_dim)
        w_index = self.index_weight(x).view(B, T, self.indexer_heads)
        index_scores = torch.einsum("bthd,bgd->bthg", q_index, index_comp).relu()
        index_scores = (index_scores * w_index.unsqueeze(-1)).sum(dim=2)
        allowed = torch.isfinite(comp_bias).unsqueeze(0).expand(B, T, index_comp.size(1))
        masked_scores = index_scores.masked_fill(~allowed, float("-inf"))
        topk = min(self.topk_compressed, index_comp.size(1))
        values, indices = masked_scores.topk(topk, dim=-1)
        keep = torch.zeros_like(masked_scores, dtype=torch.bool)
        keep.scatter_(-1, indices, torch.isfinite(values))
        return keep


@register_attention("hca")
class HeavilyCompressedAttention(_CompressedAttentionBase):
    """DeepSeek-V4-inspired heavily compressed dense attention reference path."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(dim, num_heads, dropout, compress_ratio=128, window_size=128)
        self.c_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.z_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.pos_bias = nn.Parameter(torch.zeros(self.compress_ratio, self.head_dim))

    def _compress_kv(self, x):
        return _compress_hca(self.c_proj(x), self.z_proj(x), self.pos_bias, self.compress_ratio)


def _local_attention_bias(T, window_size, device, dtype, is_causal):
    idx = torch.arange(T, device=device)
    delta = idx[:, None] - idx[None, :]
    if is_causal:
        allowed = (delta >= 0) & (delta < window_size)
    else:
        allowed = delta.abs() < window_size
    return _bool_to_additive_bias(allowed, dtype)


def _block_sparse_attention_bias(T, block_size, local_blocks, global_tokens, random_blocks, seed, device, dtype, is_causal):
    num_blocks = math.ceil(T / block_size)
    allowed_blocks = torch.zeros(num_blocks, num_blocks, device=device, dtype=torch.bool)
    for q_block in range(num_blocks):
        lo = max(0, q_block - local_blocks)
        hi = min(num_blocks, q_block + local_blocks + 1)
        if is_causal:
            hi = min(hi, q_block + 1)
        allowed_blocks[q_block, lo:hi] = True
        if random_blocks > 0:
            candidates = torch.arange(num_blocks, device=device)
            if is_causal:
                candidates = candidates[candidates <= q_block]
            if candidates.numel() > 0:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed + q_block)
                perm = torch.randperm(candidates.numel(), generator=gen, device="cpu")[:random_blocks].to(device)
                allowed_blocks[q_block, candidates[perm]] = True
    allowed = allowed_blocks.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)[:T, :T]
    if global_tokens > 0:
        g = min(global_tokens, T)
        allowed[:g, :] = True
        allowed[:, :g] = True
    if is_causal:
        causal = torch.arange(T, device=device)[:, None] >= torch.arange(T, device=device)[None, :]
        allowed &= causal
    return _bool_to_additive_bias(allowed, dtype)


def _cosformer_reweight(q, k):
    T = q.size(2)
    angle = torch.arange(T, device=q.device, dtype=q.dtype) * (math.pi / (2 * max(1, T)))
    angle = angle.view(1, 1, T, 1)
    cos = angle.cos()
    sin = angle.sin()
    return q * cos, q * sin, k * cos, k * sin


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x.float() * x.float()).sum(dim=dim, keepdim=True) + eps).to(x.dtype)


def _pad_compressor_inputs(entries, weights, ratio):
    T = entries.size(1)
    pad = math.ceil(T / ratio) * ratio - T
    if pad == 0:
        return entries, weights
    entries = F.pad(entries, (0, 0, 0, pad))
    weights = F.pad(weights, (0, 0, 0, pad), value=float("-inf"))
    return entries, weights


def _compressed_attention_bias(T, comp_start, comp_end, compress_ratio, device, dtype, is_causal):
    query_pos = torch.arange(T, device=device).unsqueeze(-1)
    comp_end = comp_end.view(1, -1)
    if is_causal:
        current_block_start = torch.div(query_pos, compress_ratio, rounding_mode="floor") * compress_ratio
        allowed = comp_end < current_block_start
    else:
        allowed = torch.ones(T, comp_start.numel(), device=device, dtype=torch.bool)
    return _bool_to_additive_bias(allowed, dtype)


def _bool_to_additive_bias(allowed, dtype):
    bias = torch.zeros(allowed.shape, device=allowed.device, dtype=dtype)
    return bias.masked_fill(~allowed, float("-inf"))


def _output_projection_groups(num_heads):
    for groups in (4, 2):
        if num_heads % groups == 0:
            return groups
    return 1


def _merge_attention_bias(base_bias, extra_bias):
    if extra_bias is None:
        return base_bias
    return base_bias + extra_bias


def _expand_external_bias(attn_bias, batch_size, num_heads, T):
    if attn_bias.dim() == 2:
        return attn_bias.view(1, 1, T, T)
    if attn_bias.dim() == 3:
        if attn_bias.size(0) == num_heads:
            return attn_bias.view(1, num_heads, T, T)
        require(attn_bias.size(0) == batch_size, (
            "3D attn_bias must be shaped as (num_heads, T, T) or (batch, T, T)"
        ))
        return attn_bias.view(batch_size, 1, T, T)
    if attn_bias.dim() == 4:
        require(attn_bias.size(-2) == T and attn_bias.size(-1) == T, (
            "4D attn_bias must end with shape (T, T)"
        ))
        return attn_bias
    raise ValueError("attn_bias must have 2, 3, or 4 dimensions")


def _apply_partial_rotary(q, k, freqs_cis, rope_dim):
    cos, sin = freqs_cis
    half = rope_dim // 2
    return apply_rotary_emb(q, k, cos[:, :half], sin[:, :half])


def _apply_tail_rotary(x, freqs_cis, positions, rope_dim=64, inverse=False):
    cos, sin = freqs_cis
    dim = min(x.size(-1), rope_dim, cos.size(-1) * 2)
    if dim % 2 == 1:
        dim -= 1
    if dim <= 0:
        return x
    half = dim // 2
    positions = positions.to(device=cos.device, dtype=torch.long)
    cos = cos.index_select(0, positions)[:, :half].to(device=x.device, dtype=x.dtype)
    sin = sin.index_select(0, positions)[:, :half].to(device=x.device, dtype=x.dtype)
    if inverse:
        sin = -sin
    view_shape = (1,) * (x.dim() - 2) + (positions.numel(), dim)
    cos = torch.cat([cos, cos], dim=-1).view(view_shape)
    sin = torch.cat([sin, sin], dim=-1).view(view_shape)
    plain, rotary = x[..., :-dim], x[..., -dim:]
    rotated = rotary * cos + rotate_half(rotary) * sin
    return torch.cat([plain, rotated], dim=-1)
