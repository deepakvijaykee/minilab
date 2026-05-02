import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.attention_common import (
    _bool_to_additive_bias,
    _local_attention_bias,
    _merge_attention_bias,
    apply_rotary_emb,
)
from minilab.registry import register_attention


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

