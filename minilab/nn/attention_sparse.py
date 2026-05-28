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


def _learned_block_attention_bias(q_idx, k_idx, block_size, top_k_blocks, local_blocks, kv_group_size, dtype, is_causal):
    B, G, T, index_dim = q_idx.shape
    require(k_idx.shape[:2] == (B, T), "index keys must match batch and sequence length")
    require(k_idx.size(-1) == index_dim, "index query/key dims must match")
    num_blocks = math.ceil(T / block_size)

    with torch.no_grad():
        scores = torch.einsum("bgtd,bsd->bgts", q_idx.float(), k_idx.float()) / math.sqrt(index_dim)
        if is_causal:
            pos = torch.arange(T, device=q_idx.device)
            future = pos[None, :] > pos[:, None]
            scores = scores.masked_fill(future.view(1, 1, T, T), float("-inf"))

        pad = num_blocks * block_size - T
        if pad > 0:
            scores = F.pad(scores, (0, pad), value=float("-inf"))
        block_scores = scores.view(B, G, T, num_blocks, block_size).amax(dim=-1)

        keep_blocks = torch.zeros(B, G, T, num_blocks, device=q_idx.device, dtype=torch.bool)
        if top_k_blocks > 0:
            k_eff = min(top_k_blocks, num_blocks)
            values, indices = block_scores.topk(k_eff, dim=-1)
            keep_blocks.scatter_(-1, indices, torch.isfinite(values))

        q_blocks = torch.div(torch.arange(T, device=q_idx.device), block_size, rounding_mode="floor")
        blocks = torch.arange(num_blocks, device=q_idx.device)
        delta = q_blocks[:, None] - blocks[None, :]
        if is_causal:
            local = (delta >= 0) & (delta <= local_blocks)
        else:
            local = delta.abs() <= local_blocks
        keep_blocks |= local.view(1, 1, T, num_blocks)

        allowed = keep_blocks.repeat_interleave(block_size, dim=-1)[..., :T]
        if is_causal:
            pos = torch.arange(T, device=q_idx.device)
            allowed &= (pos[None, :] <= pos[:, None]).view(1, 1, T, T)

    return _bool_to_additive_bias(allowed.repeat_interleave(kv_group_size, dim=1), dtype)


def _lighthouse_metadata(T, num_levels, pooling_factor, device):
    levels, starts, sizes = [], [], []
    for level in range(num_levels):
        size = pooling_factor ** level
        count = math.ceil(T / size)
        start = torch.arange(count, device=device, dtype=torch.long) * size
        levels.append(torch.full((count,), level, device=device, dtype=torch.long))
        starts.append(start)
        sizes.append((T - start).clamp(max=size))
    levels = torch.cat(levels)
    starts = torch.cat(starts)
    sizes = torch.cat(sizes)
    return levels, starts, sizes, starts + sizes - 1


def _lighthouse_pyramid(x, num_levels, pooling_factor):
    B, H, T, D = x.shape
    pooled = []
    for level in range(num_levels):
        size = pooling_factor ** level
        count = math.ceil(T / size)
        pad = count * size - T
        level_x = F.pad(x, (0, 0, 0, pad)) if pad else x
        summed = level_x.reshape(B, H, count, size, D).sum(dim=3)
        actual = torch.full((count,), size, device=x.device, dtype=x.dtype)
        if pad:
            actual[-1] = size - pad
        pooled.append(summed / actual.view(1, 1, count, 1))
    return torch.cat(pooled, dim=2)


def _lighthouse_scores(q, k, num_levels, pooling_factor):
    B, H, T, _ = q.shape
    base = torch.maximum(q.norm(dim=-1), k.norm(dim=-1)).detach()
    scores = []
    for level in range(num_levels):
        size = pooling_factor ** level
        count = math.ceil(T / size)
        pad = count * size - T
        level_base = F.pad(base, (0, pad), value=float("-inf")) if pad else base
        scores.append(level_base.reshape(B, H, count, size).amax(dim=-1))
    return torch.cat(scores, dim=2)


def _lighthouse_scatter_range(start, size, T):
    lo = start + size - 1
    hi = min(T, start + 2 * size - 1)
    return lo, hi


def _lighthouse_select_indices(scores, levels, starts, sizes, ends, num_levels, top_k, T):
    level0 = (levels == 0).nonzero(as_tuple=False).squeeze(1)
    coarsest = (levels == num_levels - 1).nonzero(as_tuple=False).squeeze(1)
    candidates = (levels != num_levels - 1).nonzero(as_tuple=False).squeeze(1)
    selected_indices = coarsest.tolist()
    if candidates.numel() > 0 and top_k > 0:
        candidate_order = torch.argsort(
            ends.index_select(0, candidates) * num_levels
            + levels.index_select(0, candidates)
        )
        prefix_top_scores = []
        for idx in candidates.index_select(0, candidate_order).tolist():
            # Admit entries when they enter the prefix top-k at their own end.
            # A later token can add a route, but cannot evict an earlier route.
            score = float(scores[idx].item())
            if len(prefix_top_scores) < top_k:
                prefix_top_scores.append(score)
                selected_indices.append(idx)
                continue
            weakest = min(range(len(prefix_top_scores)), key=lambda i: prefix_top_scores[i])
            if score > prefix_top_scores[weakest]:
                prefix_top_scores[weakest] = score
                selected_indices.append(idx)

    coverage = torch.zeros(T, device=scores.device, dtype=torch.bool)
    selected = torch.tensor(selected_indices, device=scores.device, dtype=torch.long)
    for idx in selected.tolist():
        lo, hi = _lighthouse_scatter_range(int(starts[idx].item()), int(sizes[idx].item()), T)
        if lo < hi:
            coverage[lo:hi] = True
    holes = (~coverage).nonzero(as_tuple=False).squeeze(1)
    if holes.numel() > 0:
        # Level-0 entries preserve dense causal outputs where shifted coarse
        # scatter ranges leave prefix positions uncovered.
        selected = torch.cat([selected, level0.index_select(0, holes)])

    selected = torch.unique(selected)
    order = torch.argsort(ends.index_select(0, selected) * num_levels + levels.index_select(0, selected))
    return selected.index_select(0, order)


@register_attention("lighthouse_mha")
class LighthouseAttention(nn.Module):
    """Training-time Lighthouse-style MHA reference path.

    This keeps the paper's dataflow visible in PyTorch: symmetric Q/K/V
    pyramids, parameter-free norm selection, dense SDPA over selected entries,
    and causal shifted scatter-back. It is not an optimized long-context kernel.
    """

    def __init__(self, dim, num_heads, dropout=0.0, num_levels=3, pooling_factor=2, top_k=32):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(num_levels > 0, "num_levels must be > 0")
        require(pooling_factor > 1, "pooling_factor must be > 1")
        require(top_k > 0, "top_k must be > 0")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.num_levels = num_levels
        self.pooling_factor = pooling_factor
        self.top_k = top_k
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        require(attn_bias is None, "Lighthouse attention does not support external attention bias")
        require(is_causal, "Lighthouse attention is a causal training-time attention path")
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        levels, starts, sizes, ends = _lighthouse_metadata(T, self.num_levels, self.pooling_factor, x.device)
        q_entries = _lighthouse_pyramid(q, self.num_levels, self.pooling_factor)
        k_entries = _lighthouse_pyramid(k, self.num_levels, self.pooling_factor)
        v_entries = _lighthouse_pyramid(v, self.num_levels, self.pooling_factor)
        scores = _lighthouse_scores(q, k, self.num_levels, self.pooling_factor)

        batch_outputs = []
        for b in range(B):
            head_outputs = []
            for h in range(self.num_heads):
                selected = _lighthouse_select_indices(
                    scores[b, h],
                    levels,
                    starts,
                    sizes,
                    ends,
                    self.num_levels,
                    self.top_k,
                    T,
                )
                q_sel = q_entries[b, h].index_select(0, selected)
                k_sel = k_entries[b, h].index_select(0, selected)
                v_sel = v_entries[b, h].index_select(0, selected)
                selected_ends = ends.index_select(0, selected)
                allowed = selected_ends.view(-1, 1) >= selected_ends.view(1, -1)
                bias = _bool_to_additive_bias(allowed, x.dtype)
                attended = F.scaled_dot_product_attention(
                    q_sel.view(1, 1, q_sel.size(0), self.head_dim),
                    k_sel.view(1, 1, k_sel.size(0), self.head_dim),
                    v_sel.view(1, 1, v_sel.size(0), self.head_dim),
                    attn_mask=bias,
                    dropout_p=self.dropout if self.training else 0.0,
                    is_causal=False,
                ).view(q_sel.size(0), self.head_dim)
                head_outputs.append(self._scatter_selected(attended, selected, starts, sizes, T))
            batch_outputs.append(torch.stack(head_outputs, dim=0))

        out = torch.stack(batch_outputs, dim=0)
        return self.out(out.transpose(1, 2).reshape(B, T, C))

    def _scatter_selected(self, attended, selected, starts, sizes, T):
        out = attended.new_zeros(T, self.head_dim)
        for row, idx in enumerate(selected.tolist()):
            lo, hi = _lighthouse_scatter_range(int(starts[idx].item()), int(sizes[idx].item()), T)
            if lo < hi:
                positions = torch.arange(lo, hi, device=attended.device)
                out.index_add_(0, positions, attended[row].expand(hi - lo, -1))
        return out


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


@register_attention("learned_block_gqa")
class LearnedBlockSparseGQAAttention(nn.Module):
    """Learned block-routed GQA reference path.

    A small index branch picks top-k KV blocks per GQA group, local blocks are
    always included, and dense SDPA applies the resulting exact token mask. This
    is for correctness experiments, not memory or kernel speedups.
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        dropout=0.0,
        block_size=128,
        top_k_blocks=32,
        local_blocks=1,
        index_dim=None,
    ):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(num_kv_heads > 0, "num_kv_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(block_size > 0, "block_size must be > 0")
        require(top_k_blocks >= 0, "top_k_blocks must be >= 0")
        require(local_blocks >= 0, "local_blocks must be >= 0")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_group_size = num_heads // num_kv_heads
        self.dropout = dropout
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.local_blocks = local_blocks
        self.index_dim = self.head_dim if index_dim is None else index_dim
        require(self.index_dim > 0, "index_dim must be > 0")

        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.q_idx_proj = nn.Linear(dim, num_kv_heads * self.index_dim, bias=False)
        self.k_idx_proj = nn.Linear(dim, self.index_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q_idx = self.q_idx_proj(x).reshape(B, T, self.num_kv_heads, self.index_dim).transpose(1, 2)
        k_idx = self.k_idx_proj(x)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        k_attn = k.repeat_interleave(self.kv_group_size, dim=1)
        v_attn = v.repeat_interleave(self.kv_group_size, dim=1)
        bias = _learned_block_attention_bias(
            q_idx,
            k_idx,
            self.block_size,
            self.top_k_blocks,
            self.local_blocks,
            self.kv_group_size,
            x.dtype,
            is_causal,
        )
        bias = _merge_attention_bias(bias, attn_bias)
        out = F.scaled_dot_product_attention(
            q,
            k_attn,
            v_attn,
            attn_mask=bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
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
