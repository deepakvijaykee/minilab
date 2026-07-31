import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.attention_common import (
    _attention_support_from_bias,
    _bool_to_additive_bias,
    _local_attention_bias,
    _merge_attention_bias,
    DEFAULT_ATTENTION_BACKEND,
    attention_sdpa,
    apply_rotary_emb,
)
from minilab.registry import register_attention


MSA_SPARSE_INDEX_PAD = -1


class SharedSparseIndexState:
    """Per-forward exchange of learned sparse index decisions across layers.

    IndexShare-style sharing (GLM-5.2 / IndexCache): owner layers publish their
    index scores and token support each forward, and the shared layers that
    follow them reuse those decisions instead of running their own indexer. The
    model clears this state at the start of every forward so a shared layer can
    never silently consume indices from a previous batch.
    """

    def __init__(self):
        self._entries = {}

    def clear(self):
        self._entries = {}

    def publish(self, key, index_scores, token_support):
        self._entries[key] = (index_scores, token_support)

    def read(self, key, batch_size, seq_len):
        require(key in self._entries, (
            "shared sparse index attention requires its owner layer to run earlier in the same forward"
        ))
        index_scores, token_support = self._entries[key]
        require(index_scores.size(0) == batch_size and index_scores.size(2) == seq_len, (
            "shared sparse index state does not match the current forward shape"
        ))
        return index_scores, token_support


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


def msa_sparse_topk_select(max_score, top_k, num_valid_pages=None, force_begin_blocks=0, force_end_blocks=0):
    """PyTorch reference for MSA's sparse_topk_select contract.

    `max_score` is shaped `(heads, kv_blocks, query_tokens)`. The returned
    block indexes are shaped `(query_tokens, heads, top_k)`, sorted ascending by
    block id and padded with `-1` when fewer valid blocks exist.
    """
    require(max_score.dim() == 3, "max_score must have shape (heads, kv_blocks, query_tokens)")
    require(top_k > 0, "top_k must be > 0")
    require(force_begin_blocks >= 0, "force_begin_blocks must be >= 0")
    require(force_end_blocks >= 0, "force_end_blocks must be >= 0")
    num_heads, max_k_tiles, total_q = max_score.shape
    if num_valid_pages is None:
        num_valid_pages = max_k_tiles
    require(0 <= num_valid_pages <= max_k_tiles, "num_valid_pages must be in [0, kv_blocks]")
    if num_valid_pages == 0:
        return torch.full((total_q, num_heads, top_k), MSA_SPARSE_INDEX_PAD, device=max_score.device, dtype=torch.int32)
    if num_valid_pages <= top_k:
        selected = torch.full(
            (total_q, num_heads, top_k),
            MSA_SPARSE_INDEX_PAD,
            device=max_score.device,
            dtype=torch.int32,
        )
        selected[..., :num_valid_pages] = torch.arange(
            num_valid_pages,
            device=max_score.device,
            dtype=torch.int32,
        ).view(1, 1, num_valid_pages)
        return selected

    force_begin_blocks = min(force_begin_blocks, num_valid_pages)
    force_end_blocks = min(force_end_blocks, num_valid_pages)
    force_mask = torch.zeros(num_valid_pages, device=max_score.device, dtype=torch.bool)
    if force_begin_blocks:
        force_mask[:force_begin_blocks] = True
    if force_end_blocks:
        force_mask[num_valid_pages - force_end_blocks:] = True
    forced = torch.arange(num_valid_pages, device=max_score.device, dtype=torch.long)[force_mask]
    require(forced.numel() <= top_k, "forced sparse blocks must fit within top_k")

    scores = max_score[:, :num_valid_pages, :].permute(2, 0, 1)
    remaining = top_k - forced.numel()
    pieces = []
    if forced.numel() > 0:
        pieces.append(forced.view(1, 1, -1).expand(total_q, num_heads, -1))
    if remaining > 0:
        candidate_scores = scores.masked_fill(force_mask.view(1, 1, -1), float("-inf"))
        k_eff = min(remaining, num_valid_pages - forced.numel())
        if k_eff > 0:
            values, indices = candidate_scores.topk(k_eff, dim=-1)
            indices = torch.where(
                torch.isfinite(values),
                indices,
                torch.full_like(indices, MSA_SPARSE_INDEX_PAD),
            )
            pieces.append(indices)

    if pieces:
        selected = torch.cat(pieces, dim=-1)
    else:
        selected = torch.empty(total_q, num_heads, 0, device=max_score.device, dtype=torch.long)
    if selected.size(-1) < top_k:
        pad = torch.full(
            (total_q, num_heads, top_k - selected.size(-1)),
            MSA_SPARSE_INDEX_PAD,
            device=max_score.device,
            dtype=selected.dtype,
        )
        selected = torch.cat([selected, pad], dim=-1)

    sentinel = torch.full_like(selected, num_valid_pages)
    order = torch.argsort(torch.where(selected != MSA_SPARSE_INDEX_PAD, selected, sentinel), dim=-1)
    selected = torch.gather(selected, -1, order)
    return selected.to(torch.int32)


def _msa_token_scores(q_idx, k_idx, is_causal):
    B, G, T, index_dim = q_idx.shape
    require(k_idx.shape[:2] == (B, T), "index keys must match batch and sequence length")
    require(k_idx.size(-1) == index_dim, "index query/key dims must match")

    scores = torch.einsum("bgtd,bsd->bgts", q_idx.float(), k_idx.float()) / math.sqrt(index_dim)
    if is_causal:
        pos = torch.arange(T, device=q_idx.device)
        future = pos[None, :] > pos[:, None]
        scores = scores.masked_fill(future.view(1, 1, T, T), float("-inf"))
    return scores


def _msa_block_max_scores_from_token_scores(token_scores, block_size):
    T = token_scores.size(-1)
    num_blocks = math.ceil(T / block_size)

    pad = num_blocks * block_size - T
    if pad > 0:
        token_scores = F.pad(token_scores, (0, pad), value=float("-inf"))
    return token_scores.view(*token_scores.shape[:-1], num_blocks, block_size).amax(dim=-1)


def _msa_block_max_scores(q_idx, k_idx, block_size, is_causal):
    return _msa_block_max_scores_from_token_scores(_msa_token_scores(q_idx, k_idx, is_causal), block_size)


def _msa_block_indexes_from_scores(block_scores, top_k_blocks):
    B, G, T, num_blocks = block_scores.shape
    rows = []
    for b in range(B):
        rows.append(msa_sparse_topk_select(block_scores[b].permute(0, 2, 1), top_k_blocks, num_blocks))
    return torch.stack(rows, dim=0).permute(0, 2, 1, 3).contiguous()


def _msa_block_indexes_with_local(block_scores, block_size, top_k_blocks):
    B, G, T, num_blocks = block_scores.shape
    if top_k_blocks >= num_blocks:
        return _msa_block_indexes_from_scores(block_scores, top_k_blocks)

    local = torch.div(
        torch.arange(T, device=block_scores.device),
        block_size,
        rounding_mode="floor",
    ).view(1, 1, T, 1).expand(B, G, T, 1)
    if top_k_blocks == 1:
        return local.to(torch.int32).contiguous()

    dynamic_scores = block_scores.clone()
    dynamic_scores.scatter_(-1, local.long(), float("-inf"))
    dynamic = _msa_block_indexes_from_scores(dynamic_scores, top_k_blocks - 1)
    selected = torch.cat([local.to(dynamic.dtype), dynamic], dim=-1)
    sentinel = torch.full_like(selected, num_blocks)
    order = torch.argsort(torch.where(selected != MSA_SPARSE_INDEX_PAD, selected, sentinel), dim=-1)
    return torch.gather(selected, -1, order).to(torch.int32).contiguous()


def _token_support_from_block_indexes(block_indexes, block_size, local_blocks, is_causal):
    B, G, T, _ = block_indexes.shape
    num_blocks = math.ceil(T / block_size)
    key_blocks = torch.div(torch.arange(T, device=block_indexes.device), block_size, rounding_mode="floor")
    allowed = (block_indexes.unsqueeze(-1).long() == key_blocks.view(1, 1, 1, 1, T)).any(dim=-2)

    if local_blocks > 0:
        blocks = torch.arange(num_blocks, device=block_indexes.device)
        keep_blocks = (block_indexes.unsqueeze(-1).long() == blocks.view(1, 1, 1, 1, num_blocks)).any(dim=-2)
        q_blocks = torch.div(torch.arange(T, device=block_indexes.device), block_size, rounding_mode="floor")
        delta = q_blocks[:, None] - blocks[None, :]
        if is_causal:
            local = (delta >= 0) & (delta <= local_blocks)
        else:
            local = delta.abs() <= local_blocks
        keep_blocks |= local.view(1, 1, T, num_blocks)
        allowed = keep_blocks.repeat_interleave(block_size, dim=-1)[..., :T]

    if is_causal:
        pos = torch.arange(T, device=block_indexes.device)
        allowed &= (pos[None, :] <= pos[:, None]).view(1, 1, T, T)
    return allowed


def _learned_block_attention_support_from_scores(block_scores, block_size, top_k_blocks, local_blocks, is_causal):
    with torch.no_grad():
        block_indexes = _msa_block_indexes_with_local(block_scores.detach(), block_size, top_k_blocks)
        return _token_support_from_block_indexes(block_indexes, block_size, local_blocks, is_causal)


def _learned_block_attention_bias_from_scores(block_scores, block_size, top_k_blocks, local_blocks, kv_group_size, dtype, is_causal):
    support = _learned_block_attention_support_from_scores(
        block_scores,
        block_size,
        top_k_blocks,
        local_blocks,
        is_causal,
    )
    return _bool_to_additive_bias(support.repeat_interleave(kv_group_size, dim=1), dtype)


def _learned_block_attention_bias(q_idx, k_idx, block_size, top_k_blocks, local_blocks, kv_group_size, dtype, is_causal):
    block_scores = _msa_block_max_scores(q_idx, k_idx, block_size, is_causal)
    return _learned_block_attention_bias_from_scores(
        block_scores,
        block_size,
        top_k_blocks,
        local_blocks,
        kv_group_size,
        dtype,
        is_causal,
    )


def _full_attention_support(batch_size, num_kv_heads, seq_len, device, is_causal):
    support = torch.ones(batch_size, num_kv_heads, seq_len, seq_len, device=device, dtype=torch.bool)
    if is_causal:
        pos = torch.arange(seq_len, device=device)
        support &= (pos[None, :] <= pos[:, None]).view(1, 1, seq_len, seq_len)
    return support


def _broadcast_additive_attention_bias(attn_bias, batch_size, num_heads, q_len, kv_len, device):
    if attn_bias is None or attn_bias.dtype == torch.bool:
        return None
    bias = attn_bias.to(device=device, dtype=torch.float32)
    if bias.dim() == 2:
        require(bias.shape == (q_len, kv_len), "2D attn_bias must match query/key lengths")
        return bias.view(1, 1, q_len, kv_len)
    if bias.dim() == 3:
        require(bias.size(-2) == q_len and bias.size(-1) == kv_len, "3D attn_bias must match query/key lengths")
        if bias.size(0) == num_heads:
            return bias.view(1, num_heads, q_len, kv_len)
        require(bias.size(0) == batch_size, "3D attn_bias must be keyed by heads or batch")
        return bias.view(batch_size, 1, q_len, kv_len)
    if bias.dim() == 4:
        require(bias.size(-2) == q_len and bias.size(-1) == kv_len, "4D attn_bias must match query/key lengths")
        require(bias.size(0) in {1, batch_size}, "4D attn_bias batch dimension must be 1 or batch size")
        require(bias.size(1) in {1, num_heads}, "4D attn_bias head dimension must be 1 or num_heads")
        return bias
    raise ValueError("attn_bias must have 2, 3, or 4 dimensions")


def _safe_log_softmax(scores, support):
    fallback = torch.zeros_like(support)
    fallback[..., 0] = True
    safe_support = support | (~support.any(dim=-1, keepdim=True) & fallback)
    return F.log_softmax(scores.masked_fill(~safe_support, float("-inf")), dim=-1)


def _learned_block_kl_loss(index_scores, q, k_attn, token_support, kv_group_size, attn_bias, loss_weight):
    if loss_weight == 0.0 or not torch.is_grad_enabled():
        return index_scores.new_zeros(())
    B, G, T, _ = index_scores.shape
    H = q.size(1)
    head_support = token_support.repeat_interleave(kv_group_size, dim=1)
    additive_bias = _broadcast_additive_attention_bias(attn_bias, B, H, T, T, q.device)
    if attn_bias is not None:
        head_support = head_support & _attention_support_from_bias(attn_bias.to(q.device), B, H, T, T)

    with torch.no_grad():
        main_scores = torch.matmul(q.detach().float(), k_attn.detach().float().transpose(-2, -1)) / math.sqrt(q.size(-1))
        if additive_bias is not None:
            main_scores = main_scores + additive_bias
        main_log_probs = _safe_log_softmax(main_scores, head_support)
        teacher = main_log_probs.exp().view(B, G, kv_group_size, T, T).mean(dim=2)
        group_support = head_support.view(B, G, kv_group_size, T, T).any(dim=2)
        teacher = teacher * group_support
        teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(teacher.dtype).tiny)
        support = teacher > 0

    index_log_probs = _safe_log_softmax(index_scores.float(), support)
    teacher_log = teacher.clamp_min(torch.finfo(teacher.dtype).tiny).log()
    kl = (teacher * (teacher_log - index_log_probs)).masked_fill(~support, 0.0).sum(dim=-1)
    valid = support.any(dim=-1)
    if not bool(valid.any().item()):
        return index_scores.new_zeros(())
    return loss_weight * kl[valid].mean().to(index_scores.dtype)


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

    def __init__(
        self,
        dim,
        num_heads,
        dropout=0.0,
        num_levels=3,
        pooling_factor=2,
        top_k=32,
        attention_backend=DEFAULT_ATTENTION_BACKEND,
    ):
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
        self.attention_backend = attention_backend
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
                attended = attention_sdpa(
                    q_sel.view(1, 1, q_sel.size(0), self.head_dim),
                    k_sel.view(1, 1, k_sel.size(0), self.head_dim),
                    v_sel.view(1, 1, v_sel.size(0), self.head_dim),
                    bias,
                    self.dropout if self.training else 0.0,
                    False,
                    backend=self.attention_backend,
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

    def __init__(self, dim, num_heads, dropout=0.0, attention_backend=DEFAULT_ATTENTION_BACKEND):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.attention_backend = attention_backend
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

        out = attention_sdpa(
            q,
            k,
            v,
            attn_bias,
            self.dropout if self.training else 0.0,
            is_causal and attn_bias is None,
            backend=self.attention_backend,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("learned_block_gqa")
class LearnedBlockSparseGQAAttention(nn.Module):
    """Learned block-routed GQA reference path.

    A small index branch picks top-k KV blocks per GQA group, local blocks are
    always included, and dense SDPA applies the resulting exact token mask. This
    is for correctness experiments, not memory or kernel speedups.

    IndexShare-style sharing: with index_role="shared" the layer holds no
    indexer parameters and reuses the index decisions its owner layer published
    to a bound SharedSparseIndexState. Every layer in a sharing group distills
    its own attention distribution into the owner's index scores; the caller
    supplies kl_loss_weight already divided by the group size, which realizes
    the multi-layer distillation average of the IndexCache training-aware recipe.
    """

    def __init__(
        self,
        dim,
        num_heads,
        num_kv_heads,
        dropout=0.0,
        block_size=128,
        top_k_blocks=32,
        local_blocks=0,
        index_dim=None,
        kl_loss_weight=0.0,
        attention_backend=DEFAULT_ATTENTION_BACKEND,
        index_role="owner",
    ):
        super().__init__()
        require(dim > 0, "dim must be > 0")
        require(num_heads > 0, "num_heads must be > 0")
        require(num_kv_heads > 0, "num_kv_heads must be > 0")
        require(dim % num_heads == 0, "dim must be divisible by num_heads")
        require(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        require(0.0 <= dropout < 1.0, "dropout must be in [0, 1)")
        require(block_size > 0, "block_size must be > 0")
        require(top_k_blocks > 0, "top_k_blocks must be > 0")
        require(local_blocks >= 0, "local_blocks must be >= 0")
        require(kl_loss_weight >= 0, "kl_loss_weight must be >= 0")
        require(index_role in {"owner", "shared"}, "index_role must be 'owner' or 'shared'")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_group_size = num_heads // num_kv_heads
        self.dropout = dropout
        self.block_size = block_size
        self.top_k_blocks = top_k_blocks
        self.local_blocks = local_blocks
        self.kl_loss_weight = kl_loss_weight
        self.sparse_index_warmup = False
        self.attention_backend = attention_backend
        self.index_role = index_role
        self._index_share_state = None
        self._index_share_key = None
        self.index_dim = self.head_dim if index_dim is None else index_dim
        require(self.index_dim > 0, "index_dim must be > 0")

        kv_dim = num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        if index_role == "owner":
            self.q_idx_proj = nn.Linear(dim, num_kv_heads * self.index_dim, bias=False)
            self.k_idx_proj = nn.Linear(dim, self.index_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def set_sparse_index_warmup(self, active):
        self.sparse_index_warmup = bool(active)

    def bind_index_share(self, state, key):
        require(isinstance(state, SharedSparseIndexState), (
            "bind_index_share requires a SharedSparseIndexState"
        ))
        self._index_share_state = state
        self._index_share_key = key

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        k_attn = k.repeat_interleave(self.kv_group_size, dim=1)
        v_attn = v.repeat_interleave(self.kv_group_size, dim=1)
        index_scores, token_support = self._index_decisions(x, B, T, is_causal)
        bias = _bool_to_additive_bias(token_support.repeat_interleave(self.kv_group_size, dim=1), x.dtype)
        self.aux_loss = _learned_block_kl_loss(
            index_scores,
            q,
            k_attn,
            token_support,
            self.kv_group_size,
            attn_bias,
            self.kl_loss_weight,
        )
        bias = _merge_attention_bias(bias, attn_bias)
        out = attention_sdpa(q, k_attn, v_attn, bias, self.dropout if self.training else 0.0, False, backend=self.attention_backend)
        return self.out(out.transpose(1, 2).reshape(B, T, C))

    def _index_decisions(self, x, B, T, is_causal):
        if self.index_role == "shared":
            require(self._index_share_state is not None, (
                "shared sparse index attention requires bind_index_share before forward"
            ))
            return self._index_share_state.read(self._index_share_key, B, T)
        index_input = x.detach()
        q_idx = self.q_idx_proj(index_input).reshape(B, T, self.num_kv_heads, self.index_dim).transpose(1, 2)
        k_idx = self.k_idx_proj(index_input)
        index_scores = _msa_token_scores(q_idx, k_idx, is_causal)
        block_scores = _msa_block_max_scores_from_token_scores(index_scores, self.block_size)
        if self.training and self.sparse_index_warmup:
            token_support = _full_attention_support(B, self.num_kv_heads, T, x.device, is_causal)
        else:
            token_support = _learned_block_attention_support_from_scores(
                block_scores,
                self.block_size,
                self.top_k_blocks,
                self.local_blocks,
                is_causal,
            )
        if self._index_share_state is not None:
            self._index_share_state.publish(self._index_share_key, index_scores, token_support)
        return index_scores, token_support


@register_attention("sliding_window")
class SlidingWindowAttention(nn.Module):
    """Local attention over a fixed token window.

    In causal mode a query attends to the previous `window_size` tokens. In
    bidirectional mode it attends to a symmetric local band.
    """

    def __init__(self, dim, num_heads, dropout=0.0, window_size=128, attention_backend=DEFAULT_ATTENTION_BACKEND):
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
        self.attention_backend = attention_backend
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
        out = attention_sdpa(q, k, v, bias, self.dropout if self.training else 0.0, False, backend=self.attention_backend)
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
        attention_backend=DEFAULT_ATTENTION_BACKEND,
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
        self.attention_backend = attention_backend
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
        out = attention_sdpa(q, k, v, bias, self.dropout if self.training else 0.0, False, backend=self.attention_backend)
        return self.out(out.transpose(1, 2).reshape(B, T, C))
