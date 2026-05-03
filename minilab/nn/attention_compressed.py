import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.attention_common import (
    _apply_tail_rotary,
    _bool_to_additive_bias,
    _local_attention_bias,
)
from minilab.nn.norm import RMSNorm
from minilab.registry import register_attention


def _pad_compressor_inputs(entries, weights, ratio):
    T = entries.size(1)
    pad = math.ceil(T / ratio) * ratio - T
    if pad == 0:
        return entries, weights
    entries = F.pad(entries, (0, 0, 0, pad))
    weights = F.pad(weights, (0, 0, 0, pad), value=float("-inf"))
    return entries, weights


def _compressed_attention_bias(T, comp_end, compress_ratio, device, dtype, is_causal):
    query_pos = torch.arange(T, device=device).unsqueeze(-1)
    comp_end = comp_end.view(1, -1)
    if is_causal:
        current_block_start = torch.div(query_pos, compress_ratio, rounding_mode="floor") * compress_ratio
        allowed = comp_end < current_block_start
    else:
        allowed = torch.ones(T, comp_end.numel(), device=device, dtype=torch.bool)
    return _bool_to_additive_bias(allowed, dtype)


def _output_projection_groups(num_heads):
    for groups in (4, 2):
        if num_heads % groups == 0:
            return groups
    return 1


def _expand_external_bias(attn_bias, batch_size, num_heads, T):
    if attn_bias.dim() == 2:
        require(attn_bias.shape == (T, T), "2D attn_bias must have shape (T, T)")
        return attn_bias.view(1, 1, T, T)
    if attn_bias.dim() == 3:
        require(attn_bias.size(-2) == T and attn_bias.size(-1) == T, (
            "3D attn_bias must end with shape (T, T)"
        ))
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
        require(attn_bias.size(0) in {1, batch_size}, (
            "4D attn_bias batch dimension must be 1 or batch size"
        ))
        require(attn_bias.size(1) in {1, num_heads}, (
            "4D attn_bias head dimension must be 1 or num_heads"
        ))
        return attn_bias
    raise ValueError("attn_bias must have 2, 3, or 4 dimensions")


def _attention_bias_has_causal_support(attn_bias, T):
    if attn_bias is None:
        return False
    require(attn_bias.size(-2) == T and attn_bias.size(-1) == T, (
        f"attn_bias must end with shape ({T}, {T})"
    ))
    support = attn_bias if attn_bias.dtype == torch.bool else torch.isfinite(attn_bias)
    future = torch.triu(torch.ones(T, T, device=attn_bias.device, dtype=torch.bool), diagonal=1)
    if not future.any():
        return True
    return bool((~support[..., future]).all().item())


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
        # External biases only cover raw token keys. Compressed key/value slots
        # need their own causal mask when the caller passes a causal relative
        # position bias, but non-causal structural masks must stay non-causal.
        structural_causal = is_causal or _attention_bias_has_causal_support(attn_bias, T)
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

        comp, comp_end = self._compress_kv(x)
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
                T, comp_end, self.compress_ratio, x.device, x.dtype, structural_causal
            )
            comp_scores = comp_scores + comp_bias
            comp_scores = self._filter_compressed_scores(x, q_latent, comp_bias, comp_scores)
            scores = torch.cat([raw_scores, comp_scores], dim=-1)
            values = torch.cat([local_kv, comp_kv], dim=-2)

        if attn_bias is not None:
            external_bias = _expand_external_bias(attn_bias, B, self.num_heads, T)
            if external_bias.dtype == torch.bool:
                external_bias = _bool_to_additive_bias(external_bias, scores.dtype)
            scores[:, :, :, :T] = scores[:, :, :, :T] + external_bias
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
        return None, None
    G = math.ceil(T / compress_ratio)
    entries, weights = _pad_compressor_inputs(entries, weights, compress_ratio)
    grouped_entries = entries.view(B, G, compress_ratio, D)
    grouped_weights = weights.view(B, G, compress_ratio, D) + pos_bias.view(1, 1, compress_ratio, D)
    compressor = F.softmax(grouped_weights, dim=2)
    comp = (compressor * grouped_entries).sum(dim=2)
    comp_end = torch.arange(1, G + 1, device=entries.device) * compress_ratio - 1
    comp_end = comp_end.clamp(max=T - 1)
    return comp, comp_end


def _compress_csa(entries_a, entries_b, weights_a, weights_b, pos_bias_a, pos_bias_b, compress_ratio):
    B, T, D = entries_a.shape
    if T < compress_ratio:
        return None, None
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
    comp_end = torch.arange(1, G + 1, device=entries_a.device) * compress_ratio - 1
    comp_end = comp_end.clamp(max=T - 1)
    return comp, comp_end


@register_attention("csa")
class CompressedSparseAttention(_CompressedAttentionBase):
    """DeepSeek-V4-style compressed sparse attention reference path."""

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
        index_comp, _ = _compress_csa(
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
    """DeepSeek-V4-style heavily compressed dense attention reference path."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__(dim, num_heads, dropout, compress_ratio=128, window_size=128)
        self.c_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.z_proj = nn.Linear(dim, self.head_dim, bias=False)
        self.pos_bias = nn.Parameter(torch.zeros(self.compress_ratio, self.head_dim))

    def _compress_kv(self, x):
        return _compress_hca(self.c_proj(x), self.z_proj(x), self.pos_bias, self.compress_ratio)
