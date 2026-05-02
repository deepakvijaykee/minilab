import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.attention_common import _QKClipMixin, apply_rotary_emb
from minilab.registry import register_attention


def _apply_partial_rotary(q, k, freqs_cis, rope_dim):
    cos, sin = freqs_cis
    half = rope_dim // 2
    return apply_rotary_emb(q, k, cos[:, :half], sin[:, :half])


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
        self._record_qk_clip_logits(q, k, attn_bias=attn_bias, is_causal=is_causal and attn_bias is None)
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

