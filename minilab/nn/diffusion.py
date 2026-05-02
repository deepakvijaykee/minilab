"""AdaLN, DiffusionBlock, SinusoidalTimeEmbedding — shared by MDLM, SEDD, D3PM."""

import math

import torch
import torch.nn as nn

from minilab.checks import require
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    QK_CLIP_ATTENTIONS,
    resolve_deepseek_v4_attention,
)
from minilab.registry import get_attention, get_ffn


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim):
        super().__init__()
        require(dim > 0, "SinusoidalTimeEmbedding dim must be > 0")
        require(dim % 2 == 0, "SinusoidalTimeEmbedding requires even dim")
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):
        require(t.dim() in {1, 2}, "diffusion time tensor must have shape (batch,) or (batch, seq)")
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
        args = t.float().unsqueeze(-1) * freqs
        return self.mlp(torch.cat([args.sin(), args.cos()], dim=-1))


class AdaLN(nn.Module):
    """Norm conditioned on time: scale and shift from time embedding."""

    def __init__(self, dim):
        super().__init__()
        require(dim > 0, "AdaLN dim must be > 0")
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(dim, 2 * dim)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        if cond.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        elif cond.dim() == 3:
            require(cond.shape[:2] == x.shape[:2], "token-wise diffusion conditioning must match x batch/seq")
        else:
            raise ValueError("diffusion conditioning must have shape (batch, dim) or (batch, seq, dim)")
        return self.norm(x) * (1 + scale) + shift


class DiffusionBlock(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        ffn_hidden,
        dropout=0.0,
        *,
        attention="mha",
        num_kv_heads=None,
        ffn="swiglu",
        num_experts=8,
        top_k_experts=2,
        block_id=0,
    ):
        super().__init__()
        require(dim > 0, "DiffusionBlock dim must be > 0")
        require(num_heads > 0, "DiffusionBlock num_heads must be > 0")
        require(ffn_hidden > 0, "DiffusionBlock ffn_hidden must be > 0")
        require(0.0 <= dropout < 1.0, "DiffusionBlock dropout must be in [0, 1)")
        self.norm1 = AdaLN(dim)
        attention = _resolve_attention_name(attention, block_id)
        self.attention_name = attention
        attn_cls = get_attention(attention)
        if attention in GQA_ATTENTIONS:
            require(num_kv_heads is not None, "gqa diffusion attention requires num_kv_heads")
            self.attn = attn_cls(dim, num_heads, num_kv_heads, dropout)
        else:
            self.attn = attn_cls(dim, num_heads, dropout)
        self.norm2 = AdaLN(dim)
        self.ffn_name = ffn
        if ffn in MOE_FFNS:
            self.ffn = get_ffn(ffn)(dim, ffn_hidden, num_experts=num_experts, top_k=top_k_experts)
        else:
            self.ffn = get_ffn(ffn)(dim, ffn_hidden)

    def forward(self, x, t_emb, freqs_cis=None, attn_bias=None):
        x = x + self.attn(self.norm1(x, t_emb), freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=False)
        x = x + self.ffn(self.norm2(x, t_emb))
        return x

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        if self.ffn_name == "aux_free_moe":
            self.ffn.commit_routing_bias_update()
        if qk_clip_threshold > 0 and self.attention_name in QK_CLIP_ATTENTIONS:
            self.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)

    def set_qk_clip_recording(self, enabled):
        if self.attention_name in QK_CLIP_ATTENTIONS:
            self.attn.set_qk_clip_recording(enabled)


def diffusion_blocks_auxiliary_loss(blocks, reference):
    loss = reference.sum() * 0.0
    for block in blocks:
        if block.ffn_name in MOE_FFNS:
            loss = loss + block.ffn.aux_loss
    return loss


def commit_diffusion_block_updates(blocks, qk_clip_threshold, qk_clip_balance):
    for block in blocks:
        block.post_optimizer_step(qk_clip_threshold, qk_clip_balance)


def _resolve_attention_name(attention, block_id):
    return resolve_deepseek_v4_attention(attention, block_id)
