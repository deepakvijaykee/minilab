import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.norm import RMSNorm
from minilab.registry import register_attention


def _cosformer_reweight(q, k):
    T = q.size(2)
    angle = torch.arange(T, device=q.device, dtype=q.dtype) * (math.pi / (2 * max(1, T)))
    angle = angle.view(1, 1, T, 1)
    cos = angle.cos()
    sin = angle.sin()
    return q * cos, q * sin, k * cos, k * sin


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x.float() * x.float()).sum(dim=dim, keepdim=True) + eps).to(x.dtype)


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
    """Qwen3-Next-style Gated DeltaNet mixer using a recurrent reference rule."""

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

