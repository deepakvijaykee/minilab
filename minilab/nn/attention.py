import torch
import torch.nn as nn
import torch.nn.functional as F

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


@register_attention("mha")
class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).unbind(2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("gqa")
class GroupedQueryAttention(nn.Module):

    def __init__(self, dim, num_heads, num_kv_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.kv_group_size = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * num_kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).reshape(B, T, 2, self.num_kv_heads, self.head_dim)
        k, v = kv.unbind(2)
        k, v = k.transpose(1, 2), v.transpose(1, 2)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, *freqs_cis)

        k = k.repeat_interleave(self.kv_group_size, dim=1)
        v = v.repeat_interleave(self.kv_group_size, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal and attn_bias is None,
        )
        return self.out(out.transpose(1, 2).reshape(B, T, C))


@register_attention("iha")
class InterleavedHeadAttention(nn.Module):
    """Cross-head mixing: each pseudo Q/K is a learned linear combination of all H original Q/K."""

    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_mix = nn.Parameter(torch.eye(num_heads))
        self.k_mix = nn.Parameter(torch.eye(num_heads))
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).unbind(2)

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
