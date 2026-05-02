import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.nn.norm import RMSNorm
from minilab.registry import register_model


@dataclass
class XLSTMConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 4
    max_seq_len: int = 1024
    dropout: float = 0.0
    use_bias: bool = False
    norm_eps: float = 1e-6
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    output_logit_soft_cap: float = 30.0
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64

    def __post_init__(self):
        require(self.vocab_size > 0, "vocab_size must be > 0")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.norm_eps > 0, "norm_eps must be > 0")
        require(self.qk_dim_factor > 0, "qk_dim_factor must be > 0")
        require(self.v_dim_factor > 0, "v_dim_factor must be > 0")
        require(self.gate_soft_cap > 0, "gate_soft_cap must be > 0")
        require(self.output_logit_soft_cap > 0, "output_logit_soft_cap must be > 0")
        require(self.ffn_proj_factor > 0, "ffn_proj_factor must be > 0")
        require(self.ffn_round_up_to_multiple_of > 0, "ffn_round_up_to_multiple_of must be > 0")
        qk_dim = int(self.dim * self.qk_dim_factor)
        v_dim = int(self.dim * self.v_dim_factor)
        require(qk_dim > 0, "qk_dim_factor produces an empty q/k projection")
        require(v_dim > 0, "v_dim_factor produces an empty value projection")
        require(qk_dim % self.num_heads == 0, "q/k projection dim must divide num_heads")
        require(v_dim % self.num_heads == 0, "value projection dim must divide num_heads")


def _round_up_to_multiple(value, multiple):
    return int(math.ceil(value / multiple) * multiple)


def _soft_cap(x, cap):
    return cap * torch.tanh(x / cap)


class MultiHeadLayerNorm(nn.Module):
    def __init__(self, num_heads, head_dim, eps, use_bias):
        super().__init__()
        require(num_heads > 0, "MultiHeadLayerNorm num_heads must be > 0")
        require(head_dim > 0, "MultiHeadLayerNorm head_dim must be > 0")
        require(eps > 0, "MultiHeadLayerNorm eps must be > 0")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_heads * head_dim))
        self.bias = nn.Parameter(torch.zeros(num_heads * head_dim)) if use_bias else None

    def forward(self, x):
        B, T, H, D = x.shape
        require(H == self.num_heads, "MultiHeadLayerNorm received the wrong number of heads")
        require(D == self.head_dim, "MultiHeadLayerNorm received the wrong head dim")
        y = x.float()
        y = y - y.mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = y.to(x.dtype).reshape(B, T, H * D)
        y = y * self.weight.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias.to(x.dtype)
        return y


def _native_mlstm_scan(q, k, v, i_gate, f_gate, eps):
    B, H, T, DQK = q.shape
    DV = v.size(-1)
    state_dtype = torch.float32
    c = torch.zeros(B, H, DQK, DV, device=q.device, dtype=state_dtype)
    n = torch.zeros(B, H, DQK, device=q.device, dtype=state_dtype)
    m = torch.zeros(B, H, 1, device=q.device, dtype=state_dtype)
    outs = []
    scale = DQK ** -0.5

    for t in range(T):
        q_t = q[:, :, t].float()
        k_t = k[:, :, t].float()
        v_t = v[:, :, t].float()
        i_t = i_gate[:, :, t, None].float()
        f_t = F.logsigmoid(f_gate[:, :, t, None].float())

        m_next = torch.maximum(f_t + m, i_t)
        f_act = torch.exp(f_t + m - m_next)
        i_act = torch.exp(i_t - m_next)

        c = (
            f_act[:, :, :, None] * c
            + i_act[:, :, :, None] * torch.einsum("bhd,bhv->bhdv", k_t, v_t)
        )
        n = f_act * n + i_act * k_t
        q_scaled = q_t * scale
        numerator = torch.einsum("bhd,bhdv->bhv", q_scaled, c)
        qn = torch.einsum("bhd,bhd->bh", q_scaled, n).unsqueeze(-1)
        denom = torch.maximum(qn.abs(), torch.exp(-m_next)) + eps
        outs.append((numerator / denom).to(q.dtype))
        m = m_next

    return torch.stack(outs, dim=2)


class MLSTMMixer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.qk_dim = int(config.dim * config.qk_dim_factor)
        self.v_dim = int(config.dim * config.v_dim_factor)
        self.q = nn.Linear(config.dim, self.qk_dim, bias=config.use_bias)
        self.k = nn.Linear(config.dim, self.qk_dim, bias=config.use_bias)
        self.v = nn.Linear(config.dim, self.v_dim, bias=config.use_bias)
        self.ogate_preact = nn.Linear(config.dim, self.v_dim, bias=config.use_bias)
        self.igate_preact = nn.Linear(config.dim, config.num_heads, bias=True)
        self.fgate_preact = nn.Linear(config.dim, config.num_heads, bias=True)
        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=config.num_heads,
            head_dim=self.v_dim // config.num_heads,
            eps=config.norm_eps,
            use_bias=config.use_bias,
        )
        self.out_proj = nn.Linear(self.v_dim, config.dim, bias=config.use_bias)
        self.gate_soft_cap = config.gate_soft_cap
        self.eps = config.norm_eps

    def forward(self, x):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, self.num_heads, -1).transpose(1, 2)
        k = self.k(x).reshape(B, T, self.num_heads, -1).transpose(1, 2)
        v = self.v(x).reshape(B, T, self.num_heads, -1).transpose(1, 2)
        i_gate = _soft_cap(self.igate_preact(x), self.gate_soft_cap).transpose(1, 2)
        f_gate = _soft_cap(self.fgate_preact(x), self.gate_soft_cap).transpose(1, 2)

        h = _native_mlstm_scan(q, k, v, i_gate, f_gate, self.eps)
        h = h.transpose(1, 2)
        h = self.multihead_norm(h)
        h = torch.sigmoid(self.ogate_preact(x)) * h
        return self.out_proj(h)


class XLSTMFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = _round_up_to_multiple(
            config.dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )
        self.proj_up_gate = nn.Linear(config.dim, hidden_dim, bias=config.use_bias)
        self.proj_up = nn.Linear(config.dim, hidden_dim, bias=config.use_bias)
        self.proj_down = nn.Linear(hidden_dim, config.dim, bias=config.use_bias)

    def forward(self, x):
        return self.proj_down(F.silu(self.proj_up_gate(x)) * self.proj_up(x))


class XLSTMBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_mlstm = RMSNorm(config.dim, eps=config.norm_eps)
        self.mlstm = MLSTMMixer(config)
        self.norm_ffn = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn = XLSTMFeedForward(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        x = x + self.drop(self.mlstm(self.norm_mlstm(x)))
        x = x + self.drop(self.ffn(self.norm_ffn(x)))
        return x


@register_model("xlstm")
class XLSTMLM(BaseModel):
    config_class = XLSTMConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([XLSTMBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.dim, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.lm_head)

    def forward(self, idx, targets=None):
        return self._causal_lm_forward(idx, targets)

    def forward_hidden(self, idx):
        require(idx.size(1) <= self.config.max_seq_len, (
            f"XLSTMLM supports at most {self.config.max_seq_len} tokens, got {idx.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        x = self.drop(x)
        for block in self.blocks:
            x = self._checkpointed_forward(block, x)
        x = self.ln_f(x)
        logits = _soft_cap(self.lm_head(x), self.config.output_logit_soft_cap)
        return logits, x
