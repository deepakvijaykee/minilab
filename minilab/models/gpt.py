from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.nn.connections import expand_residual_stream, reduce_residual_stream
from minilab.registry import get_attention, get_connection, get_ffn, get_norm, get_position, register_model


@dataclass
class GPTConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int | None = None
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    attention: str = "mha"
    position: str = "rope"
    norm: str = "rmsnorm"
    ffn: str = "swiglu"
    connection: str = "residual"
    connection_expansion: int = 4
    num_experts: int = 8
    top_k_experts: int = 2

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        require(self.vocab_size > 0, "vocab_size must be > 0")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.num_kv_heads > 0, "num_kv_heads must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(self.dim % self.num_heads == 0, "dim must be divisible by num_heads")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.ffn_mult > 0, "ffn_mult must be > 0")
        require(self.connection_expansion > 0, "connection_expansion must be > 0")
        if self.attention == "gqa":
            require(self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        if self.position == "rope":
            require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        if self.position == "sinusoidal":
            require(self.dim % 2 == 0, "sinusoidal position requires even dim")
        if self.ffn == "moe":
            require(self.num_experts > 0, "num_experts must be > 0")
            require(1 <= self.top_k_experts <= self.num_experts, "top_k_experts must be in [1, num_experts]")


PRESETS = {
    "gpt-tiny": dict(dim=128, num_layers=4, num_heads=4, max_seq_len=256),
    "gpt-small": dict(dim=256, num_layers=6, num_heads=8, max_seq_len=512),
    "gpt-medium": dict(dim=512, num_layers=12, num_heads=8, max_seq_len=1024),
    "gpt-large": dict(dim=768, num_layers=24, num_heads=12, max_seq_len=2048),
}


def gpt_preset(name, vocab_size, **overrides):
    require(name in PRESETS, f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    return GPTConfig(vocab_size=vocab_size, **{**PRESETS[name], **overrides})


class TransformerBlock(nn.Module):
    def __init__(self, config, block_id):
        super().__init__()
        self.attn_norm = get_norm(config.norm)(config.dim)
        self.ffn_norm = get_norm(config.norm)(config.dim)
        self.drop = nn.Dropout(config.dropout)

        ffn_hidden = int(config.dim * config.ffn_mult)
        if config.ffn == "moe":
            self.ffn = get_ffn("moe")(config.dim, ffn_hidden, num_experts=config.num_experts, top_k=config.top_k_experts)
        else:
            self.ffn = get_ffn(config.ffn)(config.dim, ffn_hidden)

        attn_cls = get_attention(config.attention)
        if config.attention == "gqa":
            self.attn = attn_cls(config.dim, config.num_heads, config.num_kv_heads, config.dropout)
        else:
            self.attn = attn_cls(config.dim, config.num_heads, config.dropout)

        conn_cls = get_connection(config.connection)
        if config.connection == "residual":
            self.attn_conn = conn_cls(config.dim)
            self.ffn_conn = conn_cls(config.dim)
        else:
            self.attn_conn = conn_cls(config.dim, config.connection_expansion, layer_id=2 * block_id)
            self.ffn_conn = conn_cls(config.dim, config.connection_expansion, layer_id=2 * block_id + 1)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        x = self.attn_conn(x, lambda h: self.drop(self.attn(self.attn_norm(h), freqs_cis, attn_bias, is_causal)))
        x = self.ffn_conn(x, lambda h: self.drop(self.ffn(self.ffn_norm(h))))
        return x


@register_model("gpt")
class GPT(BaseModel):
    config_class = GPTConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        self.ln_f = get_norm(config.norm)(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight

        if config.position == "rope":
            self.pos_enc = get_position("rope")(config.dim // config.num_heads, config.max_seq_len)
        elif config.position == "alibi":
            self.pos_enc = get_position("alibi")(config.num_heads, config.max_seq_len)
        else:
            self.pos_enc = get_position(config.position)(config.dim, config.max_seq_len)

        self.apply(self._init_weights)
        if config.connection in ("hc", "mhc"):
            for block in self.blocks:
                block.attn_conn.reset_dynamic_parameters()
                block.ffn_conn.reset_dynamic_parameters()

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.config.position == "learned":
            modules.append(self.pos_enc)
        if self.config.connection != "residual":
            for block in self.blocks:
                modules.extend((block.attn_conn, block.ffn_conn))
        return tuple(modules)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self._cast_hidden(self.tok_emb(idx))

        freqs_cis, attn_bias, is_causal = None, None, True
        if self.pos_enc.kind == "rotary":
            freqs_cis = self.pos_enc(T)
        elif self.pos_enc.kind == "bias":
            attn_bias = self.pos_enc(T).unsqueeze(0)
            is_causal = False
        elif self.pos_enc.kind == "additive":
            x = x + self._cast_hidden(self.pos_enc(T))

        x = self.drop(x)

        if self.config.connection != "residual":
            x = expand_residual_stream(x, self.config.connection_expansion)

        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, freqs_cis, attn_bias, is_causal, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)

        if self.config.connection != "residual":
            x = reduce_residual_stream(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
            if self.config.ffn == "moe":
                for block in self.blocks:
                    loss = loss + block.ffn.aux_loss

        return logits, loss
