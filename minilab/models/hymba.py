"""Hymba-style hybrid-head language model.

Hymba combines attention and SSM processing inside each layer: attention heads
retain high-resolution recall while SSM heads summarize context efficiently.
This implementation keeps that invariant directly with parallel attention and
Mamba-2 SSD branches over the same normalized stream, then merges them before
the feed-forward block. Optional learned meta tokens prepend trainable context
to every sequence and are removed from returned logits.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.checks import require
from minilab.models.transformer_utils import (
    apply_logit_softcap,
    apply_simple_position,
    build_rope_or_simple_position,
    commit_transformer_block_updates,
    set_transformer_qk_clip_recording,
    transformer_auxiliary_loss,
    transformer_supports_qk_clip,
)
from minilab.models.gpt import (
    _build_transformer_attention,
    _build_transformer_ffn,
    GPTConfig,
)
from minilab.nn.ssm import Mamba2Mixer
from minilab.registry import get_norm, register_model


@dataclass
class HymbaConfig(GPTConfig):
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    num_meta_tokens: int = 0
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        require(self.connection == "residual", "HymbaLM currently requires residual connections")
        require(self.per_layer_embedding_dim == 0, "HymbaLM does not support per-layer embeddings")
        require(self.mtp_depth == 0 and self.mtp_loss_weight == 0, "HymbaLM does not implement MTP")
        require(self.position in {"rope", "yarn_rope", "none", "sinusoidal"}, (
            "HymbaLM supports RoPE, YaRN RoPE, sinusoidal, or no position encoding; "
            "Gemma/Qwen local-global rotary schedules are implemented by GPT, not HymbaLM"
        ))
        require(self.d_state > 0, "d_state must be > 0")
        require(self.d_conv > 0, "d_conv must be > 0")
        require(self.expand > 0, "expand must be > 0")
        require(self.headdim > 0, "headdim must be > 0")
        require((self.expand * self.dim) % self.headdim == 0, "expand * dim must be divisible by headdim")
        require(self.ngroups > 0, "ngroups must be > 0")
        require((self.expand * self.dim // self.headdim) % self.ngroups == 0, (
            "Hymba SSM nheads must be divisible by ngroups"
        ))
        require(self.num_meta_tokens >= 0, "num_meta_tokens must be >= 0")
        require(0 < self.dt_min < self.dt_max, "dt_min and dt_max must satisfy 0 < dt_min < dt_max")
        require(self.dt_init_floor > 0, "dt_init_floor must be > 0")


class HymbaBlock(nn.Module):
    def __init__(self, config, block_id):
        super().__init__()
        self.branch_norm = get_norm(config.norm)(config.dim)
        self.ffn_norm = get_norm(config.norm)(config.dim)
        self.attention_name, self.attn = _build_transformer_attention(config, block_id)
        self.ssm = Mamba2Mixer(
            config.dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
        )
        self.merge = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.ffn = _build_transformer_ffn(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        h = self.branch_norm(x)
        attn = self.attn(h, freqs_cis, attn_bias, is_causal)
        ssm = self.ssm(h)
        x = x + self.drop(self.merge(torch.cat([attn, ssm], dim=-1)))
        x = x + self.drop(self.ffn(self.ffn_norm(x)))
        return x


@register_model("hymba")
class HymbaLM(BaseModel):
    config_class = HymbaConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.meta_tokens = nn.Parameter(torch.zeros(config.num_meta_tokens, config.dim))
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([HymbaBlock(config, i) for i in range(config.num_layers)])
        self.ln_f = get_norm(config.norm)(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.pos_enc = build_rope_or_simple_position(config, "HymbaLM")
        self.apply(self._init_weights)
        if config.num_meta_tokens:
            nn.init.normal_(self.meta_tokens, mean=0.0, std=0.02)

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.pos_enc is not None and any(param.requires_grad for param in self.pos_enc.parameters()):
            modules.append(self.pos_enc)
        return tuple(modules)

    def no_weight_decay_parameter_names(self):
        return self._parameter_names_ending_with(".A_log", ".dt_bias")

    def set_qk_clip_recording(self, enabled):
        set_transformer_qk_clip_recording(self.blocks, enabled)

    def supports_qk_clip(self):
        return transformer_supports_qk_clip(self.blocks)

    def auxiliary_loss(self):
        return transformer_auxiliary_loss(self.blocks, self.config.ffn, next(self.parameters()))

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_transformer_block_updates(
            self.blocks,
            self.config.ffn,
            qk_clip_threshold,
            qk_clip_balance,
        )

    def forward(self, idx, targets=None):
        return self._causal_lm_forward(idx, targets, include_auxiliary_loss=True)

    def forward_hidden(self, idx):
        meta = self.config.num_meta_tokens
        total_len = idx.size(1) + meta
        require(total_len <= self.config.max_seq_len, (
            f"HymbaLM supports at most {self.config.max_seq_len} tokens including meta tokens, got {total_len}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        if meta:
            prefix = self.meta_tokens.unsqueeze(0).expand(idx.size(0), -1, -1)
            x = torch.cat([prefix.to(dtype=x.dtype), x], dim=1)
        x, freqs_cis, attn_bias, is_causal = apply_simple_position(self.pos_enc, x, total_len, "HymbaLM")
        x = self.drop(x)
        for block in self.blocks:
            x = self._checkpointed_forward(
                block,
                x,
                freqs_cis=freqs_cis,
                attn_bias=attn_bias,
                is_causal=is_causal,
            )
        x = self.ln_f(x)
        if meta:
            x = x[:, meta:]
        return apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap), x
