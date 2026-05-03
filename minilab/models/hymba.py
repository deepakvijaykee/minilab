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
from minilab.config import BaseConfig
from minilab.models.transformer_utils import (
    DEFAULT_LOCAL_ATTENTION_WINDOW,
    DEFAULT_NUM_EXPERTS,
    DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL,
    DEFAULT_ROPE_BASE,
    DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN,
    DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR,
    DEFAULT_ROPE_SCALING_FACTOR,
    DEFAULT_TOP_K_EXPERTS,
    DEFAULT_YARN_BETA_FAST,
    DEFAULT_YARN_BETA_SLOW,
    attention_freqs_for_attention,
    apply_logit_softcap,
    apply_simple_position,
    build_rope_or_simple_position,
    commit_transformer_block_updates,
    set_transformer_qk_clip_recording,
    transformer_auxiliary_loss,
    transformer_supports_qk_clip,
    validate_parallel_or_interleaved_lm_config,
)
from minilab.models.gpt import (
    _build_transformer_attention,
    _build_transformer_ffn,
)
from minilab.nn.ssm import Mamba2Mixer
from minilab.registry import get_norm, register_model


@dataclass
class HymbaConfig(BaseConfig):
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
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_k_experts: int = DEFAULT_TOP_K_EXPERTS
    post_norm: bool = False
    rope_base: float = DEFAULT_ROPE_BASE
    rope_scaling_factor: float = DEFAULT_ROPE_SCALING_FACTOR
    rope_original_max_seq_len: int = DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN
    rope_partial_rotary_factor: float = DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR
    yarn_beta_fast: float = DEFAULT_YARN_BETA_FAST
    yarn_beta_slow: float = DEFAULT_YARN_BETA_SLOW
    local_attention_window: int = DEFAULT_LOCAL_ATTENTION_WINDOW
    qwen3_next_full_attention_interval: int = DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL
    final_logit_softcap: float = 0.0
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
        validate_parallel_or_interleaved_lm_config(self, "HymbaLM")
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
        self.post_norm = config.post_norm
        if self.post_norm:
            self.branch_post_norm = get_norm(config.norm)(config.dim)
            self.ffn_post_norm = get_norm(config.norm)(config.dim)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        h = self.branch_norm(x)
        block_freqs = attention_freqs_for_attention(self.attention_name, freqs_cis)
        attn = self.attn(h, block_freqs, attn_bias, is_causal)
        ssm = self.ssm(h)
        branch_out = self.merge(torch.cat([attn, ssm], dim=-1))
        if self.post_norm:
            branch_out = self.branch_post_norm(branch_out)
        x = x + self.drop(branch_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        if self.post_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        return x + self.drop(ffn_out)


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
