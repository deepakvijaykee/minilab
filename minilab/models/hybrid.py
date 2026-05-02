"""Hybrid attention/SSM language model.

The layer schedule follows the practical Jamba/Hymba pattern of interleaving
Transformer attention blocks with Mamba state-space blocks while preserving the
normal causal LM objective and weight tying used elsewhere in minilab.
"""

from dataclasses import dataclass

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
    GPTConfig,
    TransformerBlock,
)
from minilab.models.mamba import MambaBlock
from minilab.registry import get_norm, register_model


@dataclass
class HybridConfig(GPTConfig):
    mamba_every: int = 2
    mamba_offset: int = 1
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.dt_rank is None:
            self.dt_rank = (self.dim + 15) // 16
        require(self.mamba_every > 0, "mamba_every must be > 0")
        require(0 <= self.mamba_offset < self.mamba_every, "mamba_offset must be in [0, mamba_every)")
        require(self.connection == "residual", "HybridLM currently requires residual connections")
        require(self.per_layer_embedding_dim == 0, "HybridLM does not support per-layer embeddings")
        require(self.mtp_depth == 0 and self.mtp_loss_weight == 0, "HybridLM does not implement MTP")
        require(self.position in {"rope", "yarn_rope", "none", "sinusoidal"}, (
            "HybridLM supports RoPE, YaRN RoPE, sinusoidal, or no position encoding; "
            "Gemma/Qwen local-global rotary schedules are implemented by GPT, not HybridLM"
        ))
        require(self.d_state > 0, "d_state must be > 0")
        require(self.d_conv > 0, "d_conv must be > 0")
        require(self.expand > 0, "expand must be > 0")
        require(self.dt_rank > 0, "dt_rank must be > 0")
        require(0 < self.dt_min < self.dt_max, "dt_min and dt_max must satisfy 0 < dt_min < dt_max")
        require(self.dt_init_floor > 0, "dt_init_floor must be > 0")


@register_model("hybrid")
class HybridLM(BaseModel):
    config_class = HybridConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            MambaBlock(config) if _is_mamba_layer(i, config) else TransformerBlock(config, i)
            for i in range(config.num_layers)
        ])
        self.ln_f = get_norm(config.norm)(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.pos_enc = build_rope_or_simple_position(config, "HybridLM")
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.pos_enc is not None and any(param.requires_grad for param in self.pos_enc.parameters()):
            modules.append(self.pos_enc)
        return tuple(modules)

    def no_weight_decay_parameter_names(self):
        return self._parameter_names_ending_with(".A_log")

    def set_qk_clip_recording(self, enabled):
        set_transformer_qk_clip_recording(self._transformer_blocks(), enabled)

    def supports_qk_clip(self):
        return transformer_supports_qk_clip(self._transformer_blocks())

    def auxiliary_loss(self):
        return transformer_auxiliary_loss(self._transformer_blocks(), self.config.ffn, next(self.parameters()))

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_transformer_block_updates(
            self._transformer_blocks(),
            self.config.ffn,
            qk_clip_threshold,
            qk_clip_balance,
        )

    def forward(self, idx, targets=None):
        return self._causal_lm_forward(idx, targets, include_auxiliary_loss=True)

    def forward_hidden(self, idx):
        require(idx.size(1) <= self.config.max_seq_len, (
            f"HybridLM supports at most {self.config.max_seq_len} tokens, got {idx.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        T = idx.size(1)
        x, freqs_cis, attn_bias, is_causal = apply_simple_position(self.pos_enc, x, T, "HybridLM")
        x = self.drop(x)
        for block in self.blocks:
            if isinstance(block, MambaBlock):
                x = self._checkpointed_forward(block, x)
            else:
                x = self._checkpointed_forward(
                    block,
                    x,
                    freqs_cis=freqs_cis,
                    attn_bias=attn_bias,
                    is_causal=is_causal,
                )
        x = self.ln_f(x)
        return apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap), x

    def _transformer_blocks(self):
        return [block for block in self.blocks if isinstance(block, TransformerBlock)]


def _is_mamba_layer(layer_id, config):
    return layer_id % config.mamba_every == config.mamba_offset
