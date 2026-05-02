"""Hybrid attention/SSM language model.

The layer schedule follows the practical Jamba/Hymba pattern of interleaving
Transformer attention blocks with Mamba state-space blocks while preserving the
normal causal LM objective and weight tying used elsewhere in minilab.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.checks import require
from minilab.models.gpt import GPTConfig, TransformerBlock
from minilab.models.mamba import MambaBlock
from minilab.nn.architecture import MOE_FFNS, QK_CLIP_ATTENTIONS
from minilab.registry import get_norm, get_position, register_model


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
        self.pos_enc = _hybrid_position(config)
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.pos_enc is not None and any(param.requires_grad for param in self.pos_enc.parameters()):
            modules.append(self.pos_enc)
        return tuple(modules)

    def no_weight_decay_parameter_names(self):
        return tuple(name for name, _ in self.named_parameters() if name.endswith(".A_log"))

    def set_qk_clip_recording(self, enabled):
        for block in self.blocks:
            if isinstance(block, TransformerBlock) and block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.set_qk_clip_recording(enabled)

    def supports_qk_clip(self):
        return any(
            isinstance(block, TransformerBlock) and block.attention_name in QK_CLIP_ATTENTIONS
            for block in self.blocks
        )

    def forward(self, idx, targets=None):
        return self._causal_lm_forward(idx, targets, include_auxiliary_loss=True)

    def forward_hidden(self, idx):
        require(idx.size(1) <= self.config.max_seq_len, (
            f"HybridLM supports at most {self.config.max_seq_len} tokens, got {idx.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        T = idx.size(1)
        freqs_cis = None
        attn_bias = None
        is_causal = True
        if self.pos_enc is not None:
            if self.pos_enc.kind == "rotary":
                freqs_cis = self.pos_enc(T)
            elif self.pos_enc.kind == "additive":
                x = x + self._cast_hidden(self.pos_enc(T))
            else:
                require(self.pos_enc.kind == "none", f"HybridLM unsupported position kind: {self.pos_enc.kind}")
        x = self.drop(x)
        for block in self.blocks:
            if isinstance(block, MambaBlock):
                if self._gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            else:
                if self._gradient_checkpointing and self.training:
                    def run_block(h, block=block):
                        return block(h, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)
                    x = torch.utils.checkpoint.checkpoint(run_block, x, use_reentrant=False)
                else:
                    x = block(x, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if self.config.final_logit_softcap > 0:
            logits = torch.tanh(logits / self.config.final_logit_softcap) * self.config.final_logit_softcap
        return logits, x

    def auxiliary_loss(self):
        loss = next(self.parameters()).sum() * 0.0
        if self.config.ffn not in MOE_FFNS:
            return loss
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                loss = loss + block.ffn.aux_loss
        return loss

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        if self.config.ffn == "aux_free_moe":
            for block in self.blocks:
                if isinstance(block, TransformerBlock):
                    block.ffn.commit_routing_bias_update()
        if qk_clip_threshold <= 0:
            return
        for block in self.blocks:
            if isinstance(block, TransformerBlock) and block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)


def _is_mamba_layer(layer_id, config):
    return layer_id % config.mamba_every == config.mamba_offset


def _hybrid_position(config):
    head_dim = config.dim // config.num_heads
    if config.position == "rope":
        return get_position("rope")(head_dim, config.max_seq_len, base=config.rope_base)
    if config.position == "sinusoidal":
        return get_position("sinusoidal")(config.dim, config.max_seq_len)
    if config.position == "none":
        return get_position("none")(config.dim, config.max_seq_len)
    if config.position == "yarn_rope":
        return get_position("yarn_rope")(
            head_dim,
            config.max_seq_len,
            base=config.rope_base,
            factor=config.rope_scaling_factor,
            original_max_seq_len=config.rope_original_max_seq_len,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )
    require(False, (
        f"HybridLM unsupported position variant: {config.position}. "
        "Gemma/Qwen local-global rotary schedules are implemented by GPT, not HybridLM."
    ))
