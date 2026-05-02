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
from minilab.models.gpt import GPTConfig, _build_transformer_attention, _build_transformer_ffn
from minilab.nn.architecture import MOE_FFNS, QK_CLIP_ATTENTIONS
from minilab.nn.ssm import Mamba2Mixer
from minilab.registry import get_norm, get_position, register_model


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
        self.pos_enc = _hymba_position(config)
        self.apply(self._init_weights)
        if config.num_meta_tokens:
            nn.init.normal_(self.meta_tokens, mean=0.0, std=0.02)

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.pos_enc is not None and any(param.requires_grad for param in self.pos_enc.parameters()):
            modules.append(self.pos_enc)
        return tuple(modules)

    def no_weight_decay_parameter_names(self):
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.endswith(".A_log") or name.endswith(".dt_bias")
        )

    def set_qk_clip_recording(self, enabled):
        for block in self.blocks:
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.set_qk_clip_recording(enabled)

    def supports_qk_clip(self):
        return any(block.attention_name in QK_CLIP_ATTENTIONS for block in self.blocks)

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
        freqs_cis = None
        attn_bias = None
        if self.pos_enc is not None:
            if self.pos_enc.kind == "rotary":
                freqs_cis = self.pos_enc(total_len)
            elif self.pos_enc.kind == "additive":
                x = x + self._cast_hidden(self.pos_enc(total_len))
            else:
                require(self.pos_enc.kind == "none", f"HymbaLM unsupported position kind: {self.pos_enc.kind}")
        x = self.drop(x)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                def run_block(h, block=block):
                    return block(h, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=True)
                x = torch.utils.checkpoint.checkpoint(run_block, x, use_reentrant=False)
            else:
                x = block(x, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=True)
        x = self.ln_f(x)
        if meta:
            x = x[:, meta:]
        logits = self.lm_head(x)
        if self.config.final_logit_softcap > 0:
            logits = torch.tanh(logits / self.config.final_logit_softcap) * self.config.final_logit_softcap
        return logits, x

    def auxiliary_loss(self):
        loss = next(self.parameters()).sum() * 0.0
        if self.config.ffn not in MOE_FFNS:
            return loss
        for block in self.blocks:
            loss = loss + block.ffn.aux_loss
        return loss

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        if self.config.ffn == "aux_free_moe":
            for block in self.blocks:
                block.ffn.commit_routing_bias_update()
        if qk_clip_threshold <= 0:
            return
        for block in self.blocks:
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)


def _hymba_position(config):
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
        f"HymbaLM unsupported position variant: {config.position}. "
        "Gemma/Qwen local-global rotary schedules are implemented by GPT, not HymbaLM."
    ))
