"""Hybrid attention/SSM language model.

The layer schedule follows the practical Jamba/Hymba pattern of interleaving
Transformer attention blocks with Mamba state-space blocks while preserving the
normal causal LM objective and weight tying used elsewhere in minilab.
"""

from dataclasses import dataclass

import torch.nn as nn

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.models.transformer_utils import (
    attention_freqs_for_attention,
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
)
from minilab.models.mamba import MambaBlock
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    TOP_ONE_MOE_FFNS,
    resolve_deepseek_v4_attention,
)
from minilab.registry import get_norm, register_model


_LOCAL_WINDOW_ATTENTIONS = {"sliding_window", "sliding_window_gqa_qknorm"}
_PARTIAL_ROPE_ATTENTIONS = {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope", "qwen3_next"}
_DEFAULT_ROPE_BASE = 10000.0
_DEFAULT_ROPE_SCALING_FACTOR = 1.0
_DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN = 4096
_DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR = 0.25
_DEFAULT_YARN_BETA_FAST = 32.0
_DEFAULT_YARN_BETA_SLOW = 1.0
_DEFAULT_LOCAL_ATTENTION_WINDOW = 1024
_DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL = 4
_DEFAULT_NUM_EXPERTS = 8
_DEFAULT_TOP_K_EXPERTS = 2


@dataclass
class HybridConfig(BaseConfig):
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
    num_experts: int = _DEFAULT_NUM_EXPERTS
    top_k_experts: int = _DEFAULT_TOP_K_EXPERTS
    post_norm: bool = False
    rope_base: float = _DEFAULT_ROPE_BASE
    rope_scaling_factor: float = _DEFAULT_ROPE_SCALING_FACTOR
    rope_original_max_seq_len: int = _DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN
    rope_partial_rotary_factor: float = _DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR
    yarn_beta_fast: float = _DEFAULT_YARN_BETA_FAST
    yarn_beta_slow: float = _DEFAULT_YARN_BETA_SLOW
    local_attention_window: int = _DEFAULT_LOCAL_ATTENTION_WINDOW
    qwen3_next_full_attention_interval: int = _DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL
    final_logit_softcap: float = 0.0
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
        if self.dt_rank is None:
            self.dt_rank = (self.dim + 15) // 16
        self._validate_lm_fields()
        self._validate_transformer_branch_fields()
        require(self.mamba_every > 0, "mamba_every must be > 0")
        require(0 <= self.mamba_offset < self.mamba_every, "mamba_offset must be in [0, mamba_every)")
        require(self.d_state > 0, "d_state must be > 0")
        require(self.d_conv > 0, "d_conv must be > 0")
        require(self.expand > 0, "expand must be > 0")
        require(self.dt_rank > 0, "dt_rank must be > 0")
        require(0 < self.dt_min < self.dt_max, "dt_min and dt_max must satisfy 0 < dt_min < dt_max")
        require(self.dt_init_floor > 0, "dt_init_floor must be > 0")

    def _validate_lm_fields(self):
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
        require(self.rope_base > 0, "rope_base must be > 0")
        require(self.rope_scaling_factor > 0, "rope_scaling_factor must be > 0")
        require(self.rope_original_max_seq_len > 0, "rope_original_max_seq_len must be > 0")
        require(0.0 < self.rope_partial_rotary_factor <= 1.0, "rope_partial_rotary_factor must be in (0, 1]")
        require(self.yarn_beta_fast > 0, "yarn_beta_fast must be > 0")
        require(self.yarn_beta_slow > 0, "yarn_beta_slow must be > 0")
        require(self.local_attention_window > 0, "local_attention_window must be > 0")
        require(self.qwen3_next_full_attention_interval > 0, "qwen3_next_full_attention_interval must be > 0")
        require(self.final_logit_softcap >= 0, "final_logit_softcap must be >= 0")

    def _validate_transformer_branch_fields(self):
        require(self.position in {"rope", "yarn_rope", "none", "sinusoidal"}, (
            "HybridLM supports RoPE, YaRN RoPE, sinusoidal, or no position encoding; "
            "Gemma/Qwen local-global rotary schedules are implemented by GPT, not HybridLM"
        ))
        require(self.attention not in {"gemma3", "gemma4"}, (
            "Gemma local-global attention schedules are implemented by GPT, not HybridLM"
        ))
        if _attention_uses_gqa(self.attention):
            require(self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        else:
            require(self.num_kv_heads == self.num_heads, "num_kv_heads only applies to GQA attention variants")
        if self.position in {"rope", "yarn_rope"}:
            require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        if self.position == "sinusoidal":
            require(self.dim % 2 == 0, "sinusoidal position requires even dim")
        if self.attention in {"cosformer", "lightning"}:
            require(self.position == "none", f"{self.attention} owns its positional rule; set position='none'")
        if self.attention == "qwen3_next":
            require(self.position == "yarn_rope", "Qwen3-Next-style HybridLM requires position='yarn_rope'")
        resolved_attention = resolve_deepseek_v4_attention(self.attention, 0)
        uses_local_window = self.attention in _LOCAL_WINDOW_ATTENTIONS or resolved_attention in _LOCAL_WINDOW_ATTENTIONS
        uses_partial_rope = self.attention in _PARTIAL_ROPE_ATTENTIONS or resolved_attention in _PARTIAL_ROPE_ATTENTIONS
        require(
            self.rope_base == _DEFAULT_ROPE_BASE or self.position in {"rope", "yarn_rope"},
            "rope_base only applies to position='rope' or position='yarn_rope'",
        )
        require(
            self.rope_scaling_factor == _DEFAULT_ROPE_SCALING_FACTOR or self.position == "yarn_rope",
            "rope_scaling_factor only applies to position='yarn_rope'",
        )
        require(
            self.rope_original_max_seq_len == _DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN or self.position == "yarn_rope",
            "rope_original_max_seq_len only applies to position='yarn_rope'",
        )
        require(
            self.yarn_beta_fast == _DEFAULT_YARN_BETA_FAST and self.yarn_beta_slow == _DEFAULT_YARN_BETA_SLOW
            or self.position == "yarn_rope",
            "yarn_beta_fast and yarn_beta_slow only apply to position='yarn_rope'",
        )
        require(
            self.local_attention_window == _DEFAULT_LOCAL_ATTENTION_WINDOW or uses_local_window,
            "local_attention_window only applies to local/sliding-window attention",
        )
        require(
            self.rope_partial_rotary_factor == _DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR or uses_partial_rope,
            "rope_partial_rotary_factor only applies to partial-RoPE attention",
        )
        require(
            self.qwen3_next_full_attention_interval == _DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL
            or self.attention == "qwen3_next",
            "qwen3_next_full_attention_interval only applies to attention='qwen3_next'",
        )
        if self.ffn in MOE_FFNS:
            require(self.num_experts > 0, "num_experts must be > 0")
            require(1 <= self.top_k_experts <= self.num_experts, "top_k_experts must be in [1, num_experts]")
            if self.ffn in TOP_ONE_MOE_FFNS:
                require(self.top_k_experts == 1, f"{self.ffn} requires top_k_experts=1")
        else:
            require(
                self.num_experts == _DEFAULT_NUM_EXPERTS and self.top_k_experts == _DEFAULT_TOP_K_EXPERTS,
                "num_experts and top_k_experts only apply to MoE FFNs",
            )


class HybridAttentionBlock(nn.Module):
    def __init__(self, config, block_id):
        super().__init__()
        self.attn_norm = get_norm(config.norm)(config.dim)
        self.ffn_norm = get_norm(config.norm)(config.dim)
        self.post_norm = config.post_norm
        if self.post_norm:
            self.attn_post_norm = get_norm(config.norm)(config.dim)
            self.ffn_post_norm = get_norm(config.norm)(config.dim)
        self.attention_name, self.attn = _build_transformer_attention(config, block_id)
        self.ffn = _build_transformer_ffn(config)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False):
        attn_out = self.attn(self.attn_norm(x), freqs_cis, attn_bias, is_causal)
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out)
        x = x + self.drop(attn_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        if self.post_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        return x + self.drop(ffn_out)


@register_model("hybrid")
class HybridLM(BaseModel):
    config_class = HybridConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([
            MambaBlock(config) if _is_mamba_layer(i, config) else HybridAttentionBlock(config, i)
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
                block_freqs = attention_freqs_for_attention(block.attention_name, freqs_cis)
                x = self._checkpointed_forward(
                    block,
                    x,
                    freqs_cis=block_freqs,
                    attn_bias=attn_bias,
                    is_causal=is_causal,
                )
        x = self.ln_f(x)
        return apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap), x

    def _transformer_blocks(self):
        return [block for block in self.blocks if isinstance(block, HybridAttentionBlock)]


def _is_mamba_layer(layer_id, config):
    return layer_id % config.mamba_every == config.mamba_offset


def _attention_uses_gqa(attention):
    return attention == "qwen3_next" or resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS
