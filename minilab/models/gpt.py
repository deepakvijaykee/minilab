import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.losses import causal_lm_cross_entropy
from minilab.models.transformer_utils import (
    apply_logit_softcap,
    commit_transformer_block_updates,
    set_transformer_qk_clip_recording,
    transformer_auxiliary_loss,
    transformer_supports_qk_clip,
)
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    TOP_ONE_MOE_FFNS,
    resolve_deepseek_v4_attention,
)
from minilab.nn.connections import expand_residual_stream, reduce_residual_stream
from minilab.registry import get_attention, get_connection, get_ffn, get_norm, get_position, register_model


_ROPE_POSITIONS = {"rope", "gemma3_rope", "gemma4_rope", "yarn_rope", "qwen3_next_rope"}
_BIAS_POSITIONS = {"alibi", "t5_relative", "kerple_log", "kerple_power"}
_DEFAULT_ROPE_BASE = 10000.0
_DEFAULT_ROPE_LOCAL_BASE = 10000.0
_DEFAULT_ROPE_GLOBAL_BASE = 1000000.0
_DEFAULT_ROPE_SCALING_FACTOR = 1.0
_DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN = 4096
_DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR = 0.25
_DEFAULT_YARN_BETA_FAST = 32.0
_DEFAULT_YARN_BETA_SLOW = 1.0
_DEFAULT_LOCAL_ATTENTION_WINDOW = 1024
_DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL = 4
_DEFAULT_NUM_EXPERTS = 8
_DEFAULT_TOP_K_EXPERTS = 2
_LOCAL_WINDOW_ATTENTIONS = {"gemma3", "gemma4", "sliding_window", "sliding_window_gqa_qknorm"}
_PARTIAL_ROPE_ATTENTIONS = {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope", "qwen3_next"}
_CACHE_ATTENTIONS = {
    "mha",
    "gqa",
    "mqa",
    "mha_qknorm",
    "gqa_qknorm",
    "gqa_qknorm_partial_rope",
    "gqa_qknorm_kv_tied",
}


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
    num_experts: int = _DEFAULT_NUM_EXPERTS
    top_k_experts: int = _DEFAULT_TOP_K_EXPERTS
    post_norm: bool = False
    rope_base: float = _DEFAULT_ROPE_BASE
    rope_local_base: float = _DEFAULT_ROPE_LOCAL_BASE
    rope_global_base: float = _DEFAULT_ROPE_GLOBAL_BASE
    rope_scaling_factor: float = _DEFAULT_ROPE_SCALING_FACTOR
    rope_original_max_seq_len: int = _DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN
    rope_partial_rotary_factor: float = _DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR
    yarn_beta_fast: float = _DEFAULT_YARN_BETA_FAST
    yarn_beta_slow: float = _DEFAULT_YARN_BETA_SLOW
    local_attention_window: int = _DEFAULT_LOCAL_ATTENTION_WINDOW
    qwen3_next_full_attention_interval: int = _DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL
    attention_k_eq_v: bool = False
    per_layer_embedding_dim: int = 0
    final_logit_softcap: float = 0.0
    mtp_depth: int = 0
    mtp_loss_weight: float = 0.0

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        self._validate_core_fields()
        self._validate_attention_position_contract()
        self._reject_unused_variant_knobs()
        self._validate_connection_knobs()
        self._validate_ffn_knobs()

    def _validate_core_fields(self):
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
        require(self.rope_base > 0, "rope_base must be > 0")
        require(self.rope_local_base > 0, "rope_local_base must be > 0")
        require(self.rope_global_base > 0, "rope_global_base must be > 0")
        require(self.rope_scaling_factor > 0, "rope_scaling_factor must be > 0")
        require(self.rope_original_max_seq_len > 0, "rope_original_max_seq_len must be > 0")
        require(0.0 < self.rope_partial_rotary_factor <= 1.0, "rope_partial_rotary_factor must be in (0, 1]")
        require(self.yarn_beta_fast > 0, "yarn_beta_fast must be > 0")
        require(self.yarn_beta_slow > 0, "yarn_beta_slow must be > 0")
        require(self.local_attention_window > 0, "local_attention_window must be > 0")
        require(self.qwen3_next_full_attention_interval > 0, "qwen3_next_full_attention_interval must be > 0")
        require(self.per_layer_embedding_dim >= 0, "per_layer_embedding_dim must be >= 0")
        require(self.final_logit_softcap >= 0, "final_logit_softcap must be >= 0")
        require(self.mtp_depth >= 0, "mtp_depth must be >= 0")
        require(self.mtp_loss_weight >= 0, "mtp_loss_weight must be >= 0")
        require((self.mtp_depth == 0) == (self.mtp_loss_weight == 0), (
            "mtp_depth and mtp_loss_weight must be enabled together"
        ))

    def _validate_attention_position_contract(self):
        if _attention_uses_gqa(self.attention):
            require(self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        else:
            require(self.num_kv_heads == self.num_heads, "num_kv_heads only applies to GQA attention variants")
        if self.position in _ROPE_POSITIONS:
            head_dim = self.dim // self.num_heads
            require(head_dim % 2 == 0, "RoPE requires even head dimension")
        if self.position == "sinusoidal":
            require(self.dim % 2 == 0, "sinusoidal position requires even dim")
        if self.attention == "cosformer":
            require(self.position == "none", "cosFormer owns its positional reweighting; set position='none'")
        if self.attention == "lightning":
            require(self.position == "none", "Lightning Attention owns positional decay; set position='none'")
        if self.attention == "gemma3":
            require(self.position == "gemma3_rope", "Gemma-style attention schedule requires position='gemma3_rope'")
        if self.attention == "gemma4":
            require(self.position == "gemma4_rope", "Gemma-style attention schedule requires position='gemma4_rope'")
        if self.attention == "qwen3_next":
            require(self.position in {"qwen3_next_rope", "yarn_rope"}, (
                "Qwen3-Next-style schedule requires position='qwen3_next_rope' or 'yarn_rope'"
            ))
        resolved_attention = resolve_deepseek_v4_attention(self.attention, 0)
        qwen_rope_attention = (
            self.attention == "qwen3_next"
            or self.attention in _PARTIAL_ROPE_ATTENTIONS
            or resolved_attention in _PARTIAL_ROPE_ATTENTIONS
        )
        require(self.position != "gemma3_rope" or self.attention == "gemma3", (
            "Gemma/Qwen local-global position='gemma3_rope' requires attention='gemma3'"
        ))
        require(self.position != "gemma4_rope" or self.attention == "gemma4", (
            "Gemma/Qwen local-global position='gemma4_rope' requires attention='gemma4'"
        ))
        require(self.position != "qwen3_next_rope" or qwen_rope_attention, (
            "Gemma/Qwen local-global position='qwen3_next_rope' requires Qwen3-Next or partial-RoPE attention"
        ))
        if self.position in {"gemma4_rope", "qwen3_next_rope"}:
            head_dim = self.dim // self.num_heads
            require(int(self.rope_partial_rotary_factor * head_dim // 2) > 0, (
                "Gemma/Qwen proportional RoPE must rotate at least one frequency; "
                "increase head dimension or rope_partial_rotary_factor"
            ))

    def _validate_connection_knobs(self):
        if self.per_layer_embedding_dim > 0:
            require(self.connection == "residual", "per-layer embeddings require residual connections")

    def _validate_ffn_knobs(self):
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

    def _reject_unused_variant_knobs(self):
        resolved_attention = resolve_deepseek_v4_attention(self.attention, 0)
        uses_local_window = (
            self.attention in _LOCAL_WINDOW_ATTENTIONS
            or resolved_attention in {"sliding_window", "sliding_window_gqa_qknorm"}
        )
        uses_partial_rope = (
            self.attention in _PARTIAL_ROPE_ATTENTIONS
            or resolved_attention in _PARTIAL_ROPE_ATTENTIONS
            or self.position in {"gemma4_rope", "qwen3_next_rope"}
        )
        require(
            self.rope_base == _DEFAULT_ROPE_BASE or self.position in {"rope", "yarn_rope"},
            "rope_base only applies to position='rope' or position='yarn_rope'",
        )
        require(
            self.rope_local_base == _DEFAULT_ROPE_LOCAL_BASE
            or self.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"},
            "rope_local_base only applies to Gemma/Qwen local-global RoPE positions",
        )
        require(
            self.rope_global_base == _DEFAULT_ROPE_GLOBAL_BASE
            or self.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"},
            "rope_global_base only applies to Gemma/Qwen local-global RoPE positions",
        )
        require(
            self.rope_scaling_factor == _DEFAULT_ROPE_SCALING_FACTOR
            or self.position in {"yarn_rope", "gemma4_rope"},
            "rope_scaling_factor only applies to YaRN RoPE or Gemma 4 proportional RoPE",
        )
        require(
            self.rope_original_max_seq_len == _DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN
            or self.position == "yarn_rope",
            "rope_original_max_seq_len only applies to position='yarn_rope'",
        )
        require(
            self.yarn_beta_fast == _DEFAULT_YARN_BETA_FAST and self.yarn_beta_slow == _DEFAULT_YARN_BETA_SLOW
            or self.position == "yarn_rope",
            "yarn_beta_fast and yarn_beta_slow only apply to position='yarn_rope'",
        )
        require(
            self.rope_partial_rotary_factor == _DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR or uses_partial_rope,
            "rope_partial_rotary_factor only applies to partial-RoPE attention or Gemma/Qwen proportional RoPE",
        )
        require(
            self.local_attention_window == _DEFAULT_LOCAL_ATTENTION_WINDOW or uses_local_window,
            "local_attention_window only applies to local/sliding-window attention",
        )
        require(
            self.qwen3_next_full_attention_interval == _DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL
            or self.attention == "qwen3_next",
            "qwen3_next_full_attention_interval only applies to attention='qwen3_next'",
        )
        require(
            not self.attention_k_eq_v or self.attention == "gemma4",
            "attention_k_eq_v only applies to attention='gemma4'",
        )


PRESETS = {
    "gpt-tiny": {"dim": 128, "num_layers": 4, "num_heads": 4, "max_seq_len": 256},
    "gpt-small": {"dim": 256, "num_layers": 6, "num_heads": 8, "max_seq_len": 512},
    "gpt-medium": {"dim": 512, "num_layers": 12, "num_heads": 8, "max_seq_len": 1024},
    "gpt-large": {"dim": 768, "num_layers": 24, "num_heads": 12, "max_seq_len": 2048},
}


def gpt_preset(name, vocab_size):
    require(name in PRESETS, f"Unknown preset: {name}. Available: {list(PRESETS.keys())}")
    preset = PRESETS[name]
    return GPTConfig(
        vocab_size=vocab_size,
        dim=preset["dim"],
        num_layers=preset["num_layers"],
        num_heads=preset["num_heads"],
        max_seq_len=preset["max_seq_len"],
    )


def _build_transformer_ffn(config):
    ffn_hidden = int(config.dim * config.ffn_mult)
    if config.ffn in MOE_FFNS:
        return get_ffn(config.ffn)(
            config.dim,
            ffn_hidden,
            num_experts=config.num_experts,
            top_k=config.top_k_experts,
        )
    return get_ffn(config.ffn)(config.dim, ffn_hidden)


def _build_transformer_attention(config, block_id):
    attention = _resolve_attention_name(config, block_id)
    attn_cls = get_attention(attention)
    if attention in GQA_ATTENTIONS:
        if attention in {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope"}:
            attn = attn_cls(
                config.dim,
                config.num_heads,
                config.num_kv_heads,
                config.dropout,
                rope_fraction=config.rope_partial_rotary_factor,
            )
        else:
            attn = attn_cls(config.dim, config.num_heads, config.num_kv_heads, config.dropout)
        if attention == "sliding_window_gqa_qknorm":
            attn.window_size = config.local_attention_window
        return attention, attn
    if attention == "sliding_window":
        return attention, attn_cls(
            config.dim,
            config.num_heads,
            config.dropout,
            window_size=config.local_attention_window,
        )
    return attention, attn_cls(config.dim, config.num_heads, config.dropout)


def _build_gpt_position_modules(config):
    head_dim = config.dim // config.num_heads
    if config.position == "rope":
        return get_position("rope")(head_dim, config.max_seq_len, base=config.rope_base), None, None
    if config.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"}:
        local_pos_enc = get_position("rope")(head_dim, config.max_seq_len, base=config.rope_local_base)
        if config.position == "gemma4_rope":
            global_pos_enc = get_position("proportional_rope")(
                head_dim,
                config.max_seq_len,
                base=config.rope_global_base,
                partial_rotary_factor=config.rope_partial_rotary_factor,
                factor=config.rope_scaling_factor,
            )
        elif config.position == "qwen3_next_rope":
            global_pos_enc = get_position("proportional_rope")(
                head_dim,
                config.max_seq_len,
                base=config.rope_global_base,
                partial_rotary_factor=config.rope_partial_rotary_factor,
            )
        else:
            global_pos_enc = get_position("rope")(head_dim, config.max_seq_len, base=config.rope_global_base)
        return None, local_pos_enc, global_pos_enc
    if config.position == "yarn_rope":
        pos_enc = get_position("yarn_rope")(
            head_dim,
            config.max_seq_len,
            base=config.rope_base,
            factor=config.rope_scaling_factor,
            original_max_seq_len=config.rope_original_max_seq_len,
            beta_fast=config.yarn_beta_fast,
            beta_slow=config.yarn_beta_slow,
        )
        return pos_enc, None, None
    if config.position in _BIAS_POSITIONS:
        return get_position(config.position)(config.num_heads, config.max_seq_len), None, None
    return get_position(config.position)(config.dim, config.max_seq_len), None, None


def _build_per_layer_embeddings(config):
    if config.per_layer_embedding_dim == 0:
        return None, None, None, None, None
    embed = nn.Embedding(config.vocab_size, config.num_layers * config.per_layer_embedding_dim)
    projection = nn.Linear(config.dim, config.num_layers * config.per_layer_embedding_dim, bias=False)
    norm = get_norm(config.norm)(config.per_layer_embedding_dim)
    return embed, projection, norm, 1.0 / math.sqrt(2.0), config.dim ** -0.5


class TransformerBlock(nn.Module):
    def __init__(self, config, block_id):
        super().__init__()
        self.block_id = block_id
        self.per_layer_embedding_dim = config.per_layer_embedding_dim if block_id < config.num_layers else 0
        self.attn_norm = get_norm(config.norm)(config.dim)
        self.ffn_norm = get_norm(config.norm)(config.dim)
        self.post_norm = config.post_norm
        if self.post_norm:
            self.attn_post_norm = get_norm(config.norm)(config.dim)
            self.ffn_post_norm = get_norm(config.norm)(config.dim)
        if self.per_layer_embedding_dim > 0:
            self.per_layer_input_gate = nn.Linear(config.dim, self.per_layer_embedding_dim, bias=False)
            self.per_layer_projection = nn.Linear(self.per_layer_embedding_dim, config.dim, bias=False)
            self.post_per_layer_input_norm = get_norm(config.norm)(config.dim)
        self.drop = nn.Dropout(config.dropout)

        self.ffn = _build_transformer_ffn(config)
        self.attention_name, self.attn = _build_transformer_attention(config, block_id)
        self.uses_residual_connection = config.connection == "residual"

        conn_cls = get_connection(config.connection)
        if config.connection == "residual":
            self.attn_conn = conn_cls(config.dim)
            self.ffn_conn = conn_cls(config.dim)
        else:
            self.attn_conn = conn_cls(config.dim, config.connection_expansion, layer_id=2 * block_id)
            self.ffn_conn = conn_cls(config.dim, config.connection_expansion, layer_id=2 * block_id + 1)

    def forward(self, x, freqs_cis=None, attn_bias=None, is_causal=False, per_layer_input=None):
        def attn_branch(h):
            out = self.attn(self.attn_norm(h), freqs_cis, attn_bias, is_causal)
            if self.post_norm:
                out = self.attn_post_norm(out)
            return self.drop(out)

        def ffn_branch(h):
            out = self.ffn(self.ffn_norm(h))
            if self.post_norm:
                out = self.ffn_post_norm(out)
            return self.drop(out)

        x = self.attn_conn(x, attn_branch)
        x = self.ffn_conn(x, ffn_branch)
        if self.per_layer_embedding_dim > 0:
            require(per_layer_input is not None, "per-layer embedding block requires per_layer_input")
            residual = x
            ple = F.gelu(self.per_layer_input_gate(x), approximate="tanh") * per_layer_input
            x = residual + self.post_per_layer_input_norm(self.per_layer_projection(ple))
        return x

    def forward_cached(self, x, freqs_cis=None, is_causal=True, past_kv=None):
        require(self.uses_residual_connection, "cached GPT blocks require residual connections")
        require(self.per_layer_embedding_dim == 0, "cached GPT blocks do not support per-layer embeddings")
        require(self.attention_name in _CACHE_ATTENTIONS, (
            f"cached GPT generation does not support attention={self.attention_name!r}"
        ))
        attn_out, next_kv = self.attn(
            self.attn_norm(x),
            freqs_cis=freqs_cis,
            attn_bias=None,
            is_causal=is_causal,
            past_kv=past_kv,
            return_kv=True,
        )
        if self.post_norm:
            attn_out = self.attn_post_norm(attn_out)
        x = x + self.drop(attn_out)

        ffn_out = self.ffn(self.ffn_norm(x))
        if self.post_norm:
            ffn_out = self.ffn_post_norm(ffn_out)
        x = x + self.drop(ffn_out)
        return x, next_kv


def _resolve_attention_name(config, block_id):
    if config.attention == "gemma3":
        return "gqa_qknorm" if _is_gemma_global_layer(block_id) else "sliding_window_gqa_qknorm"
    if config.attention == "gemma4":
        if _is_gemma_global_layer(block_id):
            return "gqa_qknorm_kv_tied" if config.attention_k_eq_v else "gqa_qknorm"
        return "sliding_window_gqa_qknorm"
    if config.attention == "qwen3_next":
        if (block_id + 1) % config.qwen3_next_full_attention_interval == 0:
            return "gated_gqa_qknorm_partial_rope"
        return "gated_deltanet"
    return resolve_deepseek_v4_attention(config.attention, block_id)


def _attention_uses_gqa(attention):
    if attention in {"gemma3", "gemma4", "qwen3_next"}:
        return True
    return resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS


def _is_gemma_global_layer(block_id):
    return (block_id + 1) % 6 == 0


class MultiTokenPredictionModule(nn.Module):
    """DeepSeek-style sequential MTP module with shared token embedding and LM head."""

    def __init__(self, config, block_id):
        super().__init__()
        self.hidden_norm = get_norm(config.norm)(config.dim)
        self.embed_norm = get_norm(config.norm)(config.dim)
        self.proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.block = TransformerBlock(config, block_id)

    def forward(self, hidden, future_emb, freqs_cis=None, attn_bias=None, is_causal=True):
        x = self.proj(torch.cat([self.hidden_norm(hidden), self.embed_norm(future_emb)], dim=-1))
        if not self.block.uses_residual_connection:
            x = expand_residual_stream(x, self.block.attn_conn.expansion)
            x = self.block(x, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)
            return reduce_residual_stream(x)
        return self.block(x, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)


@register_model("gpt")
class GPT(BaseModel):
    config_class = GPTConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.num_layers)])
        self.ln_f = get_norm(config.norm)(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        (
            self.embed_tokens_per_layer,
            self.per_layer_model_projection,
            self.per_layer_projection_norm,
            self.per_layer_input_scale,
            self.per_layer_model_projection_scale,
        ) = _build_per_layer_embeddings(config)
        self.mtp_modules = nn.ModuleList([
            MultiTokenPredictionModule(config, config.num_layers + i)
            for i in range(config.mtp_depth)
        ])

        self.pos_enc, self.local_pos_enc, self.global_pos_enc = _build_gpt_position_modules(config)

        self.apply(self._init_weights)
        if config.connection in ("hc", "mhc"):
            for block in self._optimizer_transformer_blocks():
                block.attn_conn.reset_dynamic_parameters()
                block.ffn_conn.reset_dynamic_parameters()

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.embed_tokens_per_layer is not None:
            modules.append(self.embed_tokens_per_layer)
        if self.pos_enc is not None and any(p.requires_grad for p in self.pos_enc.parameters()):
            modules.append(self.pos_enc)
        if self.config.connection != "residual":
            for block in self._optimizer_transformer_blocks():
                modules.extend((block.attn_conn, block.ffn_conn))
        return tuple(modules)

    def set_qk_clip_recording(self, enabled):
        set_transformer_qk_clip_recording(self._optimizer_transformer_blocks(), enabled)

    def supports_qk_clip(self):
        return transformer_supports_qk_clip(self.blocks)

    def auxiliary_loss(self):
        return transformer_auxiliary_loss(self.blocks, self.config.ffn, next(self.parameters()))

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_transformer_block_updates(
            self._optimizer_transformer_blocks(),
            self.config.ffn,
            qk_clip_threshold,
            qk_clip_balance,
        )

    def forward(self, idx, targets=None):
        logits, _, main_hidden = self.forward_hidden(idx, return_residual=True)

        loss = None
        if targets is not None:
            loss = causal_lm_cross_entropy(logits, targets)
            loss = loss + self._mtp_loss(main_hidden, idx, targets)
            loss = loss + self.auxiliary_loss()

        return logits, loss

    def forward_hidden(self, idx, return_residual=False):
        _, T = idx.shape
        require(T <= self.config.max_seq_len, (
            f"GPT supports at most {self.config.max_seq_len} tokens, got {T}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        per_layer_inputs = self._per_layer_inputs(idx, x)

        freqs_cis, attn_bias, is_causal = self._position_inputs(T)
        if self.pos_enc is not None and self.pos_enc.kind == "additive":
            x = x + self._cast_hidden(self.pos_enc(T))

        x = self.drop(x)

        if self.config.connection != "residual":
            x = expand_residual_stream(x, self.config.connection_expansion)

        for block in self.blocks:
            block_freqs = self._block_freqs(block, T, freqs_cis)
            per_layer_input = per_layer_inputs[:, :, block.block_id, :] if per_layer_inputs is not None else None
            x = self._checkpointed_forward(
                block,
                x,
                freqs_cis=block_freqs,
                attn_bias=attn_bias,
                is_causal=is_causal,
                per_layer_input=per_layer_input,
            )

        if self.config.connection != "residual":
            x = reduce_residual_stream(x)

        main_hidden = x
        x = self.ln_f(main_hidden)
        logits = apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap)

        if return_residual:
            return logits, x, main_hidden
        return logits, x

    def forward_cached(self, idx, past_kv=None):
        require(not self.training, "forward_cached expects model.eval() at the call boundary")
        require(idx.dim() == 2, "forward_cached idx must have shape (batch, seq)")
        require(idx.size(1) > 0, "forward_cached requires a non-empty input")
        require(self.config.connection == "residual", "forward_cached currently supports residual GPT blocks")
        require(self.config.per_layer_embedding_dim == 0, "forward_cached does not support per-layer embeddings")
        require(not self._gradient_checkpointing, "forward_cached does not use gradient checkpointing")
        require(self.pos_enc is None or self.pos_enc.kind in {"rotary", "none"}, (
            "forward_cached supports rotary or no positional encoding"
        ))
        for block in self.blocks:
            require(block.attention_name in _CACHE_ATTENTIONS, (
                f"forward_cached does not support attention={block.attention_name!r}"
            ))

        B, T = idx.shape
        past_len = 0
        if past_kv is None:
            past_kv = [None] * len(self.blocks)
        else:
            require(len(past_kv) == len(self.blocks), "past_kv must have one entry per GPT block")
            past_len = past_kv[0][0].size(2)
            for key, value in past_kv:
                require(key.size(0) == B and value.size(0) == B, "past_kv batch size must match idx")
                require(key.size(2) == past_len and value.size(2) == past_len, (
                    "all cached GPT layers must have the same sequence length"
                ))
        require(past_len + T <= self.config.max_seq_len, (
            f"GPT cache supports at most {self.config.max_seq_len} tokens, got {past_len + T}"
        ))

        x = self._cast_hidden(self.tok_emb(idx))
        freqs_cis, attn_bias, is_causal = self._position_inputs(T, offset=past_len)
        require(attn_bias is None, "forward_cached does not support additive attention bias")
        x = self.drop(x)

        next_kv = []
        for block, block_past in zip(self.blocks, past_kv, strict=True):
            block_freqs = self._block_freqs(block, T, freqs_cis, offset=past_len)
            x, block_kv = block.forward_cached(
                x,
                freqs_cis=block_freqs,
                is_causal=is_causal,
                past_kv=block_past,
            )
            next_kv.append(block_kv)

        x = self.ln_f(x)
        return apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap), next_kv

    def _position_inputs(self, seq_len, offset=0):
        freqs_cis, attn_bias, is_causal = None, None, True
        if self.pos_enc is not None:
            if self.pos_enc.kind == "rotary":
                freqs_cis = self.pos_enc(seq_len, offset=offset)
            elif self.pos_enc.kind == "bias":
                require(offset == 0, "relative position bias does not support cached offset scoring")
                attn_bias = self.pos_enc(seq_len).unsqueeze(0)
                is_causal = False
            else:
                require(self.pos_enc.kind in {"additive", "none"}, f"Unknown position kind: {self.pos_enc.kind}")
                require(offset == 0 or self.pos_enc.kind == "none", (
                    "cached offset scoring supports rotary or no positional encoding"
                ))
        return freqs_cis, attn_bias, is_causal

    def _block_freqs(self, block, seq_len, default_freqs=None, offset=0):
        if block.attention_name == "gated_deltanet":
            return None
        if self.config.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"}:
            if _is_global_attention_name(block.attention_name):
                return self.global_pos_enc(seq_len, offset=offset)
            return self.local_pos_enc(seq_len, offset=offset)
        return default_freqs

    def _per_layer_inputs(self, idx, inputs_embeds):
        if self.config.per_layer_embedding_dim == 0:
            return None
        B, T = idx.shape
        D = self.config.per_layer_embedding_dim
        token_inputs = self.embed_tokens_per_layer(idx).reshape(B, T, self.config.num_layers, D)
        token_inputs = token_inputs * math.sqrt(D)
        projected = self.per_layer_model_projection(inputs_embeds) * self.per_layer_model_projection_scale
        projected = projected.reshape(B, T, self.config.num_layers, D)
        projected = self.per_layer_projection_norm(projected)
        return (projected + token_inputs) * self.per_layer_input_scale

    def _mtp_position_inputs(self, block, seq_len):
        freqs_cis, attn_bias, is_causal = self._position_inputs(seq_len)
        return self._block_freqs(block, seq_len, freqs_cis), attn_bias, is_causal

    def _mtp_logits(self, hidden):
        return apply_logit_softcap(self.lm_head(self.ln_f(hidden)), self.config.final_logit_softcap)

    def _mtp_loss(self, main_hidden, idx, targets):
        if self.config.mtp_depth == 0 or self.config.mtp_loss_weight == 0:
            return main_hidden.sum() * 0.0
        mtp_losses = []
        mtp_aux = main_hidden.sum() * 0.0
        hidden = main_hidden
        for depth, module in enumerate(self.mtp_modules, start=1):
            if idx.size(1) <= depth:
                break
            mtp_target = targets[:, depth:]
            if not (mtp_target != -100).any():
                break
            current_hidden = hidden[:, : idx.size(1) - depth]
            future_emb = self._cast_hidden(self.tok_emb(idx[:, depth:]))
            seq_len = current_hidden.size(1)
            freqs_cis, attn_bias, is_causal = self._mtp_position_inputs(module.block, seq_len)
            hidden = module(current_hidden, future_emb, freqs_cis=freqs_cis, attn_bias=attn_bias, is_causal=is_causal)
            mtp_logits = self._mtp_logits(hidden)
            mtp_losses.append(F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                mtp_target.reshape(-1),
                ignore_index=-100,
            ))
            if self.config.ffn in MOE_FFNS:
                mtp_aux = mtp_aux + module.block.ffn.aux_loss
        if not mtp_losses:
            return main_hidden.sum() * 0.0
        return self.config.mtp_loss_weight * torch.stack(mtp_losses).mean() + mtp_aux

    def _optimizer_transformer_blocks(self):
        return [*self.blocks, *(module.block for module in self.mtp_modules)]

    def supports_kv_cache(self):
        if self.training:
            return False
        if self.config.connection != "residual" or self.config.per_layer_embedding_dim > 0:
            return False
        if self.pos_enc is not None and self.pos_enc.kind not in {"rotary", "none"}:
            return False
        return all(block.attention_name in _CACHE_ATTENTIONS for block in self.blocks)


def _is_global_attention_name(attention_name):
    return attention_name in {
        "gqa_qknorm",
        "gqa_qknorm_partial_rope",
        "gqa_qknorm_kv_tied",
        "gated_gqa_qknorm",
        "gated_gqa_qknorm_partial_rope",
    }
