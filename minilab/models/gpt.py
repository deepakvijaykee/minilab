import math
from dataclasses import MISSING, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.losses import causal_lm_cross_entropy
from minilab.models.transformer_utils import (
    DEFAULT_LOCAL_ATTENTION_WINDOW,
    DEFAULT_NUM_EXPERTS,
    DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL,
    DEFAULT_ROPE_BASE,
    DEFAULT_ROPE_GLOBAL_BASE,
    DEFAULT_ROPE_LOCAL_BASE,
    DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN,
    DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR,
    DEFAULT_ROPE_SCALING_FACTOR,
    DEFAULT_SPARSE_BLOCK_SIZE,
    DEFAULT_SPARSE_INDEX_DIM,
    DEFAULT_SPARSE_LOCAL_BLOCKS,
    DEFAULT_SPARSE_TOP_K_BLOCKS,
    DEFAULT_LIGHTHOUSE_NUM_LEVELS,
    DEFAULT_LIGHTHOUSE_POOLING_FACTOR,
    DEFAULT_LIGHTHOUSE_TOP_K,
    DEFAULT_TOP_K_EXPERTS,
    DEFAULT_YARN_BETA_FAST,
    DEFAULT_YARN_BETA_SLOW,
    apply_logit_softcap,
    attention_uses_gqa,
    commit_transformer_block_updates,
    require_default_unless,
    set_transformer_qk_clip_recording,
    transformer_auxiliary_loss,
    transformer_supports_qk_clip,
    validate_moe_fields,
)
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    resolve_deepseek_v4_attention,
)
from minilab.nn.connections import expand_residual_stream, reduce_residual_stream
from minilab.registry import get_attention, get_connection, get_ffn, get_norm, get_position, register_model


_ROPE_POSITIONS = {"rope", "gemma3_rope", "gemma4_rope", "yarn_rope", "qwen3_next_rope"}
_BIAS_POSITIONS = {"alibi", "t5_relative", "kerple_log", "kerple_power"}
_LOCAL_WINDOW_ATTENTIONS = {"gemma3", "gemma4", "sliding_window", "sliding_window_gqa_qknorm"}
_PARTIAL_ROPE_ATTENTIONS = {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope", "qwen3_next"}
_MTP_MODES = {"sequential", "parallel"}
_CACHE_ATTENTIONS = {
    "mha",
    "gqa",
    "mqa",
    "mha_qknorm",
    "gqa_qknorm",
    "gqa_qknorm_partial_rope",
    "gqa_qknorm_kv_tied",
}
_GPT_COMPAT_DEFAULT_FIELDS = {
    "sparse_block_size",
    "sparse_top_k_blocks",
    "sparse_local_blocks",
    "sparse_index_dim",
    "lighthouse_num_levels",
    "lighthouse_pooling_factor",
    "lighthouse_top_k",
    "mtp_mode",
    "layerskip_loss_weight",
    "layerskip_dropout",
    "layerskip_min_layer",
    "future_summary_window",
    "future_summary_loss_weight",
    "jacobi_loss_weight",
    "jacobi_iterations",
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
    norm_eps: float = 1e-6
    attention: str = "mha"
    position: str = "rope"
    norm: str = "rmsnorm"
    ffn: str = "swiglu"
    connection: str = "residual"
    connection_expansion: int = 4
    num_experts: int = DEFAULT_NUM_EXPERTS
    top_k_experts: int = DEFAULT_TOP_K_EXPERTS
    post_norm: bool = False
    rope_base: float = DEFAULT_ROPE_BASE
    rope_local_base: float = DEFAULT_ROPE_LOCAL_BASE
    rope_global_base: float = DEFAULT_ROPE_GLOBAL_BASE
    rope_scaling_factor: float = DEFAULT_ROPE_SCALING_FACTOR
    rope_original_max_seq_len: int = DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN
    rope_partial_rotary_factor: float = DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR
    yarn_beta_fast: float = DEFAULT_YARN_BETA_FAST
    yarn_beta_slow: float = DEFAULT_YARN_BETA_SLOW
    local_attention_window: int = DEFAULT_LOCAL_ATTENTION_WINDOW
    qwen3_next_full_attention_interval: int = DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL
    sparse_block_size: int = DEFAULT_SPARSE_BLOCK_SIZE
    sparse_top_k_blocks: int = DEFAULT_SPARSE_TOP_K_BLOCKS
    sparse_local_blocks: int = DEFAULT_SPARSE_LOCAL_BLOCKS
    sparse_index_dim: int = DEFAULT_SPARSE_INDEX_DIM
    lighthouse_num_levels: int = DEFAULT_LIGHTHOUSE_NUM_LEVELS
    lighthouse_pooling_factor: int = DEFAULT_LIGHTHOUSE_POOLING_FACTOR
    lighthouse_top_k: int = DEFAULT_LIGHTHOUSE_TOP_K
    attention_k_eq_v: bool = False
    per_layer_embedding_dim: int = 0
    final_logit_softcap: float = 0.0
    mtp_depth: int = 0
    mtp_loss_weight: float = 0.0
    mtp_mode: str = "sequential"
    layerskip_loss_weight: float = 0.0
    layerskip_dropout: float = 0.0
    layerskip_min_layer: int = 1
    future_summary_window: int = 0
    future_summary_loss_weight: float = 0.0
    jacobi_loss_weight: float = 0.0
    jacobi_iterations: int = 0

    @classmethod
    def from_dict(cls, d):
        require(isinstance(d, dict), f"{cls.__name__} config must be a JSON object")
        valid = set(cls.__dataclass_fields__)
        provided = set(d)
        unknown = provided - valid
        missing = valid - provided
        require(not unknown, f"Unknown {cls.__name__} fields: {sorted(unknown)}")
        remaining = missing - _GPT_COMPAT_DEFAULT_FIELDS
        require(not remaining, f"Missing {cls.__name__} fields: {sorted(remaining)}")
        data = dict(d)
        for name in sorted(missing & _GPT_COMPAT_DEFAULT_FIELDS):
            data[name] = _dataclass_default(cls, name)
        return cls(**data)

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
        require(self.norm_eps > 0, "norm_eps must be > 0")
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
        require(self.sparse_block_size > 0, "sparse_block_size must be > 0")
        require(self.sparse_top_k_blocks >= 0, "sparse_top_k_blocks must be >= 0")
        require(self.sparse_local_blocks >= 0, "sparse_local_blocks must be >= 0")
        require(self.sparse_index_dim >= 0, "sparse_index_dim must be >= 0")
        require(self.lighthouse_num_levels > 0, "lighthouse_num_levels must be > 0")
        require(self.lighthouse_pooling_factor > 1, "lighthouse_pooling_factor must be > 1")
        require(self.lighthouse_top_k > 0, "lighthouse_top_k must be > 0")
        require(self.per_layer_embedding_dim >= 0, "per_layer_embedding_dim must be >= 0")
        require(self.final_logit_softcap >= 0, "final_logit_softcap must be >= 0")
        require(self.mtp_depth >= 0, "mtp_depth must be >= 0")
        require(self.mtp_loss_weight >= 0, "mtp_loss_weight must be >= 0")
        require(self.mtp_mode in _MTP_MODES, f"Unknown mtp_mode: {self.mtp_mode!r}. Available: {sorted(_MTP_MODES)}")
        require((self.mtp_depth == 0) == (self.mtp_loss_weight == 0), (
            "mtp_depth and mtp_loss_weight must be enabled together"
        ))
        require(self.mtp_depth > 0 or self.mtp_mode == "sequential", (
            "mtp_mode only applies when mtp_depth > 0"
        ))
        require(self.layerskip_loss_weight >= 0, "layerskip_loss_weight must be >= 0")
        require(0 <= self.layerskip_dropout < 1, "layerskip_dropout must be in [0, 1)")
        require(self.layerskip_min_layer > 0, "layerskip_min_layer must be > 0")
        require(self.future_summary_window >= 0, "future_summary_window must be >= 0")
        require(self.future_summary_loss_weight >= 0, "future_summary_loss_weight must be >= 0")
        require((self.future_summary_window == 0) == (self.future_summary_loss_weight == 0), (
            "future_summary_window and future_summary_loss_weight must be enabled together"
        ))
        require(self.jacobi_loss_weight >= 0, "jacobi_loss_weight must be >= 0")
        require(self.jacobi_iterations >= 0, "jacobi_iterations must be >= 0")
        require((self.jacobi_iterations == 0) == (self.jacobi_loss_weight == 0), (
            "jacobi_iterations and jacobi_loss_weight must be enabled together"
        ))

    def _validate_attention_position_contract(self):
        if attention_uses_gqa(self.attention):
            require(self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        else:
            require(self.num_kv_heads == self.num_heads, "num_kv_heads only applies to GQA attention variants")
        if self.position in _ROPE_POSITIONS:
            head_dim = self.dim // self.num_heads
            require(head_dim % 2 == 0, "RoPE requires even head dimension")
        if self.position == "yarn_rope":
            require(
                self.yarn_beta_fast > self.yarn_beta_slow,
                "yarn_beta_fast must be > yarn_beta_slow for position='yarn_rope'",
            )
        if self.position == "sinusoidal":
            require(self.dim % 2 == 0, "sinusoidal position requires even dim")
        if self.attention in {"cosformer", "lightning", "gated_deltanet", "gated_deltanet2"}:
            require(self.position == "none", f"{self.attention} owns its positional rule; set position='none'")
        if self.attention == "gemma3":
            require(self.position == "gemma3_rope", "Gemma-style attention schedule requires position='gemma3_rope'")
        if self.attention == "gemma4":
            require(self.position == "gemma4_rope", "Gemma-style attention schedule requires position='gemma4_rope'")
        if self.attention == "qwen3_next":
            require(self.position in {"qwen3_next_rope", "yarn_rope"}, (
                "Qwen3-Next-style schedule requires position='qwen3_next_rope' or 'yarn_rope'"
            ))
        if self.attention == "lighthouse_mha":
            require(self.position not in _BIAS_POSITIONS, (
                "Lighthouse attention does not support additive attention-bias positions"
            ))
        resolved_attention = resolve_deepseek_v4_attention(self.attention, 0)
        qwen_rope_attention = (
            self.attention == "qwen3_next"
            or self.attention in _PARTIAL_ROPE_ATTENTIONS
            or resolved_attention in _PARTIAL_ROPE_ATTENTIONS
        )
        if qwen_rope_attention:
            require(self.position in _ROPE_POSITIONS, (
                "partial-RoPE attention requires a RoPE-compatible position"
            ))
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
        if self.layerskip_loss_weight > 0 or self.layerskip_dropout > 0:
            require(self.connection == "residual", "LayerSkip training currently requires residual connections")
            require(self.layerskip_min_layer < self.num_layers, (
                "layerskip_min_layer must be before the final layer"
            ))
        if self.layerskip_dropout > 0:
            require(self.layerskip_loss_weight > 0, "layerskip_dropout requires layerskip_loss_weight > 0")

    def _validate_ffn_knobs(self):
        validate_moe_fields(self)

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
        uses_learned_block = resolved_attention == "learned_block_gqa"
        uses_lighthouse = resolved_attention == "lighthouse_mha"
        require_default_unless(
            self.rope_base,
            DEFAULT_ROPE_BASE,
            self.position in {"rope", "yarn_rope"},
            "rope_base only applies to position='rope' or position='yarn_rope'",
        )
        require_default_unless(
            self.rope_local_base,
            DEFAULT_ROPE_LOCAL_BASE,
            self.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"},
            "rope_local_base only applies to Gemma/Qwen local-global RoPE positions",
        )
        require_default_unless(
            self.rope_global_base,
            DEFAULT_ROPE_GLOBAL_BASE,
            self.position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"},
            "rope_global_base only applies to Gemma/Qwen local-global RoPE positions",
        )
        require_default_unless(
            self.rope_scaling_factor,
            DEFAULT_ROPE_SCALING_FACTOR,
            self.position in {"yarn_rope", "gemma4_rope"},
            "rope_scaling_factor only applies to YaRN RoPE or Gemma 4 proportional RoPE",
        )
        require_default_unless(
            self.rope_original_max_seq_len,
            DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN,
            self.position == "yarn_rope",
            "rope_original_max_seq_len only applies to position='yarn_rope'",
        )
        require(
            (
                self.yarn_beta_fast == DEFAULT_YARN_BETA_FAST
                and self.yarn_beta_slow == DEFAULT_YARN_BETA_SLOW
            )
            or self.position == "yarn_rope",
            "yarn_beta_fast and yarn_beta_slow only apply to position='yarn_rope'",
        )
        require_default_unless(
            self.rope_partial_rotary_factor,
            DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR,
            uses_partial_rope,
            "rope_partial_rotary_factor only applies to partial-RoPE attention or Gemma/Qwen proportional RoPE",
        )
        require_default_unless(
            self.local_attention_window,
            DEFAULT_LOCAL_ATTENTION_WINDOW,
            uses_local_window,
            "local_attention_window only applies to local/sliding-window attention",
        )
        require_default_unless(
            self.qwen3_next_full_attention_interval,
            DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL,
            self.attention == "qwen3_next",
            "qwen3_next_full_attention_interval only applies to attention='qwen3_next'",
        )
        require_default_unless(
            self.sparse_block_size,
            DEFAULT_SPARSE_BLOCK_SIZE,
            uses_learned_block,
            "sparse_block_size only applies to attention='learned_block_gqa'",
        )
        require_default_unless(
            self.sparse_top_k_blocks,
            DEFAULT_SPARSE_TOP_K_BLOCKS,
            uses_learned_block,
            "sparse_top_k_blocks only applies to attention='learned_block_gqa'",
        )
        require_default_unless(
            self.sparse_local_blocks,
            DEFAULT_SPARSE_LOCAL_BLOCKS,
            uses_learned_block,
            "sparse_local_blocks only applies to attention='learned_block_gqa'",
        )
        require_default_unless(
            self.sparse_index_dim,
            DEFAULT_SPARSE_INDEX_DIM,
            uses_learned_block,
            "sparse_index_dim only applies to attention='learned_block_gqa'",
        )
        require_default_unless(
            self.lighthouse_num_levels,
            DEFAULT_LIGHTHOUSE_NUM_LEVELS,
            uses_lighthouse,
            "lighthouse_num_levels only applies to attention='lighthouse_mha'",
        )
        require_default_unless(
            self.lighthouse_pooling_factor,
            DEFAULT_LIGHTHOUSE_POOLING_FACTOR,
            uses_lighthouse,
            "lighthouse_pooling_factor only applies to attention='lighthouse_mha'",
        )
        require_default_unless(
            self.lighthouse_top_k,
            DEFAULT_LIGHTHOUSE_TOP_K,
            uses_lighthouse,
            "lighthouse_top_k only applies to attention='lighthouse_mha'",
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


def _build_transformer_norm(config, dim):
    return get_norm(config.norm)(dim, eps=config.norm_eps)


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
        elif attention == "learned_block_gqa":
            attn = attn_cls(
                config.dim,
                config.num_heads,
                config.num_kv_heads,
                config.dropout,
                block_size=config.sparse_block_size,
                top_k_blocks=config.sparse_top_k_blocks,
                local_blocks=config.sparse_local_blocks,
                index_dim=None if config.sparse_index_dim == 0 else config.sparse_index_dim,
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
    if attention == "lighthouse_mha":
        return attention, attn_cls(
            config.dim,
            config.num_heads,
            config.dropout,
            num_levels=config.lighthouse_num_levels,
            pooling_factor=config.lighthouse_pooling_factor,
            top_k=config.lighthouse_top_k,
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
    norm = _build_transformer_norm(config, config.per_layer_embedding_dim)
    return embed, projection, norm, 1.0 / math.sqrt(2.0), config.dim ** -0.5


def _future_multi_hot_targets(targets, window, vocab_size, dtype):
    require(window > 0, "future multi-hot targets require window > 0")
    target = torch.zeros((*targets.shape, vocab_size), device=targets.device, dtype=dtype)
    valid = torch.zeros(targets.shape, device=targets.device, dtype=torch.bool)
    T = targets.size(1)
    for offset in range(window):
        if T <= offset:
            break
        future = targets[:, offset:]
        future_valid = future != -100
        if not future_valid.any():
            continue
        rows = target[:, : T - offset].reshape(-1, vocab_size)
        flat_future = future.reshape(-1)
        flat_valid = future_valid.reshape(-1)
        row_ids = flat_valid.nonzero(as_tuple=False).squeeze(1)
        rows[row_ids, flat_future[row_ids]] = 1.0
        valid[:, : T - offset] |= future_valid
    return target, valid


def _multi_hot_cross_entropy(logits, targets, bag_size):
    target, valid = _future_multi_hot_targets(targets, bag_size, logits.size(-1), logits.dtype)
    counts = target.sum(dim=-1)
    valid = valid & (counts > 0)
    if not valid.any():
        return logits.sum() * 0.0
    per_position = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1) / counts.clamp_min(1.0)
    return per_position[valid].mean()


class TransformerBlock(nn.Module):
    def __init__(self, config, block_id):
        super().__init__()
        self.block_id = block_id
        self.per_layer_embedding_dim = config.per_layer_embedding_dim if block_id < config.num_layers else 0
        self.attn_norm = _build_transformer_norm(config, config.dim)
        self.ffn_norm = _build_transformer_norm(config, config.dim)
        self.post_norm = config.post_norm
        if self.post_norm:
            self.attn_post_norm = _build_transformer_norm(config, config.dim)
            self.ffn_post_norm = _build_transformer_norm(config, config.dim)
        if self.per_layer_embedding_dim > 0:
            self.per_layer_input_gate = nn.Linear(config.dim, self.per_layer_embedding_dim, bias=False)
            self.per_layer_projection = nn.Linear(self.per_layer_embedding_dim, config.dim, bias=False)
            self.post_per_layer_input_norm = _build_transformer_norm(config, config.dim)
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


def _is_gemma_global_layer(block_id):
    return (block_id + 1) % 6 == 0


def _dataclass_default(cls, name):
    default = cls.__dataclass_fields__[name].default
    require(default is not MISSING, f"{cls.__name__}.{name} has no compatibility default")
    return default


class MultiTokenPredictionModule(nn.Module):
    """DeepSeek-style sequential MTP module with shared token embedding and LM head."""

    def __init__(self, config, block_id):
        super().__init__()
        self.hidden_norm = _build_transformer_norm(config, config.dim)
        self.embed_norm = _build_transformer_norm(config, config.dim)
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
        self.ln_f = _build_transformer_norm(config, config.dim)
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
        ]) if config.mtp_mode == "sequential" else nn.ModuleList()
        self.parallel_mtp_heads = nn.ModuleList([
            nn.Linear(config.dim, config.vocab_size, bias=False)
            for _ in range(config.mtp_depth)
        ]) if config.mtp_mode == "parallel" else nn.ModuleList()
        self.future_summary_head = (
            nn.Linear(config.dim, config.vocab_size, bias=False)
            if config.future_summary_window > 0
            else None
        )

        self.pos_enc, self.local_pos_enc, self.global_pos_enc = _build_gpt_position_modules(config)

        self.apply(self._init_weights)
        if config.connection in ("hc", "mhc"):
            for block in self._optimizer_transformer_blocks():
                block.attn_conn.reset_dynamic_parameters()
                block.ffn_conn.reset_dynamic_parameters()

    def muon_auxiliary_modules(self):
        modules = [self.tok_emb, self.lm_head]
        if self.parallel_mtp_heads:
            modules.append(self.parallel_mtp_heads)
        if self.future_summary_head is not None:
            modules.append(self.future_summary_head)
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
        collect_layer_hiddens = targets is not None and self.config.layerskip_loss_weight > 0
        if collect_layer_hiddens:
            logits, _, main_hidden, layer_hiddens = self.forward_hidden(
                idx,
                return_residual=True,
                return_layer_hiddens=True,
            )
        else:
            logits, _, main_hidden = self.forward_hidden(idx, return_residual=True)
            layer_hiddens = None

        loss = None
        if targets is not None:
            loss = causal_lm_cross_entropy(logits, targets)
            loss = loss + self._mtp_loss(main_hidden, idx, targets)
            loss = loss + self._future_summary_loss(main_hidden, targets)
            loss = loss + self._layerskip_loss(layer_hiddens, targets, main_hidden)
            loss = loss + self._jacobi_forcing_loss(idx, targets)
            loss = loss + self.auxiliary_loss()

        return logits, loss

    def forward_hidden(
        self,
        idx,
        return_residual=False,
        return_layer_hiddens=False,
        apply_layerskip_dropout=True,
    ):
        _, T = idx.shape
        require(T <= self.config.max_seq_len, (
            f"GPT supports at most {self.config.max_seq_len} tokens, got {T}"
        ))
        require(not return_layer_hiddens or return_residual, (
            "return_layer_hiddens requires return_residual"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        per_layer_inputs = self._per_layer_inputs(idx, x)

        freqs_cis, attn_bias, is_causal = self._position_inputs(T)
        if self.pos_enc is not None and self.pos_enc.kind == "additive":
            x = x + self._cast_hidden(self.pos_enc(T))

        x = self.drop(x)

        if self.config.connection != "residual":
            x = expand_residual_stream(x, self.config.connection_expansion)

        layer_hiddens = []
        for block in self.blocks:
            block_freqs = self._block_freqs(block, T, freqs_cis)
            per_layer_input = per_layer_inputs[:, :, block.block_id, :] if per_layer_inputs is not None else None
            if not self._drop_layerskip_block(block, x, apply_layerskip_dropout):
                x = self._checkpointed_forward(
                    block,
                    x,
                    freqs_cis=block_freqs,
                    attn_bias=attn_bias,
                    is_causal=is_causal,
                    per_layer_input=per_layer_input,
                )
            if return_layer_hiddens:
                layer_hiddens.append(x)

        if self.config.connection != "residual":
            x = reduce_residual_stream(x)

        main_hidden = x
        x = self.ln_f(main_hidden)
        logits = apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap)

        if return_layer_hiddens:
            return logits, x, main_hidden, layer_hiddens
        if return_residual:
            return logits, x, main_hidden
        return logits, x

    def _drop_layerskip_block(self, block, x, enabled):
        if not enabled or not self.training or self.config.layerskip_dropout == 0:
            return False
        # LayerSkip uses shallower exits during training; keep the final block
        # active so the main LM loss always has a full-depth path.
        if block.block_id + 1 == self.config.num_layers:
            return False
        drop_prob = self.config.layerskip_dropout * (block.block_id + 1) / self.config.num_layers
        return bool((torch.rand((), device=x.device) < drop_prob).item())

    def forward_cached(self, idx, past_kv=None, return_hidden=False):
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

        main_hidden = x
        x = self.ln_f(main_hidden)
        logits = apply_logit_softcap(self.lm_head(x), self.config.final_logit_softcap)
        if return_hidden:
            return logits, next_kv, main_hidden
        return logits, next_kv

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
        if block.attention_name in {"gated_deltanet", "gated_deltanet2"}:
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
        if self.config.mtp_mode == "parallel":
            return self._parallel_mtp_loss(main_hidden, targets)
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

    def _parallel_mtp_loss(self, main_hidden, targets):
        mtp_losses = []
        normalized = self.ln_f(main_hidden)
        for depth, head in enumerate(self.parallel_mtp_heads, start=1):
            if targets.size(1) <= depth:
                break
            mtp_target = targets[:, depth:]
            if not (mtp_target != -100).any():
                break
            mtp_logits = apply_logit_softcap(
                head(normalized[:, : targets.size(1) - depth]),
                self.config.final_logit_softcap,
            )
            mtp_losses.append(F.cross_entropy(
                mtp_logits.reshape(-1, mtp_logits.size(-1)),
                mtp_target.reshape(-1),
                ignore_index=-100,
            ))
        if not mtp_losses:
            return main_hidden.sum() * 0.0
        return self.config.mtp_loss_weight * torch.stack(mtp_losses).mean()

    def _future_summary_loss(self, main_hidden, targets):
        if self.future_summary_head is None or self.config.future_summary_loss_weight == 0:
            return main_hidden.sum() * 0.0
        summary_logits = apply_logit_softcap(
            self.future_summary_head(self.ln_f(main_hidden)),
            self.config.final_logit_softcap,
        )
        target, valid = _future_multi_hot_targets(
            targets,
            self.config.future_summary_window,
            summary_logits.size(-1),
            summary_logits.dtype,
        )
        if not valid.any():
            return main_hidden.sum() * 0.0
        per_position = F.binary_cross_entropy_with_logits(
            summary_logits,
            target,
            reduction="none",
        ).mean(dim=-1)
        return self.config.future_summary_loss_weight * (per_position * valid).sum() / valid.sum()

    def _layerskip_loss(self, layer_hiddens, targets, main_hidden):
        if self.config.layerskip_loss_weight == 0:
            return main_hidden.sum() * 0.0
        require(layer_hiddens is not None, "LayerSkip loss requires collected layer hidden states")
        losses = []
        for layer_id, hidden in enumerate(layer_hiddens, start=1):
            if layer_id < self.config.layerskip_min_layer or layer_id == self.config.num_layers:
                continue
            logits = apply_logit_softcap(self.lm_head(self.ln_f(hidden)), self.config.final_logit_softcap)
            losses.append(causal_lm_cross_entropy(logits, targets))
        if not losses:
            return main_hidden.sum() * 0.0
        return self.config.layerskip_loss_weight * torch.stack(losses).mean()

    def _jacobi_forcing_loss(self, idx, targets):
        if self.config.jacobi_loss_weight == 0:
            return self.tok_emb.weight.sum() * 0.0
        draft = idx
        with torch.no_grad():
            for _ in range(self.config.jacobi_iterations):
                logits, _ = self.forward_hidden(draft, apply_layerskip_dropout=False)
                predicted_next = logits.argmax(dim=-1)
                draft = draft.clone()
                if draft.size(1) > 1:
                    draft[:, 1:] = predicted_next[:, :-1]
        logits, _ = self.forward_hidden(draft, apply_layerskip_dropout=False)
        return self.config.jacobi_loss_weight * causal_lm_cross_entropy(logits, targets)

    def token_superposition_loss(self, idx, targets, bag_size):
        require(bag_size > 1, "token_superposition_loss requires bag_size > 1")
        collect_layer_hiddens = self.config.layerskip_loss_weight > 0
        if collect_layer_hiddens:
            logits, _, main_hidden, layer_hiddens = self.forward_hidden(
                idx,
                return_residual=True,
                return_layer_hiddens=True,
            )
        else:
            logits, _, main_hidden = self.forward_hidden(idx, return_residual=True)
            layer_hiddens = None
        loss = _multi_hot_cross_entropy(logits, targets, bag_size)
        loss = loss + self._mtp_loss(main_hidden, idx, targets)
        loss = loss + self._future_summary_loss(main_hidden, targets)
        loss = loss + self._layerskip_loss(layer_hiddens, targets, main_hidden)
        loss = loss + self._jacobi_forcing_loss(idx, targets)
        loss = loss + self.auxiliary_loss()
        return loss

    @torch.no_grad()
    def early_exit_state(self, idx, exit_layer):
        require(self.config.connection == "residual", "early_exit_state requires residual GPT blocks")
        require(0 < exit_layer < self.config.num_layers, "exit_layer must be in [1, num_layers)")
        require(idx.dim() == 2 and idx.size(1) > 0, "early_exit_state requires non-empty (batch, seq) ids")
        require(idx.size(1) <= self.config.max_seq_len, "early_exit_state input exceeds max_seq_len")
        _, T = idx.shape
        x = self._cast_hidden(self.tok_emb(idx))
        per_layer_inputs = self._per_layer_inputs(idx, x)
        freqs_cis, attn_bias, is_causal = self._position_inputs(T)
        if self.pos_enc is not None and self.pos_enc.kind == "additive":
            x = x + self._cast_hidden(self.pos_enc(T))
        x = self.drop(x)
        for block in self.blocks[:exit_layer]:
            block_freqs = self._block_freqs(block, T, freqs_cis)
            per_layer_input = per_layer_inputs[:, :, block.block_id, :] if per_layer_inputs is not None else None
            x = block(x, freqs_cis=block_freqs, attn_bias=attn_bias, is_causal=is_causal, per_layer_input=per_layer_input)
        logits = apply_logit_softcap(self.lm_head(self.ln_f(x)), self.config.final_logit_softcap)
        return logits, x

    @torch.no_grad()
    def early_exit_logits(self, idx, exit_layer):
        logits, _ = self.early_exit_state(idx, exit_layer)
        return logits

    @torch.no_grad()
    def continue_from_hidden(self, hidden, start_layer):
        require(self.config.connection == "residual", "continue_from_hidden requires residual GPT blocks")
        require(self.config.per_layer_embedding_dim == 0, (
            "continue_from_hidden does not support per-layer embeddings"
        ))
        require(0 < start_layer < self.config.num_layers, "start_layer must be in [1, num_layers)")
        require(hidden.dim() == 3 and hidden.size(1) > 0, (
            "continue_from_hidden requires non-empty (batch, seq, dim) hidden states"
        ))
        require(hidden.size(2) == self.config.dim, "continue_from_hidden hidden dim must match GPT dim")
        require(hidden.size(1) <= self.config.max_seq_len, "continue_from_hidden input exceeds max_seq_len")

        T = hidden.size(1)
        x = hidden
        freqs_cis, attn_bias, is_causal = self._position_inputs(T)
        for block in self.blocks[start_layer:]:
            block_freqs = self._block_freqs(block, T, freqs_cis)
            x = block(x, freqs_cis=block_freqs, attn_bias=attn_bias, is_causal=is_causal)
        return apply_logit_softcap(self.lm_head(self.ln_f(x)), self.config.final_logit_softcap)

    @torch.no_grad()
    def mtp_draft_logits(self, idx):
        logits, _, main_hidden = self.forward_hidden(idx, return_residual=True)
        return self.mtp_draft_logits_from_hidden(logits, main_hidden)

    @torch.no_grad()
    def mtp_draft_logits_from_hidden(self, logits, main_hidden):
        require(self.config.mtp_mode == "parallel" and self.config.mtp_depth > 0, (
            "mtp_draft_logits_from_hidden requires parallel MTP heads"
        ))
        require(logits.dim() == 3 and main_hidden.dim() == 3, (
            "mtp_draft_logits_from_hidden requires (batch, seq, dim/vocab) tensors"
        ))
        require(logits.shape[:2] == main_hidden.shape[:2], (
            "mtp_draft_logits_from_hidden logits and hidden must share batch and seq shape"
        ))
        require(main_hidden.size(2) == self.config.dim, (
            "mtp_draft_logits_from_hidden hidden dim must match GPT dim"
        ))
        last_hidden = self.ln_f(main_hidden[:, -1:])
        draft_logits = [logits[:, -1]]
        for head in self.parallel_mtp_heads:
            draft_logits.append(apply_logit_softcap(
                head(last_hidden).squeeze(1),
                self.config.final_logit_softcap,
            ))
        return draft_logits

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
