import torch

from minilab.checks import require
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    QK_CLIP_ATTENTIONS,
    TOP_ONE_MOE_FFNS,
    resolve_deepseek_v4_attention,
)
from minilab.registry import get_position


_SIMPLE_HYBRID_POSITIONS = {"rope", "yarn_rope", "none", "sinusoidal"}
_LOCAL_WINDOW_ATTENTIONS = {"sliding_window", "sliding_window_gqa_qknorm"}
_PARTIAL_ROPE_ATTENTIONS = {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope", "qwen3_next"}
DEFAULT_ROPE_BASE = 10000.0
DEFAULT_ROPE_LOCAL_BASE = 10000.0
DEFAULT_ROPE_GLOBAL_BASE = 1000000.0
DEFAULT_ROPE_SCALING_FACTOR = 1.0
DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN = 4096
DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR = 0.25
DEFAULT_YARN_BETA_FAST = 32.0
DEFAULT_YARN_BETA_SLOW = 1.0
DEFAULT_LOCAL_ATTENTION_WINDOW = 1024
DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL = 4
DEFAULT_NUM_EXPERTS = 8
DEFAULT_TOP_K_EXPERTS = 2


def require_default_unless(value, default, condition, message):
    require(value == default or condition, message)


def validate_parallel_or_interleaved_lm_config(config, owner):
    """Shared config contract for compact HybridLM/HymbaLM-style backbones."""
    if config.num_kv_heads is None:
        config.num_kv_heads = config.num_heads
    _validate_simple_lm_fields(config)
    _validate_simple_transformer_branch(config, owner)


def validate_fixed_rope_transformer_config(config, owner):
    """Shared contract for small stacks that always use full-context RoPE."""
    if config.num_kv_heads is None:
        config.num_kv_heads = config.num_heads
    require(config.dim > 0, "dim must be > 0")
    require(config.num_layers > 0, "num_layers must be > 0")
    require(config.num_heads > 0, "num_heads must be > 0")
    require(config.num_kv_heads > 0, "num_kv_heads must be > 0")
    require(config.dim % config.num_heads == 0, "dim must be divisible by num_heads")
    require((config.dim // config.num_heads) % 2 == 0, "RoPE requires even head dimension")
    require(config.max_seq_len > 0, "max_seq_len must be > 0")
    require(0.0 <= config.dropout < 1.0, "dropout must be in [0, 1)")
    require(config.ffn_mult > 0, "ffn_mult must be > 0")
    require(config.attention not in {"cosformer", "lightning", "gated_deltanet", "gemma3", "gemma4", "qwen3_next"}, (
        f"{owner} uses a fixed RoPE transformer; choose a RoPE-compatible attention variant"
    ))
    if attention_uses_gqa(config.attention):
        require(config.num_heads % config.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
    else:
        require(config.num_kv_heads == config.num_heads, "num_kv_heads only applies to GQA attention variants")
    validate_moe_fields(config)


def _validate_simple_lm_fields(config):
    require(config.vocab_size > 0, "vocab_size must be > 0")
    require(config.dim > 0, "dim must be > 0")
    require(config.num_layers > 0, "num_layers must be > 0")
    require(config.num_heads > 0, "num_heads must be > 0")
    require(config.num_kv_heads > 0, "num_kv_heads must be > 0")
    require(config.max_seq_len > 0, "max_seq_len must be > 0")
    require(config.dim % config.num_heads == 0, "dim must be divisible by num_heads")
    require(0.0 <= config.dropout < 1.0, "dropout must be in [0, 1)")
    require(config.ffn_mult > 0, "ffn_mult must be > 0")
    require(config.rope_base > 0, "rope_base must be > 0")
    require(config.rope_scaling_factor > 0, "rope_scaling_factor must be > 0")
    require(config.rope_original_max_seq_len > 0, "rope_original_max_seq_len must be > 0")
    require(0.0 < config.rope_partial_rotary_factor <= 1.0, "rope_partial_rotary_factor must be in (0, 1]")
    require(config.yarn_beta_fast > 0, "yarn_beta_fast must be > 0")
    require(config.yarn_beta_slow > 0, "yarn_beta_slow must be > 0")
    require(config.local_attention_window > 0, "local_attention_window must be > 0")
    require(config.qwen3_next_full_attention_interval > 0, "qwen3_next_full_attention_interval must be > 0")
    require(config.final_logit_softcap >= 0, "final_logit_softcap must be >= 0")


def _validate_simple_transformer_branch(config, owner):
    require(config.position in _SIMPLE_HYBRID_POSITIONS, (
        f"{owner} supports RoPE, YaRN RoPE, sinusoidal, or no position encoding; "
        f"Gemma/Qwen local-global rotary schedules are implemented by GPT, not {owner}"
    ))
    require(config.attention not in {"gemma3", "gemma4"}, (
        f"Gemma local-global attention schedules are implemented by GPT, not {owner}"
    ))
    if attention_uses_gqa(config.attention):
        require(config.num_heads % config.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
    else:
        require(config.num_kv_heads == config.num_heads, "num_kv_heads only applies to GQA attention variants")
    if config.position in {"rope", "yarn_rope"}:
        require((config.dim // config.num_heads) % 2 == 0, "RoPE requires even head dimension")
    if config.position == "yarn_rope":
        require(
            config.yarn_beta_fast > config.yarn_beta_slow,
            "yarn_beta_fast must be > yarn_beta_slow for position='yarn_rope'",
        )
    if config.position == "sinusoidal":
        require(config.dim % 2 == 0, "sinusoidal position requires even dim")
    if config.attention in {"cosformer", "lightning", "gated_deltanet"}:
        require(config.position == "none", f"{config.attention} owns its positional rule; set position='none'")
    if config.attention == "qwen3_next":
        require(config.position == "yarn_rope", f"Qwen3-Next-style {owner} requires position='yarn_rope'")

    resolved_attention = resolve_deepseek_v4_attention(config.attention, 0)
    uses_local_window = config.attention in _LOCAL_WINDOW_ATTENTIONS or resolved_attention in _LOCAL_WINDOW_ATTENTIONS
    uses_partial_rope = config.attention in _PARTIAL_ROPE_ATTENTIONS or resolved_attention in _PARTIAL_ROPE_ATTENTIONS
    if uses_partial_rope:
        require(config.position in {"rope", "yarn_rope"}, (
            f"partial-RoPE attention in {owner} requires a RoPE-compatible position"
        ))
    require_default_unless(
        config.rope_base,
        DEFAULT_ROPE_BASE,
        config.position in {"rope", "yarn_rope"},
        "rope_base only applies to position='rope' or position='yarn_rope'",
    )
    require_default_unless(
        config.rope_scaling_factor,
        DEFAULT_ROPE_SCALING_FACTOR,
        config.position == "yarn_rope",
        "rope_scaling_factor only applies to position='yarn_rope'",
    )
    require_default_unless(
        config.rope_original_max_seq_len,
        DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN,
        config.position == "yarn_rope",
        "rope_original_max_seq_len only applies to position='yarn_rope'",
    )
    require(
        (
            config.yarn_beta_fast == DEFAULT_YARN_BETA_FAST
            and config.yarn_beta_slow == DEFAULT_YARN_BETA_SLOW
        )
        or config.position == "yarn_rope",
        "yarn_beta_fast and yarn_beta_slow only apply to position='yarn_rope'",
    )
    require_default_unless(
        config.local_attention_window,
        DEFAULT_LOCAL_ATTENTION_WINDOW,
        uses_local_window,
        "local_attention_window only applies to local/sliding-window attention",
    )
    require_default_unless(
        config.rope_partial_rotary_factor,
        DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR,
        uses_partial_rope,
        "rope_partial_rotary_factor only applies to partial-RoPE attention",
    )
    require_default_unless(
        config.qwen3_next_full_attention_interval,
        DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL,
        config.attention == "qwen3_next",
        "qwen3_next_full_attention_interval only applies to attention='qwen3_next'",
    )
    validate_moe_fields(config)


def validate_moe_fields(config):
    if config.ffn in MOE_FFNS:
        require(config.num_experts > 0, "num_experts must be > 0")
        require(1 <= config.top_k_experts <= config.num_experts, "top_k_experts must be in [1, num_experts]")
        if config.ffn in TOP_ONE_MOE_FFNS:
            require(config.top_k_experts == 1, f"{config.ffn} requires top_k_experts=1")
    else:
        require(
            config.num_experts == DEFAULT_NUM_EXPERTS and config.top_k_experts == DEFAULT_TOP_K_EXPERTS,
            "num_experts and top_k_experts only apply to MoE FFNs",
        )


def attention_uses_gqa(attention):
    return attention in {"gemma3", "gemma4", "qwen3_next"} or resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS


def transformer_auxiliary_loss(blocks, ffn_name, reference):
    loss = reference.sum() * 0.0
    if ffn_name not in MOE_FFNS:
        return loss
    for block in blocks:
        loss = loss + block.ffn.aux_loss
    return loss


def set_transformer_qk_clip_recording(blocks, enabled):
    for block in blocks:
        if block.attention_name in QK_CLIP_ATTENTIONS:
            block.attn.set_qk_clip_recording(enabled)


def transformer_supports_qk_clip(blocks):
    return any(block.attention_name in QK_CLIP_ATTENTIONS for block in blocks)


def commit_transformer_block_updates(blocks, ffn_name, qk_clip_threshold, qk_clip_balance):
    if ffn_name == "aux_free_moe":
        for block in blocks:
            block.ffn.commit_routing_bias_update()
    if qk_clip_threshold <= 0:
        return
    for block in blocks:
        if block.attention_name in QK_CLIP_ATTENTIONS:
            block.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)


def build_rope_or_simple_position(config, owner):
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
        f"{owner} unsupported position variant: {config.position}. "
        f"Gemma/Qwen local-global rotary schedules are implemented by GPT, not {owner}."
    ))


def apply_simple_position(pos_enc, x, seq_len, owner):
    freqs_cis = None
    attn_bias = None
    is_causal = True
    if pos_enc is None:
        return x, freqs_cis, attn_bias, is_causal
    if pos_enc.kind == "rotary":
        freqs_cis = pos_enc(seq_len)
    elif pos_enc.kind == "additive":
        x = x + pos_enc(seq_len).to(device=x.device, dtype=x.dtype)
    else:
        require(pos_enc.kind == "none", f"{owner} unsupported position kind: {pos_enc.kind}")
    return x, freqs_cis, attn_bias, is_causal


def attention_freqs_for_attention(attention_name, freqs_cis):
    if attention_name == "gated_deltanet":
        return None
    return freqs_cis


def apply_logit_softcap(logits, softcap):
    if softcap <= 0:
        return logits
    return torch.tanh(logits / softcap) * softcap
