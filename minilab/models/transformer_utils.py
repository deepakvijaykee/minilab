import torch

from minilab.checks import require, require_finite_fields, require_integer_fields
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    QK_CLIP_ATTENTIONS,
    TOP_ONE_MOE_FFNS,
    resolve_deepseek_v4_attention,
)


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
DEFAULT_SPARSE_BLOCK_SIZE = 128
DEFAULT_SPARSE_TOP_K_BLOCKS = 32
DEFAULT_SPARSE_LOCAL_BLOCKS = 1
DEFAULT_SPARSE_INDEX_DIM = 0
DEFAULT_LIGHTHOUSE_NUM_LEVELS = 3
DEFAULT_LIGHTHOUSE_POOLING_FACTOR = 2
DEFAULT_LIGHTHOUSE_TOP_K = 32
DEFAULT_NUM_EXPERTS = 8
DEFAULT_TOP_K_EXPERTS = 2


def require_default_unless(value, default, condition, message):
    require(value == default or condition, message)




def validate_fixed_rope_transformer_config(config, owner):
    """Shared contract for small stacks that always use full-context RoPE."""
    if config.num_kv_heads is None:
        config.num_kv_heads = config.num_heads
    require_finite_fields(config, (
        "vocab_size", "dim", "num_layers", "num_heads", "num_kv_heads", "max_seq_len",
        "dropout", "ffn_mult", "num_experts", "top_k_experts",
    ))
    require_integer_fields(config, (
        "vocab_size", "dim", "num_layers", "num_heads", "num_kv_heads",
        "max_seq_len", "num_experts", "top_k_experts",
    ))
    require(config.vocab_size > 0, "vocab_size must be > 0")
    require(config.dim > 0, "dim must be > 0")
    require(config.num_layers > 0, "num_layers must be > 0")
    require(config.num_heads > 0, "num_heads must be > 0")
    require(config.num_kv_heads > 0, "num_kv_heads must be > 0")
    require(config.dim % config.num_heads == 0, "dim must be divisible by num_heads")
    require((config.dim // config.num_heads) % 2 == 0, "RoPE requires even head dimension")
    require(config.max_seq_len > 0, "max_seq_len must be > 0")
    require(0.0 <= config.dropout < 1.0, "dropout must be in [0, 1)")
    require(config.ffn_mult > 0, "ffn_mult must be > 0")
    require(config.attention != "lighthouse_mha", (
        f"Lighthouse attention is wired through GPT, not {owner}"
    ))
    require(config.attention not in {
        "cosformer",
        "lightning",
        "gated_deltanet",
        "gated_deltanet2",
        "gemma3",
        "gemma4",
        "qwen3_next",
    }, (
        f"{owner} uses a fixed RoPE transformer; choose a RoPE-compatible attention variant"
    ))
    if attention_uses_gqa(config.attention):
        require(config.num_heads % config.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
    else:
        require(config.num_kv_heads == config.num_heads, "num_kv_heads only applies to GQA attention variants")
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








def apply_logit_softcap(logits, softcap):
    if softcap <= 0:
        return logits
    return torch.tanh(logits / softcap) * softcap
