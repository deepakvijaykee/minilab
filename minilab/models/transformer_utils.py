import torch

from minilab.checks import require
from minilab.nn.architecture import MOE_FFNS, QK_CLIP_ATTENTIONS
from minilab.registry import get_position


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


def apply_logit_softcap(logits, softcap):
    if softcap <= 0:
        return logits
    return torch.tanh(logits / softcap) * softcap
