MOE_FFNS = {
    "moe", "switch_moe", "mixtral_moe", "expert_choice_moe",
    "deepseek_moe", "aux_free_moe", "base_moe", "gemma4_moe",
}

TOP_ONE_MOE_FFNS = {"switch_moe", "base_moe"}

GQA_ATTENTIONS = {
    "gqa",
    "gqa_qknorm",
    "gated_gqa_qknorm",
    "gated_gqa_qknorm_partial_rope",
    "gqa_qknorm_kv_tied",
    "gqa_qknorm_partial_rope",
    "sliding_window_gqa_qknorm",
}

QK_CLIP_ATTENTIONS = {
    "mha",
    "mha_qknorm",
    "gqa",
    "gqa_qknorm",
    "gated_gqa_qknorm",
    "gated_gqa_qknorm_partial_rope",
    "gqa_qknorm_kv_tied",
    "gqa_qknorm_partial_rope",
    "sliding_window_gqa_qknorm",
    "mqa",
    "mla",
}


def resolve_deepseek_v4_attention(attention, block_id):
    if attention == "deepseek_v4_flash":
        return "sliding_window" if block_id < 2 else interleaved_csa_hca(block_id)
    if attention in {"deepseek_v4", "deepseek_v4_pro"}:
        return "hca" if block_id < 2 else interleaved_csa_hca(block_id)
    return attention


def interleaved_csa_hca(block_id):
    return "csa" if (block_id - 2) % 2 == 0 else "hca"
