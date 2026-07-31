"""Pretrain an autoregressive LM.

    python scripts/pretrain_lm.py --tokenizer tokenizer.json
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention iha --connection mhc
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention gqa --num-kv-heads 4
"""

import argparse
import torch
from torch.utils.data import DataLoader
from common import (
    MODEL_CHOICES,
    PRETRAIN_DATASET_CHOICES,
    build_lm_model,
    lm_model_kwargs,
    load_pretrain_dataset,
    load_pretrain_eval_dataset,
    load_model_checkpoint,
    reject_supplied,
    resolve_pretrain_max_examples,
    resolve_default,
    resolve_save_every,
)
from minilab.checks import require
from minilab.nn.attention_common import DEFAULT_ATTENTION_BACKEND, attention_backend_choices
from minilab.nn.optimizers import DEFAULT_KL_SHAMPOO_BETAS, DEFAULT_KL_SHAMPOO_LR, DEFAULT_SOFT_MUON_POWER
from minilab.presets import get_lm_model_preset, lm_model_preset_choices
from minilab.tokenizers import load_tokenizer
from minilab.trainer import (
    LMTrainer,
    TrainConfig,
    autocast_context,
    run_signature,
    set_seed,
    tokenizer_signature,
    validate_checkpoint_tokenizer,
)
from minilab.generation import generate
from minilab.evaluation import perplexity
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
    DEFAULT_SITU_GATE_CAP,
    DEFAULT_SITU_UP_CAP,
    DEFAULT_SPARSE_BLOCK_SIZE,
    DEFAULT_SPARSE_INDEX_DIM,
    DEFAULT_SPARSE_INDEX_SHARE_INTERVAL,
    DEFAULT_SPARSE_INDEX_WARMUP_STEPS,
    DEFAULT_SPARSE_KL_LOSS_WEIGHT,
    DEFAULT_SPARSE_LOCAL_BLOCKS,
    DEFAULT_SPARSE_TOP_K_BLOCKS,
    DEFAULT_LIGHTHOUSE_NUM_LEVELS,
    DEFAULT_LIGHTHOUSE_POOLING_FACTOR,
    DEFAULT_LIGHTHOUSE_TOP_K,
    DEFAULT_TOP_K_EXPERTS,
    DEFAULT_YARN_BETA_FAST,
    DEFAULT_YARN_BETA_SLOW,
    attention_uses_gqa,
)
from minilab.nn.architecture import (
    MOE_FFNS,
    resolve_deepseek_v4_attention,
)


_MODEL_SELECTOR_FLAGS = ("preset",)
_MODEL_BUILD_FLAGS = (
    "dim", "num_layers", "num_heads", "num_kv_heads", "attention",
    "attention_backend", "position", "norm",
    "rope_base", "rope_local_base", "rope_global_base", "rope_scaling_factor",
    "rope_original_max_seq_len", "rope_partial_rotary_factor", "yarn_beta_fast",
    "yarn_beta_slow", "local_attention_window", "qwen3_next_full_attention_interval",
    "sparse_block_size", "sparse_top_k_blocks", "sparse_local_blocks", "sparse_index_dim",
    "sparse_kl_loss_weight", "sparse_index_warmup_steps", "sparse_index_share_interval",
    "situ_gate_cap", "situ_up_cap",
    "lighthouse_num_levels", "lighthouse_pooling_factor", "lighthouse_top_k",
    "attention_k_eq_v", "per_layer_embedding_dim", "final_logit_softcap",
    "connection", "ffn", "num_experts", "top_k_experts", "post_norm",
    "mtp_depth", "mtp_loss_weight", "mtp_mode",
    "layerskip_loss_weight", "layerskip_dropout", "layerskip_min_layer",
    "future_summary_window", "future_summary_loss_weight",
    "jacobi_loss_weight", "jacobi_iterations",
)
_LOCAL_WINDOW_ATTENTIONS = {"gemma3", "gemma4", "sliding_window", "sliding_window_gqa_qknorm"}
_PARTIAL_ROPE_ATTENTIONS = {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope", "qwen3_next"}


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/lm")
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="model family for new runs; inferred from checkpoints")
p.add_argument("--preset", choices=lm_model_preset_choices(), default=None, help="tiny model preset for new runs")
p.add_argument("--dataset", choices=PRETRAIN_DATASET_CHOICES, default="tinystories")
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=None)
p.add_argument("--attention", default=None)
p.add_argument("--attention-backend", choices=attention_backend_choices(), default=None)
p.add_argument("--position", default=None)
p.add_argument("--norm", default=None)
p.add_argument("--rope-base", type=float, default=None)
p.add_argument("--rope-local-base", type=float, default=None)
p.add_argument("--rope-global-base", type=float, default=None)
p.add_argument("--rope-scaling-factor", type=float, default=None)
p.add_argument("--rope-original-max-seq-len", type=int, default=None)
p.add_argument("--rope-partial-rotary-factor", type=float, default=None)
p.add_argument("--yarn-beta-fast", type=float, default=None)
p.add_argument("--yarn-beta-slow", type=float, default=None)
p.add_argument("--local-attention-window", type=int, default=None)
p.add_argument("--qwen3-next-full-attention-interval", type=int, default=None)
p.add_argument("--sparse-block-size", type=int, default=None)
p.add_argument("--sparse-top-k-blocks", type=int, default=None)
p.add_argument("--sparse-local-blocks", type=int, default=None)
p.add_argument("--sparse-index-dim", type=int, default=None, help="0 uses the attention head dimension")
p.add_argument("--sparse-kl-loss-weight", type=float, default=None)
p.add_argument("--sparse-index-warmup-steps", type=int, default=None)
p.add_argument("--sparse-index-share-interval", type=int, default=None, help="reuse each indexer across this many sparse layers")
p.add_argument("--situ-gate-cap", type=float, default=None)
p.add_argument("--situ-up-cap", type=float, default=None)
p.add_argument("--lighthouse-num-levels", type=int, default=None)
p.add_argument("--lighthouse-pooling-factor", type=int, default=None)
p.add_argument("--lighthouse-top-k", type=int, default=None)
p.add_argument("--attention-k-eq-v", action="store_true", default=None)
p.add_argument("--per-layer-embedding-dim", type=int, default=None)
p.add_argument("--final-logit-softcap", type=float, default=None)
p.add_argument("--connection", default=None)
p.add_argument("--ffn", default=None)
p.add_argument("--num-experts", type=int, default=None)
p.add_argument("--top-k-experts", type=int, default=None)
p.add_argument("--post-norm", action="store_true", default=None)
p.add_argument("--mtp-depth", type=int, default=None)
p.add_argument("--mtp-loss-weight", type=float, default=None)
p.add_argument("--mtp-mode", choices=["sequential", "parallel"], default=None)
p.add_argument("--layerskip-loss-weight", type=float, default=None)
p.add_argument("--layerskip-dropout", type=float, default=None)
p.add_argument("--layerskip-min-layer", type=int, default=None)
p.add_argument("--future-summary-window", type=int, default=None)
p.add_argument("--future-summary-loss-weight", type=float, default=None)
p.add_argument("--jacobi-loss-weight", type=float, default=None)
p.add_argument("--jacobi-iterations", type=int, default=None)
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--muon-lr", type=float, default=None, help="defaults to 0.02 for Muon-family optimizers")
p.add_argument("--muon-per-head", action="store_true", default=None, help="per-head Newton-Schulz for attention projections")
p.add_argument("--soft-muon-power", type=float, default=None, help="fixed p=0.4 profile for --optimizer soft_muon")
p.add_argument("--kl-shampoo-lr", type=float, default=None, help="defaults to 0.02 for --optimizer kl_shampoo")
p.add_argument("--kl-shampoo-beta1", type=float, default=None, help="momentum beta, defaults to 0.95")
p.add_argument("--kl-shampoo-beta2", type=float, default=None, help="preconditioner EMA beta, defaults to beta1**2")
p.add_argument("--optimizer", choices=["adamw", "lion", "muon", "soft_muon", "kl_shampoo"], default="adamw")
p.add_argument("--lr-schedule", choices=["cosine", "linear", "constant", "wsd"], default="cosine")
p.add_argument("--qk-clip-threshold", type=float, default=None)
p.add_argument("--qk-clip-balance", type=float, default=None)
p.add_argument("--token-superposition-size", type=int, default=None)
p.add_argument("--token-superposition-steps", type=int, default=None)
p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--grad-checkpoint", action="store_true")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
preset = get_lm_model_preset(args.preset) if args.preset else {}
if args.preset and args.model is not None:
    require(
        args.model == preset["model"],
        f"--model {args.model} conflicts with --preset {args.preset} ({preset['model']})",
    )
model_name = preset.get("model") or args.model or "gpt"

if args.resume_from:
    reject_supplied(args, _MODEL_SELECTOR_FLAGS + _MODEL_BUILD_FLAGS, "only applies when starting a new model")
if args.qk_clip_threshold is not None:
    require(args.qk_clip_threshold > 0, "--qk-clip-threshold must be > 0 when supplied")
if args.qk_clip_balance is not None:
    require(
        args.qk_clip_threshold is not None,
        "--qk-clip-balance only applies when --qk-clip-threshold is supplied",
    )
if args.optimizer not in {"muon", "soft_muon"}:
    require(args.muon_lr is None, "--muon-lr only applies to --optimizer muon or soft_muon")
if args.optimizer != "muon":
    require(args.muon_per_head is None, "--muon-per-head only applies to --optimizer muon")
if args.optimizer != "soft_muon":
    require(args.soft_muon_power is None, "--soft-muon-power only applies to --optimizer soft_muon")
if args.optimizer != "kl_shampoo":
    require(
        args.kl_shampoo_lr is None and args.kl_shampoo_beta1 is None and args.kl_shampoo_beta2 is None,
        "--kl-shampoo-lr and --kl-shampoo-beta* only apply to --optimizer kl_shampoo",
    )
if args.soft_muon_power is not None:
    require(
        args.soft_muon_power == DEFAULT_SOFT_MUON_POWER,
        "--soft-muon-power currently supports the fixed p=0.4 coefficient profile",
    )

set_seed(args.seed)

dim = resolve_default(args.dim, preset.get("dim", 256))
num_layers = resolve_default(args.num_layers, preset.get("num_layers", 6))
num_heads = resolve_default(args.num_heads, preset.get("num_heads", 8))
seq_len = resolve_default(args.seq_len, preset.get("seq_len", 256))
attention = resolve_default(args.attention, "mha")
attention_backend = resolve_default(args.attention_backend, DEFAULT_ATTENTION_BACKEND)
position = resolve_default(args.position, "rope")
norm = resolve_default(args.norm, "rmsnorm")
rope_base = resolve_default(args.rope_base, DEFAULT_ROPE_BASE)
rope_local_base = resolve_default(args.rope_local_base, DEFAULT_ROPE_LOCAL_BASE)
rope_global_base = resolve_default(args.rope_global_base, DEFAULT_ROPE_GLOBAL_BASE)
rope_scaling_factor = resolve_default(args.rope_scaling_factor, DEFAULT_ROPE_SCALING_FACTOR)
rope_original_max_seq_len = resolve_default(args.rope_original_max_seq_len, DEFAULT_ROPE_ORIGINAL_MAX_SEQ_LEN)
rope_partial_rotary_factor = resolve_default(args.rope_partial_rotary_factor, DEFAULT_ROPE_PARTIAL_ROTARY_FACTOR)
yarn_beta_fast = resolve_default(args.yarn_beta_fast, DEFAULT_YARN_BETA_FAST)
yarn_beta_slow = resolve_default(args.yarn_beta_slow, DEFAULT_YARN_BETA_SLOW)
local_attention_window = resolve_default(args.local_attention_window, DEFAULT_LOCAL_ATTENTION_WINDOW)
qwen3_next_full_attention_interval = resolve_default(
    args.qwen3_next_full_attention_interval,
    DEFAULT_QWEN3_NEXT_FULL_ATTENTION_INTERVAL,
)
sparse_block_size = resolve_default(args.sparse_block_size, DEFAULT_SPARSE_BLOCK_SIZE)
sparse_top_k_blocks = resolve_default(args.sparse_top_k_blocks, DEFAULT_SPARSE_TOP_K_BLOCKS)
sparse_local_blocks = resolve_default(args.sparse_local_blocks, DEFAULT_SPARSE_LOCAL_BLOCKS)
sparse_index_dim = resolve_default(args.sparse_index_dim, DEFAULT_SPARSE_INDEX_DIM)
sparse_kl_loss_weight = resolve_default(args.sparse_kl_loss_weight, DEFAULT_SPARSE_KL_LOSS_WEIGHT)
sparse_index_warmup_steps = resolve_default(args.sparse_index_warmup_steps, DEFAULT_SPARSE_INDEX_WARMUP_STEPS)
sparse_index_share_interval = resolve_default(args.sparse_index_share_interval, DEFAULT_SPARSE_INDEX_SHARE_INTERVAL)
situ_gate_cap = resolve_default(args.situ_gate_cap, DEFAULT_SITU_GATE_CAP)
situ_up_cap = resolve_default(args.situ_up_cap, DEFAULT_SITU_UP_CAP)
lighthouse_num_levels = resolve_default(args.lighthouse_num_levels, DEFAULT_LIGHTHOUSE_NUM_LEVELS)
lighthouse_pooling_factor = resolve_default(args.lighthouse_pooling_factor, DEFAULT_LIGHTHOUSE_POOLING_FACTOR)
lighthouse_top_k = resolve_default(args.lighthouse_top_k, DEFAULT_LIGHTHOUSE_TOP_K)
attention_k_eq_v = resolve_default(args.attention_k_eq_v, False)
per_layer_embedding_dim = resolve_default(args.per_layer_embedding_dim, 0)
final_logit_softcap = resolve_default(args.final_logit_softcap, 0.0)
connection = resolve_default(args.connection, "residual")
ffn = resolve_default(args.ffn, "swiglu")
num_experts = resolve_default(args.num_experts, DEFAULT_NUM_EXPERTS)
top_k_experts = resolve_default(args.top_k_experts, DEFAULT_TOP_K_EXPERTS)
post_norm = resolve_default(args.post_norm, False)
mtp_depth = resolve_default(args.mtp_depth, 0)
mtp_loss_weight = resolve_default(args.mtp_loss_weight, 0.0)
mtp_mode = resolve_default(args.mtp_mode, "sequential")
layerskip_loss_weight = resolve_default(args.layerskip_loss_weight, 0.0)
layerskip_dropout = resolve_default(args.layerskip_dropout, 0.0)
layerskip_min_layer = resolve_default(args.layerskip_min_layer, 1)
future_summary_window = resolve_default(args.future_summary_window, 0)
future_summary_loss_weight = resolve_default(args.future_summary_loss_weight, 0.0)
jacobi_loss_weight = resolve_default(args.jacobi_loss_weight, 0.0)
jacobi_iterations = resolve_default(args.jacobi_iterations, 0)
qk_clip_threshold = resolve_default(args.qk_clip_threshold, 0.0)
qk_clip_balance = resolve_default(args.qk_clip_balance, 0.5)
token_superposition_size = resolve_default(args.token_superposition_size, 1)
token_superposition_steps = resolve_default(args.token_superposition_steps, 0)
muon_lr = resolve_default(args.muon_lr, 0.02)
muon_per_head = resolve_default(args.muon_per_head, False)
soft_muon_power = resolve_default(args.soft_muon_power, DEFAULT_SOFT_MUON_POWER)
kl_shampoo_lr = resolve_default(args.kl_shampoo_lr, DEFAULT_KL_SHAMPOO_LR)
kl_shampoo_beta1 = resolve_default(args.kl_shampoo_beta1, DEFAULT_KL_SHAMPOO_BETAS[0])
kl_shampoo_beta2 = resolve_default(args.kl_shampoo_beta2, DEFAULT_KL_SHAMPOO_BETAS[1])

if args.num_kv_heads is not None:
    require(attention_uses_gqa(attention), "--num-kv-heads only applies to GQA attention variants")
if args.num_experts is not None or args.top_k_experts is not None:
    require(ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")
if args.attention_k_eq_v is not None:
    require(
        model_name == "gpt" and attention == "gemma4",
        "--attention-k-eq-v only applies to --model gpt --attention gemma4",
    )
if args.qwen3_next_full_attention_interval is not None:
    require(
        attention == "qwen3_next",
        "--qwen3-next-full-attention-interval only applies to --attention qwen3_next",
    )
if args.local_attention_window is not None:
    resolved_attention = resolve_deepseek_v4_attention(attention, 0)
    require(
        attention in _LOCAL_WINDOW_ATTENTIONS
        or resolved_attention in {"sliding_window", "sliding_window_gqa_qknorm"},
        "--local-attention-window only applies to local/sliding-window attention",
    )
if (
    args.sparse_block_size is not None
    or args.sparse_top_k_blocks is not None
    or args.sparse_local_blocks is not None
    or args.sparse_index_dim is not None
    or args.sparse_kl_loss_weight is not None
    or args.sparse_index_warmup_steps is not None
    or args.sparse_index_share_interval is not None
):
    require(
        attention == "learned_block_gqa",
        "--sparse-* flags only apply to --attention learned_block_gqa",
    )
if args.situ_gate_cap is not None or args.situ_up_cap is not None:
    require(
        ffn == "situ_glu",
        "--situ-gate-cap and --situ-up-cap only apply to --ffn situ_glu",
    )
if (
    args.lighthouse_num_levels is not None
    or args.lighthouse_pooling_factor is not None
    or args.lighthouse_top_k is not None
):
    require(
        model_name == "gpt" and attention == "lighthouse_mha",
        "--lighthouse-* flags only apply to --model gpt --attention lighthouse_mha",
    )
if args.rope_base is not None:
    require(position in {"rope", "yarn_rope"}, "--rope-base only applies to --position rope or yarn_rope")
if args.rope_local_base is not None or args.rope_global_base is not None:
    require(
        position in {"gemma3_rope", "gemma4_rope", "qwen3_next_rope"},
        "--rope-local-base and --rope-global-base only apply to Gemma/Qwen local-global RoPE positions",
    )
if args.rope_scaling_factor is not None:
    require(
        position in {"yarn_rope", "gemma4_rope"},
        "--rope-scaling-factor only applies to YaRN RoPE or Gemma 4 proportional RoPE",
    )
if args.rope_original_max_seq_len is not None or args.yarn_beta_fast is not None or args.yarn_beta_slow is not None:
    require(position == "yarn_rope", "--rope-original-max-seq-len and YaRN beta flags only apply to --position yarn_rope")
if args.rope_partial_rotary_factor is not None:
    require(
        attention in _PARTIAL_ROPE_ATTENTIONS
        or resolve_deepseek_v4_attention(attention, 0) in _PARTIAL_ROPE_ATTENTIONS
        or position in {"gemma4_rope", "qwen3_next_rope"},
        "--rope-partial-rotary-factor only applies to partial-RoPE attention or Gemma/Qwen proportional RoPE",
    )
if args.mtp_depth is not None:
    require(mtp_depth > 0, "--mtp-depth must be > 0 when supplied")
    require(mtp_loss_weight > 0, "--mtp-depth requires --mtp-loss-weight > 0")
if args.mtp_loss_weight is not None:
    require(mtp_loss_weight > 0, "--mtp-loss-weight must be > 0 when supplied")
    require(mtp_depth > 0, "--mtp-loss-weight only applies when --mtp-depth > 0")
if args.mtp_mode is not None:
    require(mtp_depth > 0, "--mtp-mode only applies when --mtp-depth > 0")
if args.layerskip_loss_weight is not None:
    require(layerskip_loss_weight > 0, "--layerskip-loss-weight must be > 0 when supplied")
if args.layerskip_dropout is not None:
    require(layerskip_dropout > 0, "--layerskip-dropout must be > 0 when supplied")
    require(layerskip_loss_weight > 0, "--layerskip-dropout requires --layerskip-loss-weight > 0")
if args.layerskip_min_layer is not None:
    require(layerskip_min_layer > 0, "--layerskip-min-layer must be > 0")
    require(layerskip_loss_weight > 0, "--layerskip-min-layer requires --layerskip-loss-weight > 0")
if args.future_summary_window is not None:
    require(future_summary_window > 0, "--future-summary-window must be > 0 when supplied")
    require(future_summary_loss_weight > 0, "--future-summary-window requires --future-summary-loss-weight > 0")
if args.future_summary_loss_weight is not None:
    require(future_summary_loss_weight > 0, "--future-summary-loss-weight must be > 0 when supplied")
    require(future_summary_window > 0, "--future-summary-loss-weight requires --future-summary-window > 0")
if args.jacobi_loss_weight is not None:
    require(jacobi_loss_weight > 0, "--jacobi-loss-weight must be > 0 when supplied")
    require(jacobi_iterations > 0, "--jacobi-loss-weight requires --jacobi-iterations > 0")
if args.jacobi_iterations is not None:
    require(jacobi_iterations > 0, "--jacobi-iterations must be > 0 when supplied")
    require(jacobi_loss_weight > 0, "--jacobi-iterations requires --jacobi-loss-weight > 0")
if args.token_superposition_size is not None:
    require(token_superposition_size > 1, "--token-superposition-size must be > 1 when supplied")
    require(token_superposition_steps > 0, "--token-superposition-size requires --token-superposition-steps > 0")
if args.token_superposition_steps is not None:
    require(token_superposition_steps > 0, "--token-superposition-steps must be > 0 when supplied")
    require(token_superposition_size > 1, "--token-superposition-steps requires --token-superposition-size > 1")

tok = load_tokenizer(args.tokenizer)
max_examples = resolve_pretrain_max_examples(args.dataset, args.max_examples, 50000)
train_ds = load_pretrain_dataset(args.dataset, tok, seq_len, "train", max_examples, "lm")
eval_ds = (
    None
    if args.dataset == "openwebtext"
    else load_pretrain_eval_dataset(args.dataset, tok, seq_len, 2000, "lm")
)
eval_count = "none" if eval_ds is None else len(eval_ds)
print(f"Data: {args.dataset} train={len(train_ds)} eval={eval_count}")

if args.resume_from:
    validate_checkpoint_tokenizer(args.resume_from, tok)
    model_name, model = load_model_checkpoint(args.resume_from, args.model)
    print(f"Resuming from {args.resume_from} ({model_name})")
else:
    config_kwargs = lm_model_kwargs(
        model_name,
        vocab_size=tok.vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=args.num_kv_heads,
        max_seq_len=seq_len,
        attention=attention,
        attention_backend=attention_backend,
        position=position,
        norm=norm,
        connection=connection,
        ffn=ffn,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        post_norm=post_norm,
        rope_base=rope_base,
        rope_local_base=rope_local_base,
        rope_global_base=rope_global_base,
        rope_scaling_factor=rope_scaling_factor,
        rope_original_max_seq_len=rope_original_max_seq_len,
        rope_partial_rotary_factor=rope_partial_rotary_factor,
        yarn_beta_fast=yarn_beta_fast,
        yarn_beta_slow=yarn_beta_slow,
        local_attention_window=local_attention_window,
        qwen3_next_full_attention_interval=qwen3_next_full_attention_interval,
        sparse_block_size=sparse_block_size,
        sparse_top_k_blocks=sparse_top_k_blocks,
        sparse_local_blocks=sparse_local_blocks,
        sparse_index_dim=sparse_index_dim,
        sparse_kl_loss_weight=sparse_kl_loss_weight,
        sparse_index_warmup_steps=sparse_index_warmup_steps,
        sparse_index_share_interval=sparse_index_share_interval,
        situ_gate_cap=situ_gate_cap,
        situ_up_cap=situ_up_cap,
        lighthouse_num_levels=lighthouse_num_levels,
        lighthouse_pooling_factor=lighthouse_pooling_factor,
        lighthouse_top_k=lighthouse_top_k,
        attention_k_eq_v=attention_k_eq_v,
        per_layer_embedding_dim=per_layer_embedding_dim,
        final_logit_softcap=final_logit_softcap,
        mtp_depth=mtp_depth,
        mtp_loss_weight=mtp_loss_weight,
        mtp_mode=mtp_mode,
        layerskip_loss_weight=layerskip_loss_weight,
        layerskip_dropout=layerskip_dropout,
        layerskip_min_layer=layerskip_min_layer,
        future_summary_window=future_summary_window,
        future_summary_loss_weight=future_summary_loss_weight,
        jacobi_loss_weight=jacobi_loss_weight,
        jacobi_iterations=jacobi_iterations,
    )
    model = build_lm_model(model_name, **config_kwargs)
if args.qk_clip_threshold is not None:
    require(model.supports_qk_clip(), "--qk-clip-threshold requires QK-Clip-capable attention")
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
print(f"{type(model).__name__}: {model.num_parameters():,} params")

tc = TrainConfig(
    max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
    muon_lr=muon_lr, muon_per_head=muon_per_head, soft_muon_power=soft_muon_power,
    kl_shampoo_lr=kl_shampoo_lr, kl_shampoo_beta1=kl_shampoo_beta1, kl_shampoo_beta2=kl_shampoo_beta2,
    optimizer=args.optimizer, lr_schedule=args.lr_schedule,
    qk_clip_threshold=qk_clip_threshold, qk_clip_balance=qk_clip_balance,
    log_every=100, eval_every=500, save_every=resolve_save_every(args.save_every, args.max_steps),
    save_dir=args.save_dir,
    resume_from=args.resume_from, seed=args.seed,
    token_superposition_size=token_superposition_size,
    token_superposition_steps=token_superposition_steps,
)
sig = run_signature(tok, {"name": args.dataset, "split": "train", "max_examples": max_examples}, seq_len)
trainer = LMTrainer(model, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok), eval_dataset=eval_ds)
trainer.train()
model = trainer.model

model.eval()
with autocast_context(next(model.parameters()).device, tc.dtype):
    if eval_ds is not None:
        ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
        print(f"\nEval perplexity: {ppl:.1f}")

    for text in ["once upon a time", "the little dog", "she was very happy"]:
        out = generate(model, torch.tensor([tok.encode(text)]), max_new_tokens=80, temperature=0.8, top_k=40)
        print(f"  {tok.decode(out[0].tolist())[:120]}")
