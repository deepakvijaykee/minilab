"""Pretrain an autoregressive LM.

    python scripts/pretrain_lm.py --tokenizer tokenizer.json
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention iha --connection mhc
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention gqa --num-kv-heads 4
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --model mamba
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --model mamba2
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --model hymba
"""

import argparse
import torch
from torch.utils.data import DataLoader
from common import (
    MODEL_CHOICES,
    PRETRAIN_DATASET_CHOICES,
    attention_uses_gqa,
    build_lm_model,
    lm_model_kwargs,
    load_pretrain_dataset,
    load_pretrain_eval_dataset,
    load_model_checkpoint,
    reject_supplied,
    resolve_pretrain_max_examples,
    resolve_default,
)
from minilab.checks import require
from minilab.tokenizers import load_tokenizer
from minilab.trainer import LMTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.generation import generate
from minilab.evaluation import perplexity
from minilab.nn.architecture import (
    MOE_FFNS,
    resolve_deepseek_v4_attention,
)


_MODEL_BUILD_FLAGS = (
    "dim", "num_layers", "num_heads", "num_kv_heads", "attention", "position",
    "norm",
    "rope_base", "rope_local_base", "rope_global_base", "rope_scaling_factor",
    "rope_original_max_seq_len", "rope_partial_rotary_factor", "yarn_beta_fast",
    "yarn_beta_slow", "local_attention_window", "qwen3_next_full_attention_interval",
    "attention_k_eq_v", "per_layer_embedding_dim", "final_logit_softcap",
    "connection", "ffn", "num_experts", "top_k_experts", "post_norm",
    "mtp_depth", "mtp_loss_weight",
)
_MAMBA_BUILD_FLAGS = ("dim", "num_layers")
_XLSTM_BUILD_FLAGS = ("dim", "num_layers", "num_heads")
_BYTE_LATENT_BUILD_FLAGS = ("dim", "num_layers", "num_heads", "attention", "norm", "ffn")
_MAMBA_ONLY_REJECTED_FLAGS = tuple(name for name in _MODEL_BUILD_FLAGS if name not in _MAMBA_BUILD_FLAGS)
_XLSTM_ONLY_REJECTED_FLAGS = tuple(name for name in _MODEL_BUILD_FLAGS if name not in _XLSTM_BUILD_FLAGS)
_BYTE_LATENT_REJECTED_FLAGS = tuple(name for name in _MODEL_BUILD_FLAGS if name not in _BYTE_LATENT_BUILD_FLAGS)
_MTP_FLAGS = ("mtp_depth", "mtp_loss_weight")
_QK_CLIP_FLAGS = ("qk_clip_threshold", "qk_clip_balance")
_QK_CLIP_MODEL_CHOICES = {"gpt", "hybrid", "hymba", "byte_latent"}
_LOCAL_WINDOW_ATTENTIONS = {"gemma3", "gemma4", "sliding_window", "sliding_window_gqa_qknorm"}
_PARTIAL_ROPE_ATTENTIONS = {"gqa_qknorm_partial_rope", "gated_gqa_qknorm_partial_rope", "qwen3_next"}


def _uses_qk_clip(model):
    return model.supports_qk_clip()


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/lm")
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="model family for new runs; inferred from checkpoints")
p.add_argument("--dataset", choices=PRETRAIN_DATASET_CHOICES, default="tinystories")
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--attention", default=None)
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
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--muon-lr", type=float, default=0.02)
p.add_argument("--optimizer", choices=["adamw", "lion", "muon"], default="adamw")
p.add_argument("--lr-schedule", choices=["cosine", "linear", "constant", "wsd"], default="cosine")
p.add_argument("--qk-clip-threshold", type=float, default=None)
p.add_argument("--qk-clip-balance", type=float, default=None)
p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--grad-checkpoint", action="store_true")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
model_name = args.model or "gpt"

if args.resume_from:
    reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
elif model_name in {"mamba", "mamba2"}:
    reject_supplied(args, _MAMBA_ONLY_REJECTED_FLAGS, "only applies to --model gpt, hymba, hybrid, or xlstm")
elif model_name in {"hybrid", "hymba"}:
    reject_supplied(args, _MTP_FLAGS, "only applies to --model gpt")
elif model_name == "xlstm":
    reject_supplied(args, _XLSTM_ONLY_REJECTED_FLAGS, "only applies to --model gpt")
elif model_name == "byte_latent":
    reject_supplied(args, _BYTE_LATENT_REJECTED_FLAGS, "only applies to --model gpt, hymba, hybrid, or xlstm")
if not args.resume_from and model_name not in _QK_CLIP_MODEL_CHOICES:
    reject_supplied(args, _QK_CLIP_FLAGS, "only applies to QK-Clip-capable attention models")
if args.qk_clip_threshold is not None:
    require(args.qk_clip_threshold > 0, "--qk-clip-threshold must be > 0 when supplied")
if args.qk_clip_balance is not None:
    require(
        args.qk_clip_threshold is not None,
        "--qk-clip-balance only applies when --qk-clip-threshold is supplied",
    )

set_seed(args.seed)

dim = resolve_default(args.dim, 256)
num_layers = resolve_default(args.num_layers, 6)
num_heads = resolve_default(args.num_heads, 8)
attention = resolve_default(args.attention, "mha")
position = resolve_default(args.position, "rope")
norm = resolve_default(args.norm, "rmsnorm")
rope_base = resolve_default(args.rope_base, 10000.0)
rope_local_base = resolve_default(args.rope_local_base, 10000.0)
rope_global_base = resolve_default(args.rope_global_base, 1000000.0)
rope_scaling_factor = resolve_default(args.rope_scaling_factor, 1.0)
rope_original_max_seq_len = resolve_default(args.rope_original_max_seq_len, 4096)
rope_partial_rotary_factor = resolve_default(args.rope_partial_rotary_factor, 0.25)
yarn_beta_fast = resolve_default(args.yarn_beta_fast, 32.0)
yarn_beta_slow = resolve_default(args.yarn_beta_slow, 1.0)
local_attention_window = resolve_default(args.local_attention_window, 1024)
qwen3_next_full_attention_interval = resolve_default(args.qwen3_next_full_attention_interval, 4)
attention_k_eq_v = resolve_default(args.attention_k_eq_v, False)
per_layer_embedding_dim = resolve_default(args.per_layer_embedding_dim, 0)
final_logit_softcap = resolve_default(args.final_logit_softcap, 0.0)
connection = resolve_default(args.connection, "residual")
ffn = resolve_default(args.ffn, "swiglu")
num_experts = resolve_default(args.num_experts, 8)
top_k_experts = resolve_default(args.top_k_experts, 2)
post_norm = resolve_default(args.post_norm, False)
mtp_depth = resolve_default(args.mtp_depth, 0)
mtp_loss_weight = resolve_default(args.mtp_loss_weight, 0.0)
qk_clip_threshold = resolve_default(args.qk_clip_threshold, 0.0)
qk_clip_balance = resolve_default(args.qk_clip_balance, 0.5)

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

tok = load_tokenizer(args.tokenizer)
max_examples = resolve_pretrain_max_examples(args.dataset, args.max_examples, 50000)
train_ds = load_pretrain_dataset(args.dataset, tok, args.seq_len, "train", max_examples, "lm")
eval_ds = (
    None
    if args.dataset == "openwebtext"
    else load_pretrain_eval_dataset(args.dataset, tok, args.seq_len, 2000, "lm")
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
        max_seq_len=args.seq_len,
        attention=attention,
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
        attention_k_eq_v=attention_k_eq_v,
        per_layer_embedding_dim=per_layer_embedding_dim,
        final_logit_softcap=final_logit_softcap,
        mtp_depth=mtp_depth,
        mtp_loss_weight=mtp_loss_weight,
    )
    model = build_lm_model(model_name, **config_kwargs)
if args.qk_clip_threshold is not None:
    require(_uses_qk_clip(model), "--qk-clip-threshold requires QK-Clip-capable attention")
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
print(f"{type(model).__name__}: {model.num_parameters():,} params")

tc = TrainConfig(
    max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
    muon_lr=args.muon_lr, optimizer=args.optimizer, lr_schedule=args.lr_schedule,
    qk_clip_threshold=qk_clip_threshold, qk_clip_balance=qk_clip_balance,
    log_every=100, eval_every=500, save_every=args.save_every or args.max_steps, save_dir=args.save_dir,
    resume_from=args.resume_from, seed=args.seed,
)
sig = run_signature(tok, {"name": args.dataset, "split": "train", "max_examples": max_examples}, args.seq_len)
trainer = LMTrainer(model, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok), eval_dataset=eval_ds)
trainer.train()
model = trainer.model

model.eval()
if eval_ds is not None:
    ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
    print(f"\nEval perplexity: {ppl:.1f}")

for text in ["once upon a time", "the little dog", "she was very happy"]:
    out = generate(model, torch.tensor([tok.encode(text)]), max_new_tokens=80, temperature=0.8, top_k=40)
    print(f"  {tok.decode(out[0].tolist())[:120]}")
