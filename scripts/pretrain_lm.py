"""Pretrain an autoregressive LM on TinyStories.

    python scripts/pretrain_lm.py --tokenizer tokenizer.json
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention iha --connection mhc
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention gqa --num-kv-heads 4
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --model mamba
"""

import argparse
import torch
from torch.utils.data import DataLoader
from common import MODEL_CHOICES, load_model_checkpoint
from minilab.checks import require
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT, GPTConfig
from minilab.models.mamba import MambaConfig, MambaLM
from minilab.data import load_tinystories
from minilab.trainer import LMTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.generation import generate
from minilab.evaluation import perplexity
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    QK_CLIP_ATTENTIONS,
    resolve_deepseek_v4_attention,
)


_MODEL_BUILD_FLAGS = (
    "dim", "num_layers", "num_heads", "num_kv_heads", "attention", "position",
    "rope_base", "rope_local_base", "rope_global_base", "rope_scaling_factor",
    "rope_original_max_seq_len", "rope_partial_rotary_factor", "yarn_beta_fast",
    "yarn_beta_slow", "local_attention_window", "qwen3_next_full_attention_interval",
    "attention_k_eq_v", "per_layer_embedding_dim", "final_logit_softcap",
    "connection", "ffn", "num_experts", "top_k_experts", "post_norm",
    "mtp_depth", "mtp_loss_weight",
)
_GPT_ONLY_BUILD_FLAGS = tuple(name for name in _MODEL_BUILD_FLAGS if name not in {"dim", "num_layers"})
_QK_CLIP_FLAGS = ("qk_clip_threshold", "qk_clip_balance")


def _flag(name):
    return "--" + name.replace("_", "-")


def _reject_supplied(args, names, reason):
    for name in names:
        require(getattr(args, name) is None, f"{_flag(name)} {reason}")


def _resolve(value, default):
    return default if value is None else value


def _uses_qk_clip(model):
    return isinstance(model, GPT) and any(
        getattr(block, "attention_name", None) in QK_CLIP_ATTENTIONS
        for block in model.blocks
    )


def _attention_uses_gqa(attention):
    return attention in {"gemma3", "gemma4", "qwen3_next"} or (
        resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS
    )


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/lm")
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="model family for new runs; inferred from checkpoints")
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--attention", default=None)
p.add_argument("--position", default=None)
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
p.add_argument("--max-examples", type=int, default=50000)
p.add_argument("--grad-checkpoint", action="store_true")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
model_name = args.model or "gpt"

if args.resume_from:
    _reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
elif model_name != "gpt":
    _reject_supplied(args, _GPT_ONLY_BUILD_FLAGS, "only applies to --model gpt")
if not args.resume_from and model_name != "gpt":
    _reject_supplied(args, _QK_CLIP_FLAGS, "only applies to --model gpt")
if args.qk_clip_threshold is not None:
    require(args.qk_clip_threshold > 0, "--qk-clip-threshold must be > 0 when supplied")
if args.qk_clip_balance is not None:
    require(
        args.qk_clip_threshold is not None,
        "--qk-clip-balance only applies when --qk-clip-threshold is supplied",
    )

set_seed(args.seed)

dim = _resolve(args.dim, 256)
num_layers = _resolve(args.num_layers, 6)
num_heads = _resolve(args.num_heads, 8)
attention = _resolve(args.attention, "mha")
position = _resolve(args.position, "rope")
rope_base = _resolve(args.rope_base, 10000.0)
rope_local_base = _resolve(args.rope_local_base, 10000.0)
rope_global_base = _resolve(args.rope_global_base, 1000000.0)
rope_scaling_factor = _resolve(args.rope_scaling_factor, 1.0)
rope_original_max_seq_len = _resolve(args.rope_original_max_seq_len, 4096)
rope_partial_rotary_factor = _resolve(args.rope_partial_rotary_factor, 0.25)
yarn_beta_fast = _resolve(args.yarn_beta_fast, 32.0)
yarn_beta_slow = _resolve(args.yarn_beta_slow, 1.0)
local_attention_window = _resolve(args.local_attention_window, 1024)
qwen3_next_full_attention_interval = _resolve(args.qwen3_next_full_attention_interval, 4)
attention_k_eq_v = _resolve(args.attention_k_eq_v, False)
per_layer_embedding_dim = _resolve(args.per_layer_embedding_dim, 0)
final_logit_softcap = _resolve(args.final_logit_softcap, 0.0)
connection = _resolve(args.connection, "residual")
ffn = _resolve(args.ffn, "swiglu")
num_experts = _resolve(args.num_experts, 8)
top_k_experts = _resolve(args.top_k_experts, 2)
post_norm = _resolve(args.post_norm, False)
mtp_depth = _resolve(args.mtp_depth, 0)
mtp_loss_weight = _resolve(args.mtp_loss_weight, 0.0)
qk_clip_threshold = _resolve(args.qk_clip_threshold, 0.0)
qk_clip_balance = _resolve(args.qk_clip_balance, 0.5)

if args.num_kv_heads is not None:
    require(_attention_uses_gqa(attention), "--num-kv-heads only applies to GQA attention variants")
if args.num_experts is not None or args.top_k_experts is not None:
    require(ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")
if args.mtp_depth is not None:
    require(mtp_depth > 0, "--mtp-depth must be > 0 when supplied")
    require(mtp_loss_weight > 0, "--mtp-depth requires --mtp-loss-weight > 0")
if args.mtp_loss_weight is not None:
    require(mtp_loss_weight > 0, "--mtp-loss-weight must be > 0 when supplied")
    require(mtp_depth > 0, "--mtp-loss-weight only applies when --mtp-depth > 0")

tok = load_tokenizer(args.tokenizer)
train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples)
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=2000)
print(f"Data: train={len(train_ds)} eval={len(eval_ds)}")

if args.resume_from:
    validate_checkpoint_tokenizer(args.resume_from, tok)
    model_name, model = load_model_checkpoint(args.resume_from, args.model)
    print(f"Resuming from {args.resume_from} ({model_name})")
else:
    if model_name == "gpt":
        config = GPTConfig(
            vocab_size=tok.vocab_size, dim=dim, num_layers=num_layers,
            num_heads=num_heads, num_kv_heads=args.num_kv_heads, max_seq_len=args.seq_len,
            attention=attention, position=position, connection=connection,
            ffn=ffn, num_experts=num_experts, top_k_experts=top_k_experts,
            post_norm=post_norm, rope_base=rope_base, rope_local_base=rope_local_base,
            rope_global_base=rope_global_base, rope_scaling_factor=rope_scaling_factor,
            rope_original_max_seq_len=rope_original_max_seq_len,
            rope_partial_rotary_factor=rope_partial_rotary_factor,
            yarn_beta_fast=yarn_beta_fast, yarn_beta_slow=yarn_beta_slow,
            local_attention_window=local_attention_window,
            qwen3_next_full_attention_interval=qwen3_next_full_attention_interval,
            attention_k_eq_v=attention_k_eq_v,
            per_layer_embedding_dim=per_layer_embedding_dim,
            final_logit_softcap=final_logit_softcap,
            mtp_depth=mtp_depth, mtp_loss_weight=mtp_loss_weight,
        )
        model = GPT(config)
    else:
        config = MambaConfig(
            vocab_size=tok.vocab_size, dim=dim, num_layers=num_layers, max_seq_len=args.seq_len,
        )
        model = MambaLM(config)
if args.qk_clip_threshold is not None:
    require(_uses_qk_clip(model), "--qk-clip-threshold requires a GPT attention with QK-Clip support")
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
sig = run_signature(tok, {"name": "tinystories", "split": "train", "max_examples": args.max_examples}, args.seq_len)
trainer = LMTrainer(model, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok), eval_dataset=eval_ds)
trainer.train()
model = trainer.model

model.eval()
ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
print(f"\nEval perplexity: {ppl:.1f}")

for text in ["Once upon a time", "The little dog", "She was very happy"]:
    out = generate(model, torch.tensor([tok.encode(text)]), max_new_tokens=80, temperature=0.8, top_k=40)
    print(f"  {tok.decode(out[0].tolist())[:120]}")
