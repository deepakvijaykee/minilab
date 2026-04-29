"""Pretrain a diffusion LM (MDLM, SEDD, or D3PM) on TinyStories.

    python scripts/pretrain_diffusion.py --tokenizer tokenizer.json
    python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --model sedd --schedule log_linear
"""

import argparse
from minilab.checks import require
from minilab.tokenizers import load_tokenizer
from minilab.data import load_tinystories
from minilab.diffusion import ForwardProcess
from minilab.trainer import DiffusionTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.nn.architecture import MOE_FFNS
from common import (
    DIFFUSION_MODEL_CHOICES,
    attention_uses_gqa,
    diffusion_config_class,
    diffusion_model_class,
    diffusion_sampler,
    load_diffusion_model_checkpoint,
    reject_supplied,
    resolve_default,
)


_MODEL_BUILD_FLAGS = (
    "dim", "num_layers", "num_heads", "num_kv_heads",
    "attention", "ffn", "num_experts", "top_k_experts",
)


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/diffusion")
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="model family for new runs; inferred from checkpoints")
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--attention", default=None)
p.add_argument("--ffn", default=None)
p.add_argument("--num-experts", type=int, default=None)
p.add_argument("--top-k-experts", type=int, default=None)
p.add_argument("--schedule", default="", choices=["", "cosine", "linear", "geometric", "log_linear"])
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max-examples", type=int, default=50000)
p.add_argument("--grad-checkpoint", action="store_true")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
model_name = args.model or "mdlm"

if args.resume_from:
    reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
    require(args.schedule == "", "--schedule only applies when starting a new model")

set_seed(args.seed)

dim = resolve_default(args.dim, 256)
num_layers = resolve_default(args.num_layers, 6)
num_heads = resolve_default(args.num_heads, 8)
attention = resolve_default(args.attention, "mha")
ffn = resolve_default(args.ffn, "swiglu")
num_experts = resolve_default(args.num_experts, 8)
top_k_experts = resolve_default(args.top_k_experts, 2)

if not args.resume_from:
    if args.num_kv_heads is not None:
        require(attention_uses_gqa(attention), "--num-kv-heads only applies to GQA attention variants")
    if args.num_experts is not None or args.top_k_experts is not None:
        require(ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")

tok = load_tokenizer(args.tokenizer)
mask_id = tok.vocab_size

train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples, mode="diffusion")
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=2000, mode="diffusion")
print(f"Data: train={len(train_ds)} eval={len(eval_ds)}")

if args.resume_from:
    validate_checkpoint_tokenizer(args.resume_from, tok)
    model_name, model = load_diffusion_model_checkpoint(args.resume_from, args.model)
    fwd = ForwardProcess.load(f"{args.resume_from}/forward_process.json")
    print(f"Resuming from {args.resume_from} ({model_name}, schedule={fwd.schedule})")
else:
    cfg_cls = diffusion_config_class(model_name)
    model_cls = diffusion_model_class(model_name)
    schedule = args.schedule or ("log_linear" if model_name == "sedd" else "cosine")
    config = cfg_cls(
        vocab_size=tok.vocab_size + 1, dim=dim, num_layers=num_layers,
        num_heads=num_heads, num_kv_heads=args.num_kv_heads, max_seq_len=args.seq_len,
        attention=attention, ffn=ffn, num_experts=num_experts,
        top_k_experts=top_k_experts, mask_token_id=mask_id,
    )
    model = model_cls(config)
    fwd = ForwardProcess(mask_id, schedule=schedule)
if model.requires_terminal_mask_prior and not fwd.has_terminal_mask_prior():
    raise ValueError(
        f"{model_name} sampling starts from all [MASK] tokens; choose a schedule with alpha[-1] = 0"
    )
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
print(f"{model_name.upper()}: {model.num_parameters():,} params")

tc = TrainConfig(
    max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
    log_every=100, eval_every=500, save_every=args.save_every or args.max_steps, save_dir=args.save_dir,
    resume_from=args.resume_from, seed=args.seed,
)
sig = run_signature(tok, {"name": "tinystories", "split": "train", "max_examples": args.max_examples, "mode": "diffusion"}, args.seq_len)
trainer = DiffusionTrainer(model, fwd, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok), eval_dataset=eval_ds)
trainer.train()
model = trainer.model

print("\n--- Samples ---")
model.eval()
sample_steps = min(256, fwd.num_timesteps)
samples = diffusion_sampler(model_name)(model, fwd, batch_size=4, seq_len=args.seq_len, num_steps=sample_steps)
for i in range(4):
    s = [t for t in samples[i].tolist() if t < tok.vocab_size]
    print(f"  {tok.decode(s)[:120]}")
