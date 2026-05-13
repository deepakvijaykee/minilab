"""Supervised fine-tuning on Alpaca.

    python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
    python scripts/sft.py --tokenizer tokenizer.json  # from scratch
"""

import argparse
import torch
from common import (
    MODEL_CHOICES,
    build_lm_model,
    lm_model_kwargs,
    load_model_checkpoint,
    reject_supplied,
    resolve_default,
    resolve_save_every,
)
from minilab.checks import require
from minilab.presets import get_lm_model_preset, lm_model_preset_choices
from minilab.tokenizers import load_tokenizer
from minilab.data import load_alpaca
from minilab.alignment import SFTTrainer
from minilab.trainer import TrainConfig, run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.generation import generate


_MODEL_BUILD_FLAGS = ("preset", "dim", "num_layers", "num_heads")


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="model family for new runs; inferred from checkpoints")
p.add_argument("--preset", choices=lm_model_preset_choices(), default=None, help="tiny model preset for new runs")
p.add_argument("--save-dir", default="checkpoints/sft")
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--seq-len", type=int, default=None)
p.add_argument("--max-steps", type=int, default=3000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max-examples", type=int, default=10000)
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
require(not (args.checkpoint and args.resume_from), "SFT accepts --checkpoint or --resume-from, not both")

if args.resume_from or args.checkpoint:
    reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
elif model_name in {"mamba", "mamba2"}:
    reject_supplied(args, ("num_heads",), "only applies to --model gpt, hybrid, hymba, xlstm, or byte_latent")

set_seed(args.seed)

dim = resolve_default(args.dim, preset.get("dim", 256))
num_layers = resolve_default(args.num_layers, preset.get("num_layers", 6))
num_heads = resolve_default(args.num_heads, preset.get("num_heads", 8))
seq_len = resolve_default(args.seq_len, preset.get("seq_len", 256))

tok = load_tokenizer(args.tokenizer)

if args.resume_from:
    validate_checkpoint_tokenizer(args.resume_from, tok)
    model_name, model = load_model_checkpoint(args.resume_from, args.model)
    print(f"Resuming from {args.resume_from} ({model_name}, {model.num_parameters():,} params)")
elif args.checkpoint:
    validate_checkpoint_tokenizer(args.checkpoint, tok)
    model_name, model = load_model_checkpoint(args.checkpoint, args.model)
    print(f"Loaded {args.checkpoint} ({model_name}, {model.num_parameters():,} params)")
else:
    config_kwargs = lm_model_kwargs(
        model_name,
        vocab_size=tok.vocab_size,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        max_seq_len=seq_len,
    )
    model = build_lm_model(model_name, **config_kwargs)
    print(f"New model ({model.num_parameters():,} params)")

ds = load_alpaca(tok, seq_len, max_examples=args.max_examples)
print(f"Alpaca: {len(ds)} examples")

tc = TrainConfig(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    log_every=100,
    eval_every=0,
    save_every=resolve_save_every(args.save_every, args.max_steps),
    save_dir=args.save_dir,
    resume_from=args.resume_from,
    seed=args.seed,
)
sig = run_signature(tok, {"name": "alpaca", "split": "train", "max_examples": args.max_examples}, seq_len)
trainer = SFTTrainer(model, ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok))
trainer.train()
model = trainer.model

print("\n--- After SFT ---")
model.eval()
for q in ["Give three tips for staying healthy.", "What is the capital of France?", "Explain gravity."]:
    ids = tok.encode(q)
    out = generate(model, torch.tensor([ids]), max_new_tokens=100, temperature=0.7, top_k=40)
    print(f"  Q: {q}")
    print(f"  A: {tok.decode(out[0].tolist()[len(ids):])[:120]}\n")
