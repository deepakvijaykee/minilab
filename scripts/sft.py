"""Supervised fine-tuning on Alpaca.

    python scripts/sft.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
    python scripts/sft.py --tokenizer tokenizer.json  # from scratch
"""

import argparse
import torch
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT, GPTConfig
from minilab.data import load_alpaca
from minilab.alignment import SFTTrainer
from minilab.trainer import TrainConfig, run_signature
from minilab.generation import generate

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default=None)
p.add_argument("--save-dir", default="checkpoints/sft")
p.add_argument("--dim", type=int, default=256)
p.add_argument("--num-layers", type=int, default=6)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=3000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max-examples", type=int, default=10000)
p.add_argument("--resume-from", default="")
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)

if args.resume_from:
    model = GPT.load(args.resume_from)
    print(f"Resuming from {args.resume_from} ({model.num_parameters():,} params)")
elif args.checkpoint:
    model = GPT.load(args.checkpoint)
    print(f"Loaded {args.checkpoint} ({model.num_parameters():,} params)")
else:
    config = GPTConfig(vocab_size=tok.vocab_size, dim=args.dim, num_layers=args.num_layers,
                       num_heads=args.num_heads, max_seq_len=args.seq_len)
    model = GPT(config)
    print(f"New model ({model.num_parameters():,} params)")

ds = load_alpaca(tok, args.seq_len, max_examples=args.max_examples)
print(f"Alpaca: {len(ds)} examples")

tc = TrainConfig(max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
                 log_every=100, eval_every=0, save_every=args.save_every or args.max_steps, save_dir=args.save_dir,
                 resume_from=args.resume_from)
sig = run_signature(tok, {"name": "alpaca", "split": "train", "max_examples": args.max_examples}, args.seq_len)
SFTTrainer(model, ds, tc, signature=sig).train()

print("\n--- After SFT ---")
model.eval()
for q in ["Give three tips for staying healthy.", "What is the capital of France?", "Explain gravity."]:
    ids = tok.encode(q)
    out = generate(model, torch.tensor([ids]), max_new_tokens=100, temperature=0.7, top_k=40)
    print(f"  Q: {q}")
    print(f"  A: {tok.decode(out[0].tolist()[len(ids):])[:120]}\n")
