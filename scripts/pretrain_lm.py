"""Pretrain a GPT on TinyStories.

    python scripts/pretrain_lm.py --tokenizer tokenizer.json
    python scripts/pretrain_lm.py --tokenizer tokenizer.json --attention iha --connection mhc
"""

import argparse
import torch
from torch.utils.data import DataLoader
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT, GPTConfig
from minilab.data import load_tinystories
from minilab.trainer import LMTrainer, TrainConfig
from minilab.generation import generate
from minilab.evaluation import perplexity

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/lm")
p.add_argument("--dim", type=int, default=256)
p.add_argument("--num-layers", type=int, default=6)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--attention", default="mha")
p.add_argument("--position", default="rope")
p.add_argument("--connection", default="residual")
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max-examples", type=int, default=50000)
p.add_argument("--grad-checkpoint", action="store_true")
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples)
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=2000)
print(f"Data: train={len(train_ds)} eval={len(eval_ds)}")

config = GPTConfig(
    vocab_size=tok.vocab_size, dim=args.dim, num_layers=args.num_layers,
    num_heads=args.num_heads, max_seq_len=args.seq_len,
    attention=args.attention, position=args.position, connection=args.connection,
)
model = GPT(config)
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
print(f"GPT: {model.num_parameters():,} params")

tc = TrainConfig(
    max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
    log_every=100, eval_every=500, save_every=args.max_steps, save_dir=args.save_dir,
)
LMTrainer(model, train_ds, tc, eval_dataset=eval_ds).train()

ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
print(f"\nEval perplexity: {ppl:.1f}")

for text in ["Once upon a time", "The little dog", "She was very happy"]:
    out = generate(model, torch.tensor([tok.encode(text)]), max_new_tokens=80, temperature=0.8, top_k=40)
    print(f"  {tok.decode(out[0].tolist())[:120]}")
