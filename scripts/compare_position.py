"""Compare position encodings.

    python scripts/compare_position.py --tokenizer tokenizer.json
"""

import argparse
from common import (
    PRETRAIN_DATASET_CHOICES,
    compare_lm_variants,
    load_pretrain_dataset,
    load_pretrain_eval_dataset,
    resolve_pretrain_max_examples,
)
from minilab.tokenizers import load_tokenizer
from minilab.trainer import TrainConfig, run_signature, set_seed

VARIANTS = [
    ("RoPE", {"position": "rope"}),
    ("ALiBi", {"position": "alibi"}),
    ("T5 relative", {"position": "t5_relative"}),
    ("KERPLE log", {"position": "kerple_log"}),
    ("KERPLE power", {"position": "kerple_power"}),
    ("Learned", {"position": "learned"}),
    ("Sinusoidal", {"position": "sinusoidal"}),
    ("NoPE", {"position": "none"}),
]

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--dataset", choices=PRETRAIN_DATASET_CHOICES, default="tinystories")
p.add_argument("--dim", type=int, default=128)
p.add_argument("--num-layers", type=int, default=4)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--max-steps", type=int, default=2000)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

tok = load_tokenizer(args.tokenizer)
max_examples = resolve_pretrain_max_examples(args.dataset, args.max_examples, 10000)
train_ds = load_pretrain_dataset(args.dataset, tok, args.seq_len, "train", max_examples, "lm")
eval_ds = load_pretrain_eval_dataset(args.dataset, tok, args.seq_len, 1000, "lm")
tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=3e-4,
                 log_every=args.max_steps, eval_every=0, save_every=0, seed=args.seed)
sig = run_signature(tok, {"name": args.dataset, "split": "train", "max_examples": max_examples}, args.seq_len)

compare_lm_variants(
    VARIANTS,
    tok,
    train_ds,
    eval_ds,
    tc,
    sig,
    seed=args.seed,
    dim=args.dim,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    seq_len=args.seq_len,
)
