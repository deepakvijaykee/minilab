"""Compare attention variants: MHA vs GQA vs IHA.

    python scripts/compare_attention.py --tokenizer tokenizer.json
"""

import argparse
from torch.utils.data import DataLoader
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT, GPTConfig
from minilab.data import load_tinystories
from minilab.trainer import LMTrainer, TrainConfig, run_signature
from minilab.evaluation import perplexity

VARIANTS = [
    ("MHA", {"attention": "mha"}),
    ("GQA (kv=4)", {"attention": "gqa", "num_kv_heads": 4}),
    ("IHA", {"attention": "iha"}),
]

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--num-layers", type=int, default=4)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--max-steps", type=int, default=2000)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--max-examples", type=int, default=10000)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples)
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=1000)
tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=3e-4,
                 log_every=args.max_steps, eval_every=0, save_every=0)
sig = run_signature(tok, {"name": "tinystories", "split": "train", "max_examples": args.max_examples}, args.seq_len)

results = []
for name, overrides in VARIANTS:
    print(f"\n=== {name} ===")
    cfg = GPTConfig(vocab_size=tok.vocab_size, dim=args.dim, num_layers=args.num_layers,
                    num_heads=args.num_heads, max_seq_len=args.seq_len, **overrides)
    model = GPT(cfg)
    print(f"  {model.num_parameters():,} params")
    trainer = LMTrainer(model, train_ds, tc, signature=sig, eval_dataset=eval_ds)
    trainer.train()
    eval_loss = trainer.evaluate()
    model.eval()
    ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
    results.append((name, model.num_parameters(), eval_loss, ppl))

print(f"\n{'Variant':<15} {'Params':>10} {'Loss':>10} {'PPL':>10}")
print("-" * 48)
for name, params, loss, ppl in results:
    print(f"{name:<15} {params:>10,} {loss:>10.4f} {ppl:>10.1f}")
