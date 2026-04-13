"""DPO on Anthropic HH-RLHF.

    python scripts/dpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse
import torch
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT
from minilab.data import load_hh_rlhf
from minilab.alignment import DPOTrainer
from minilab.trainer import TrainConfig
from minilab.generation import generate

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--save-dir", default="checkpoints/dpo")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=1000)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--max-examples", type=int, default=5000)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
model = GPT.load(args.checkpoint)
print(f"Loaded {args.checkpoint} ({model.num_parameters():,} params)")

ds = load_hh_rlhf(tok, args.seq_len, max_examples=args.max_examples)
print(f"HH-RLHF: {len(ds)} preference pairs")

tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
                 dpo_beta=args.beta, log_every=50, eval_every=0,
                 save_every=args.max_steps, save_dir=args.save_dir)
DPOTrainer(model, ds, tc).train()

print("\n--- After DPO ---")
for q in ["What makes a good friend?", "How do I learn to cook?", "Tell me about dogs."]:
    ids = tok.encode(q)
    out = generate(model, torch.tensor([ids]), max_new_tokens=100, temperature=0.7, top_k=40)
    print(f"  Q: {q}")
    print(f"  A: {tok.decode(out[0].tolist()[len(ids):])[:120]}\n")
