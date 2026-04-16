"""DPO on Anthropic HH-RLHF.

    python scripts/dpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse
from pathlib import Path
import torch
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT
from minilab.data import load_hh_rlhf
from minilab.alignment import DPOTrainer
from minilab.trainer import TrainConfig, run_signature
from minilab.generation import generate

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--save-dir", default="checkpoints/dpo")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=1000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--max-examples", type=int, default=5000)
p.add_argument("--resume-from", default="")
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)

# Resolve the frozen reference path: on resume, restore from the saved pointer so the
# DPO KL anchor cannot silently drift to the resumed policy.
ref_path = args.checkpoint
if args.resume_from:
    saved = Path(args.resume_from) / "ref_path.txt"
    if saved.exists():
        ref_path = saved.read_text().strip()
assert ref_path, "DPO requires a frozen reference. Pass --checkpoint or resume from a run that saved ref_path.txt."

model_path = args.resume_from or args.checkpoint
model = GPT.load(model_path)
ref_model = GPT.load(ref_path)
print(f"Trainable: {model_path} ({model.num_parameters():,} params)")
print(f"Frozen reference: {ref_path}")

ds = load_hh_rlhf(tok, args.seq_len, max_examples=args.max_examples)
print(f"HH-RLHF: {len(ds)} preference pairs")

tc = TrainConfig(max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
                 dpo_beta=args.beta, log_every=50, eval_every=0,
                 save_every=args.save_every or args.max_steps, save_dir=args.save_dir,
                 resume_from=args.resume_from)
sig = run_signature(tok, {"name": "hh-rlhf", "split": "train", "max_examples": args.max_examples}, args.seq_len)
DPOTrainer(model, ds, tc, ref_model=ref_model, ref_model_path=ref_path, signature=sig).train()

print("\n--- After DPO ---")
model.eval()
for q in ["What makes a good friend?", "How do I learn to cook?", "Tell me about dogs."]:
    ids = tok.encode(q)
    out = generate(model, torch.tensor([ids]), max_new_tokens=100, temperature=0.7, top_k=40)
    print(f"  Q: {q}")
    print(f"  A: {tok.decode(out[0].tolist()[len(ids):])[:120]}\n")
