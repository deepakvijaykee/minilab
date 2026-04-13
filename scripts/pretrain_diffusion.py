"""Pretrain a diffusion LM (MDLM, SEDD, or D3PM) on TinyStories.

    python scripts/pretrain_diffusion.py --tokenizer tokenizer.json
    python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --model sedd --schedule log_linear
"""

import argparse
from minilab.tokenizers import load_tokenizer
from minilab.models.mdlm import MDLM, MDLMConfig
from minilab.models.sedd import SEDD, SEDDConfig
from minilab.models.d3pm import D3PM, D3PMConfig
from minilab.data import load_tinystories
from minilab.diffusion import ForwardProcess
from minilab.trainer import DiffusionTrainer, TrainConfig
from minilab.generation import sample_diffusion

MODELS = {"mdlm": (MDLM, MDLMConfig), "sedd": (SEDD, SEDDConfig), "d3pm": (D3PM, D3PMConfig)}

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/diffusion")
p.add_argument("--model", default="mdlm", choices=list(MODELS))
p.add_argument("--dim", type=int, default=256)
p.add_argument("--num-layers", type=int, default=6)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--schedule", default="cosine", choices=["cosine", "linear", "log_linear"])
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max-examples", type=int, default=50000)
p.add_argument("--grad-checkpoint", action="store_true")
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
mask_id = tok.vocab_size

train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples, mode="diffusion")
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=2000, mode="diffusion")
print(f"Data: train={len(train_ds)} eval={len(eval_ds)}")

cls, cfg_cls = MODELS[args.model]
config = cfg_cls(
    vocab_size=tok.vocab_size + 1, dim=args.dim, num_layers=args.num_layers,
    num_heads=args.num_heads, max_seq_len=args.seq_len, mask_token_id=mask_id,
)
model = cls(config)
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
print(f"{args.model.upper()}: {model.num_parameters():,} params")

fwd = ForwardProcess(mask_id, schedule=args.schedule)
tc = TrainConfig(
    max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
    log_every=100, eval_every=500, save_every=args.max_steps, save_dir=args.save_dir,
)
DiffusionTrainer(model, fwd, train_ds, tc, eval_dataset=eval_ds).train()

print("\n--- Samples ---")
samples = sample_diffusion(model, fwd, batch_size=4, seq_len=args.seq_len, num_steps=256)
for i in range(4):
    s = [t for t in samples[i].tolist() if t < tok.vocab_size]
    print(f"  {tok.decode(s)[:120]}")
