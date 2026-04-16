"""Compare diffusion models: MDLM vs SEDD vs D3PM.

    python scripts/compare_diffusion.py --tokenizer tokenizer.json
"""

import argparse
from minilab.tokenizers import load_tokenizer
from minilab.models.mdlm import MDLM, MDLMConfig
from minilab.models.sedd import SEDD, SEDDConfig
from minilab.models.d3pm import D3PM, D3PMConfig
from minilab.data import load_tinystories
from minilab.diffusion import ForwardProcess
from minilab.trainer import DiffusionTrainer, TrainConfig, run_signature

VARIANTS = [
    ("MDLM", MDLM, MDLMConfig),
    ("SEDD", SEDD, SEDDConfig),
    ("D3PM", D3PM, D3PMConfig),
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
mask_id = tok.vocab_size
train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples, mode="diffusion")
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=1000, mode="diffusion")
fwd = ForwardProcess(mask_id)
tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=3e-4,
                 log_every=args.max_steps, eval_every=0, save_every=0)
sig = run_signature(tok, {"name": "tinystories", "split": "train", "max_examples": args.max_examples, "mode": "diffusion"}, args.seq_len)

results = []
for name, cls, cfg_cls in VARIANTS:
    print(f"\n=== {name} ===")
    cfg = cfg_cls(vocab_size=tok.vocab_size + 1, dim=args.dim, num_layers=args.num_layers,
                  num_heads=args.num_heads, max_seq_len=args.seq_len, mask_token_id=mask_id)
    model = cls(cfg)
    print(f"  {model.num_parameters():,} params")
    trainer = DiffusionTrainer(model, fwd, train_ds, tc, signature=sig, eval_dataset=eval_ds)
    trainer.train()
    eval_loss = trainer.evaluate()
    results.append((name, model.num_parameters(), eval_loss))

print(f"\n{'Model':<10} {'Params':>10} {'Eval Loss':>12}")
print("-" * 35)
for name, params, loss in results:
    print(f"{name:<10} {params:>10,} {loss:>12.4f}")
