"""Compare diffusion models: MDLM vs SEDD vs D3PM.

    python scripts/compare_diffusion.py --tokenizer tokenizer.json
"""

import argparse
from minilab.checks import require
from minilab.tokenizers import load_tokenizer
from minilab.models.mdlm import MDLM, MDLMConfig
from minilab.models.sedd import SEDD, SEDDConfig
from minilab.models.d3pm import D3PM, D3PMConfig
from minilab.data import load_tinystories
from minilab.diffusion import ForwardProcess
from minilab.trainer import DiffusionTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature
from minilab.nn.architecture import MOE_FFNS
from common import attention_uses_gqa, resolve_default

VARIANTS = [
    ("MDLM", MDLM, MDLMConfig, "cosine"),
    ("SEDD", SEDD, SEDDConfig, "log_linear"),
    ("D3PM", D3PM, D3PMConfig, "cosine"),
]


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--dim", type=int, default=128)
p.add_argument("--num-layers", type=int, default=4)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--attention", default="mha")
p.add_argument("--ffn", default="swiglu")
p.add_argument("--num-experts", type=int, default=None)
p.add_argument("--top-k-experts", type=int, default=None)
p.add_argument("--max-steps", type=int, default=2000)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--max-examples", type=int, default=10000)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
if args.num_kv_heads is not None:
    require(attention_uses_gqa(args.attention), "--num-kv-heads only applies to GQA attention variants")
if args.num_experts is not None or args.top_k_experts is not None:
    require(args.ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")
set_seed(args.seed)

num_experts = resolve_default(args.num_experts, 8)
top_k_experts = resolve_default(args.top_k_experts, 2)

tok = load_tokenizer(args.tokenizer)
mask_id = tok.vocab_size
train_ds = load_tinystories(tok, args.seq_len, max_examples=args.max_examples, mode="diffusion")
eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=1000, mode="diffusion")
tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=3e-4,
                 log_every=args.max_steps, eval_every=0, save_every=0, seed=args.seed)
sig = run_signature(tok, {"name": "tinystories", "split": "train", "max_examples": args.max_examples, "mode": "diffusion"}, args.seq_len)

results = []
for name, cls, cfg_cls, schedule in VARIANTS:
    print(f"\n=== {name} (schedule={schedule}) ===")
    set_seed(args.seed)
    cfg = cfg_cls(vocab_size=tok.vocab_size + 1, dim=args.dim, num_layers=args.num_layers,
                  num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, max_seq_len=args.seq_len,
                  attention=args.attention, ffn=args.ffn, num_experts=num_experts,
                  top_k_experts=top_k_experts, mask_token_id=mask_id)
    model = cls(cfg)
    fwd = ForwardProcess(mask_id, schedule=schedule)
    print(f"  {model.num_parameters():,} params")
    trainer = DiffusionTrainer(model, fwd, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok), eval_dataset=eval_ds)
    trainer.train()
    eval_loss = trainer.evaluate()
    results.append((name, schedule, model.num_parameters(), eval_loss))

print(f"\n{'Model':<10} {'Schedule':<12} {'Params':>10} {'Eval Loss':>12}")
print("-" * 48)
for name, schedule, params, loss in results:
    print(f"{name:<10} {schedule:<12} {params:>10,} {loss:>12.4f}")
