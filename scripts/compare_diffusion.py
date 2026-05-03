"""Compare diffusion models: MDLM vs SEDD vs D3PM.

    python scripts/compare_diffusion.py --tokenizer tokenizer.json
"""

import argparse
from minilab.checks import require
from minilab.tokenizers import load_tokenizer
from minilab.diffusion import ForwardProcess
from minilab.trainer import DiffusionTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature
from minilab.nn.architecture import MOE_FFNS
from minilab.models.transformer_utils import DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K_EXPERTS
from common import (
    PRETRAIN_EVAL_DATASET_CHOICES,
    attention_uses_gqa,
    build_diffusion_model,
    load_pretrain_dataset,
    load_pretrain_eval_dataset,
    resolve_pretrain_max_examples,
    resolve_default,
)

VARIANTS = [
    ("MDLM", "mdlm", "cosine"),
    ("SEDD", "sedd", "log_linear"),
    ("D3PM", "d3pm", "cosine"),
]


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--dataset", choices=PRETRAIN_EVAL_DATASET_CHOICES, default="tinystories")
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
p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
if args.num_kv_heads is not None:
    require(attention_uses_gqa(args.attention), "--num-kv-heads only applies to GQA attention variants")
if args.num_experts is not None or args.top_k_experts is not None:
    require(args.ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")
set_seed(args.seed)

num_experts = resolve_default(args.num_experts, DEFAULT_NUM_EXPERTS)
top_k_experts = resolve_default(args.top_k_experts, DEFAULT_TOP_K_EXPERTS)

tok = load_tokenizer(args.tokenizer)
mask_id = tok.vocab_size
max_examples = resolve_pretrain_max_examples(args.dataset, args.max_examples, 10000)
train_ds = load_pretrain_dataset(args.dataset, tok, args.seq_len, "train", max_examples, "diffusion")
eval_ds = load_pretrain_eval_dataset(args.dataset, tok, args.seq_len, 1000, "diffusion")
tc = TrainConfig(max_steps=args.max_steps, batch_size=args.batch_size, lr=3e-4,
                 log_every=args.max_steps, eval_every=0, save_every=0, seed=args.seed)
sig = run_signature(tok, {"name": args.dataset, "split": "train", "max_examples": max_examples, "mode": "diffusion"}, args.seq_len)

results = []
for name, model_name, schedule in VARIANTS:
    print(f"\n=== {name} (schedule={schedule}) ===")
    set_seed(args.seed)
    model = build_diffusion_model(
        model_name,
        vocab_size=tok.vocab_size + 1,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        max_seq_len=args.seq_len,
        attention=args.attention,
        ffn=args.ffn,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        mask_token_id=mask_id,
    )
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
