"""Compare future-token and parallel-decoding training objectives.

    python scripts/compare_mtp.py --tokenizer tokenizer.json
"""

import argparse
from torch.utils.data import DataLoader

from common import (
    PRETRAIN_EVAL_DATASET_CHOICES,
    load_pretrain_dataset,
    load_pretrain_eval_dataset,
    model_class,
    resolve_pretrain_max_examples,
)
from minilab.evaluation import perplexity
from minilab.tokenizers import load_tokenizer
from minilab.trainer import LMTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature


def variants(token_superposition_steps):
    return [
        ("NTP", {}, {}),
        ("DeepSeek MTP", {"mtp_depth": 2, "mtp_loss_weight": 0.2}, {}),
        ("Parallel MTP", {"mtp_depth": 2, "mtp_loss_weight": 0.2, "mtp_mode": "parallel"}, {}),
        ("LayerSkip", {"layerskip_loss_weight": 0.2, "layerskip_dropout": 0.1}, {}),
        ("Future Summary", {"future_summary_window": 8, "future_summary_loss_weight": 0.05}, {}),
        ("Jacobi Forcing", {"jacobi_loss_weight": 0.1, "jacobi_iterations": 1}, {}),
        ("Token Superpos", {}, {
            "token_superposition_size": 4,
            "token_superposition_steps": token_superposition_steps,
        }),
    ]


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--dataset", choices=PRETRAIN_EVAL_DATASET_CHOICES, default="tinystories")
p.add_argument("--dim", type=int, default=128)
p.add_argument("--num-layers", type=int, default=4)
p.add_argument("--num-heads", type=int, default=8)
p.add_argument("--seq-len", type=int, default=128)
p.add_argument("--max-steps", type=int, default=1000)
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

tok = load_tokenizer(args.tokenizer)
max_examples = resolve_pretrain_max_examples(args.dataset, args.max_examples, 10000)
train_ds = load_pretrain_dataset(args.dataset, tok, args.seq_len, "train", max_examples, "lm")
eval_ds = load_pretrain_eval_dataset(args.dataset, tok, args.seq_len, 1000, "lm")
signature = run_signature(tok, {"name": args.dataset, "split": "train", "max_examples": max_examples}, args.seq_len)
token_superposition_steps = max(1, args.max_steps // 2)

results = []
gpt_cls = model_class("gpt")
for name, model_fields, train_fields in variants(token_superposition_steps):
    print(f"\n=== {name} ===")
    set_seed(args.seed)
    cfg = gpt_cls.config_class(
        vocab_size=tok.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_len,
        **model_fields,
    )
    model = gpt_cls(cfg)
    train_cfg = TrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=3e-4,
        log_every=args.max_steps,
        eval_every=0,
        save_every=0,
        seed=args.seed,
        **train_fields,
    )
    trainer = LMTrainer(
        model,
        train_ds,
        train_cfg,
        signature=signature,
        tokenizer_sig=tokenizer_signature(tok),
        eval_dataset=eval_ds,
    )
    trainer.train()
    eval_loss = trainer.evaluate()
    model.eval()
    ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
    results.append((name, model.num_parameters(), eval_loss, ppl))

print(f"\n{'Variant':<16} {'Params':>10} {'Loss':>10} {'PPL':>10}")
print("-" * 50)
for name, params, loss, ppl in results:
    print(f"{name:<16} {params:>10,} {loss:>10.4f} {ppl:>10.1f}")
