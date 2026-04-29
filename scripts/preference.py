"""Offline preference optimization on HH-RLHF or UltraFeedback.

    python scripts/preference.py --algorithm simpo --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse

import torch

from common import MODEL_CHOICES, load_model_checkpoint
from minilab.alignment import (
    CPOTrainConfig,
    CPOTrainer,
    DPOTrainConfig,
    DPOTrainer,
    IPOTrainer,
    KTOTrainConfig,
    KTOTrainer,
    ORPOTrainConfig,
    ORPOTrainer,
    SimPOTrainConfig,
    SimPOTrainer,
    resolve_reference_path,
)
from minilab.checks import require
from minilab.data import load_hh_rlhf, load_hh_rlhf_kto, load_ultrafeedback, load_ultrafeedback_kto
from minilab.generation import generate
from minilab.tokenizers import load_tokenizer
from minilab.trainer import run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer


_PREFERENCE_LOADERS = {"hh": load_hh_rlhf, "ultrafeedback": load_ultrafeedback}
_KTO_LOADERS = {"hh": load_hh_rlhf_kto, "ultrafeedback": load_ultrafeedback_kto}


def _load_dataset(name, algorithm, tok, seq_len, max_examples):
    loaders = _KTO_LOADERS if algorithm == "kto" else _PREFERENCE_LOADERS
    if name in loaders:
        return loaders[name](tok, seq_len, max_examples)
    raise ValueError(f"Unknown dataset: {name}")


p = argparse.ArgumentParser()
p.add_argument("--algorithm", choices=["dpo", "ipo", "cpo", "simpo", "orpo", "kto"], default="dpo")
p.add_argument("--dataset", choices=["hh", "ultrafeedback"], default="hh")
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--save-dir", default="")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=1000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--cpo-alpha", type=float, default=None, help="defaults to 1.0 for CPO")
p.add_argument("--simpo-gamma", type=float, default=None, help="defaults to 0.5 for SimPO")
p.add_argument("--kto-desirable-weight", type=float, default=None, help="defaults to 1.0 for KTO")
p.add_argument("--kto-undesirable-weight", type=float, default=None, help="defaults to 1.0 for KTO")
p.add_argument("--max-examples", type=int, default=5000)
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

if args.algorithm != "cpo":
    require(args.cpo_alpha is None, "--cpo-alpha only applies to --algorithm cpo")
if args.algorithm != "simpo":
    require(args.simpo_gamma is None, "--simpo-gamma only applies to --algorithm simpo")
if args.algorithm != "kto":
    require(args.kto_desirable_weight is None, "--kto-desirable-weight only applies to --algorithm kto")
    require(args.kto_undesirable_weight is None, "--kto-undesirable-weight only applies to --algorithm kto")
cpo_alpha = 1.0 if args.cpo_alpha is None else args.cpo_alpha
simpo_gamma = 0.5 if args.simpo_gamma is None else args.simpo_gamma
kto_desirable_weight = 1.0 if args.kto_desirable_weight is None else args.kto_desirable_weight
kto_undesirable_weight = 1.0 if args.kto_undesirable_weight is None else args.kto_undesirable_weight

set_seed(args.seed)

tok = load_tokenizer(args.tokenizer)
model_path = args.resume_from or args.checkpoint
validate_checkpoint_tokenizer(model_path, tok)
model_name, model = load_model_checkpoint(model_path, args.model)
print(f"Trainable: {model_path} ({model_name}, {model.num_parameters():,} params)")

needs_ref = args.algorithm in {"dpo", "ipo", "kto"}
ref_path = None
if needs_ref:
    ref_path = resolve_reference_path(args.checkpoint, args.resume_from, args.algorithm.upper())
    validate_checkpoint_tokenizer(ref_path, tok)
    print(f"Frozen reference: {ref_path}")

ds = _load_dataset(args.dataset, args.algorithm, tok, args.seq_len, args.max_examples)
print(f"{args.dataset}: {len(ds)} examples for {args.algorithm}")

save_dir = args.save_dir or f"checkpoints/{args.algorithm}"
base_kwargs = dict(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    log_every=50,
    eval_every=0,
    save_every=args.save_every or args.max_steps,
    save_dir=save_dir,
    resume_from=args.resume_from,
    seed=args.seed,
)
sig = run_signature(tok, {"name": args.dataset, "algorithm": args.algorithm, "max_examples": args.max_examples}, args.seq_len)
tok_sig = tokenizer_signature(tok)

if args.algorithm == "dpo":
    tc = DPOTrainConfig(dpo_beta=args.beta, **base_kwargs)
    trainer = DPOTrainer(model, ds, tc, ref_model_path=ref_path, signature=sig, tokenizer_sig=tok_sig)
elif args.algorithm == "ipo":
    tc = DPOTrainConfig(dpo_beta=args.beta, **base_kwargs)
    trainer = IPOTrainer(model, ds, tc, ref_model_path=ref_path, signature=sig, tokenizer_sig=tok_sig)
elif args.algorithm == "kto":
    tc = KTOTrainConfig(
        dpo_beta=args.beta,
        kto_desirable_weight=kto_desirable_weight,
        kto_undesirable_weight=kto_undesirable_weight,
        **base_kwargs,
    )
    trainer = KTOTrainer(model, ds, tc, ref_model_path=ref_path, signature=sig, tokenizer_sig=tok_sig)
elif args.algorithm == "orpo":
    tc = ORPOTrainConfig(orpo_beta=args.beta, **base_kwargs)
    trainer = ORPOTrainer(model, ds, tc, signature=sig, tokenizer_sig=tok_sig)
elif args.algorithm == "cpo":
    tc = CPOTrainConfig(dpo_beta=args.beta, cpo_alpha=cpo_alpha, **base_kwargs)
    trainer = CPOTrainer(model, ds, tc, signature=sig, tokenizer_sig=tok_sig)
else:
    tc = SimPOTrainConfig(dpo_beta=args.beta, simpo_gamma=simpo_gamma, **base_kwargs)
    trainer = SimPOTrainer(model, ds, tc, signature=sig, tokenizer_sig=tok_sig)

trainer.train()
model = trainer.model

print(f"\n--- After {args.algorithm.upper()} ---")
model.eval()
for q in ["What makes a good friend?", "How do I learn to cook?", "Tell me about dogs."]:
    ids = tok.encode(q)
    out = generate(model, torch.tensor([ids]), max_new_tokens=100, temperature=0.7, top_k=40)
    print(f"  Q: {q}")
    print(f"  A: {tok.decode(out[0].tolist()[len(ids):])[:120]}\n")
