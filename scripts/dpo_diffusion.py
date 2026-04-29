"""Diffusion-DPO on preference pairs.

    python scripts/dpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
"""

import argparse

import torch

from minilab.alignment import DPOTrainConfig, DiffusionDPOTrainer, resolve_reference_path
from minilab.data import load_hh_rlhf_diffusion, load_ultrafeedback_diffusion
from minilab.diffusion import ForwardProcess
from minilab.generation import infill
from minilab.tokenizers import load_tokenizer
from minilab.trainer import run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from common import (
    DIFFUSION_MODEL_CHOICES,
    load_diffusion_model_checkpoint,
    require_checkpoint_path,
)


DATASETS = {"hh-rlhf": load_hh_rlhf_diffusion, "ultrafeedback": load_ultrafeedback_diffusion}

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--save-dir", default="checkpoints/diffusion_dpo")
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="override checkpoint model family")
p.add_argument("--dataset", default="hh-rlhf", choices=list(DATASETS))
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=1000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--max-examples", type=int, default=5000)
p.add_argument("--sample-new-tokens", type=int, default=80)
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

tok = load_tokenizer(args.tokenizer)
model_path = require_checkpoint_path(args.checkpoint, args.resume_from, "Diffusion DPO")
ref_path = resolve_reference_path(args.checkpoint, args.resume_from, "Diffusion DPO")
validate_checkpoint_tokenizer(model_path, tok)
validate_checkpoint_tokenizer(ref_path, tok)

model_name, model = load_diffusion_model_checkpoint(model_path, args.model)
fwd = ForwardProcess.load(f"{model_path}/forward_process.json")
print(f"Trainable: {model_path} ({model_name}, {model.num_parameters():,} params, schedule={fwd.schedule})")
print(f"Frozen reference: {ref_path}")

ds = DATASETS[args.dataset](tok, args.seq_len, max_examples=args.max_examples)
print(f"{args.dataset}: {len(ds)} diffusion preference pairs")

tc = DPOTrainConfig(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    dpo_beta=args.beta,
    log_every=50,
    eval_every=0,
    save_every=args.save_every or args.max_steps,
    save_dir=args.save_dir,
    resume_from=args.resume_from,
    seed=args.seed,
)
sig = run_signature(
    tok,
    {"name": args.dataset, "split": "train", "max_examples": args.max_examples, "mode": "diffusion_dpo"},
    args.seq_len,
)
trainer = DiffusionDPOTrainer(
    model,
    fwd,
    ds,
    tc,
    ref_model_path=ref_path,
    signature=sig,
    tokenizer_sig=tokenizer_signature(tok),
)
trainer.train()
model = trainer.model

print("\n--- After Diffusion DPO ---")
model.eval()
sample_steps = min(128, fwd.num_timesteps)
for q in ["What makes a good friend?", "How do I learn to cook?", "Tell me about dogs."]:
    prompt = tok.encode(q)[: max(1, args.seq_len - args.sample_new_tokens)]
    gen_len = min(args.sample_new_tokens, args.seq_len - len(prompt))
    tokens = prompt + [fwd.mask_token_id] * gen_len
    mask = [False] * len(prompt) + [True] * gen_len
    filled = infill(
        model,
        fwd,
        torch.tensor([tokens], dtype=torch.long),
        torch.tensor([mask], dtype=torch.bool),
        num_steps=sample_steps,
        temperature=0.7,
    )[0]
    answer = [t for t in filled[len(prompt) : len(prompt) + gen_len].tolist() if t < tok.vocab_size]
    print(f"  Q: {q}")
    print(f"  A: {tok.decode(answer)[:120]}\n")
