"""Diffusion GRPO on GSM8K math problems.

    python scripts/grpo_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion_sft/step_3000
"""

import argparse

import torch

from minilab.alignment import DiffusionGRPOTrainConfig, DiffusionGRPOTrainer, resolve_reference_path
from minilab.checks import require
from minilab.data import load_gsm8k_diffusion
from minilab.diffusion import ForwardProcess
from minilab.generation import infill
from minilab.tasks.gsm8k import batch_reward, extract_answer, reward as gsm8k_reward
from minilab.tokenizers import load_tokenizer
from minilab.trainer import run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from common import (
    DIFFUSION_MODEL_CHOICES,
    load_diffusion_model_checkpoint,
    require_checkpoint_path,
)


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--save-dir", default="checkpoints/diffusion_grpo")
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="override checkpoint model family")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--num-generations", type=int, default=4)
p.add_argument("--inner-epochs", type=int, default=4, help="PPO-style inner update epochs per rollout")
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--diffusion-steps", type=int, default=128)
p.add_argument("--max-examples", type=int, default=2000)
p.add_argument("--eval-examples", type=int, default=0, help="0 = full GSM8K test split; set >0 for faster debugging")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

tok = load_tokenizer(args.tokenizer)
model_path = require_checkpoint_path(args.checkpoint, args.resume_from, "Diffusion GRPO")
ref_path = resolve_reference_path(args.checkpoint, args.resume_from, "Diffusion GRPO")
validate_checkpoint_tokenizer(model_path, tok)
validate_checkpoint_tokenizer(ref_path, tok)

model_name, model = load_diffusion_model_checkpoint(model_path, args.model)
fwd = ForwardProcess.load(f"{model_path}/forward_process.json")
require(model.supports_unconditional_diffusion_sampling(), (
    "Diffusion GRPO reverse rollouts require a model that can score denoising "
    "steps without clean x_0 context"
))
print(f"Trainable: {model_path} ({model_name}, {model.num_parameters():,} params, schedule={fwd.schedule})")
print(f"Frozen reference: {ref_path}")

train_ds = load_gsm8k_diffusion(
    tok,
    args.seq_len,
    args.max_new_tokens,
    max_examples=args.max_examples,
    split="train",
)
eval_ds = load_gsm8k_diffusion(
    tok,
    args.seq_len,
    args.max_new_tokens,
    max_examples=args.eval_examples,
    split="test",
)
print(f"GSM8K: train={len(train_ds)} test={len(eval_ds)}")


def math_reward(batch, completions, completion_mask):
    return batch_reward(tok, train_ds.answers, batch, completions, completion_mask)


tc = DiffusionGRPOTrainConfig(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    grpo_num_generations=args.num_generations,
    grpo_max_new_tokens=args.max_new_tokens,
    grpo_inner_epochs=args.inner_epochs,
    diffusion_num_steps=args.diffusion_steps,
    log_every=50,
    eval_every=0,
    save_every=args.save_every or args.max_steps,
    save_dir=args.save_dir,
    resume_from=args.resume_from,
    seed=args.seed,
)
sig = run_signature(
    tok,
    {
        "name": "gsm8k",
        "split": "train",
        "max_examples": args.max_examples,
        "mode": "diffusion_grpo",
        "reserved_response_tokens": args.max_new_tokens,
    },
    args.seq_len,
)
trainer = DiffusionGRPOTrainer(
    model,
    fwd,
    math_reward,
    train_ds,
    tc,
    ref_model_path=ref_path,
    signature=sig,
    tokenizer_sig=tokenizer_signature(tok),
)
trainer.train()
model = trainer.model

print("\n--- After Diffusion GRPO (held-out GSM8K test) ---")
model.eval()
correct = 0
total = len(eval_ds)
sample_steps = min(args.diffusion_steps, fwd.num_timesteps)
for i in range(total):
    plen = eval_ds.prompt_lens[i]
    gen_len = min(args.max_new_tokens, args.seq_len - plen)
    prompt = eval_ds.data[i][:plen].clone()
    tokens = torch.cat([prompt, torch.full((gen_len,), fwd.mask_token_id, dtype=torch.long)])
    mask = torch.cat([torch.zeros(plen, dtype=torch.bool), torch.ones(gen_len, dtype=torch.bool)])
    filled = infill(
        model,
        fwd,
        tokens.unsqueeze(0),
        mask.unsqueeze(0),
        num_steps=sample_steps,
        temperature=0,
    )[0]
    text = tok.decode(filled[plen : plen + gen_len].tolist())
    predicted = extract_answer(text)
    expected = eval_ds.answers[i]
    hit = gsm8k_reward(text, expected)
    correct += hit
    if i < 5:
        print(f"  Q: {tok.decode(prompt[:plen].tolist())[:80]}...")
        print(f"  A: {text[:80]}  (predicted={predicted}, expected={expected}, {'OK' if hit else 'WRONG'})\n")

label = "GSM8K test" if args.eval_examples == 0 else f"GSM8K test subset ({total} of full split)"
print(f"{label} accuracy: {correct}/{total} = {correct/total:.1%}")
