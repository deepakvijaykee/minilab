"""GRPO on GSM8K math problems.

    python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse
from common import MODEL_CHOICES, load_model_checkpoint, require_checkpoint_path, resolve_default
from minilab.tokenizers import load_tokenizer
from minilab.data import load_gsm8k
from minilab.alignment import (
    DAPOTrainConfig,
    DAPOTrainer,
    GRPOTrainConfig,
    GRPOTrainer,
    GSPOTrainConfig,
    GSPOTrainer,
    PPOTrainConfig,
    PPOTrainer,
    RLOOTrainConfig,
    RLOOTrainer,
    resolve_reference_path,
)
from minilab.checks import require
from minilab.trainer import run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.tasks.gsm8k import batch_reward, extract_answer, reward as gsm8k_reward
from minilab.generation import generate


p = argparse.ArgumentParser()
p.add_argument("--algorithm", choices=["ppo", "grpo", "dapo", "gspo", "rloo"], default="grpo")
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--save-dir", default="")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--num-generations", type=int, default=None, help="defaults to 4 for group policy algorithms")
p.add_argument("--inner-epochs", type=int, default=None, help="inner update epochs per rollout; defaults to 1 for RLOO and 4 otherwise")
p.add_argument("--kl-coef", type=float, default=None, help="defaults to 0.0 for DAPO and 0.1 otherwise")
p.add_argument(
    "--clip-ratio",
    type=float,
    default=None,
    help="defaults to 0.2 for PPO/GRPO, 4e-4 for GSPO; DAPO and RLOO reject it",
)
p.add_argument("--value-clip", type=float, default=None, help="defaults to 0.2 for PPO")
p.add_argument("--value-coef", type=float, default=None, help="defaults to 0.5 for PPO")
p.add_argument("--entropy-coef", type=float, default=None, help="defaults to 0.0 for PPO")
p.add_argument("--gae-lambda", type=float, default=None, help="defaults to 0.95 for PPO")
p.add_argument("--clip-ratio-low", type=float, default=None, help="defaults to 0.2 for DAPO")
p.add_argument("--clip-ratio-high", type=float, default=None, help="defaults to 0.28 for DAPO")
p.add_argument("--safe-length", type=int, default=None, help="defaults to 0 for DAPO")
p.add_argument("--length-penalty", type=float, default=None, help="defaults to 0.0 for DAPO")
p.add_argument("--max-resample", type=int, default=None, help="defaults to 5 for DAPO")
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--max-examples", type=int, default=2000)
p.add_argument("--eval-examples", type=int, default=0, help="0 = full GSM8K test split (paper metric); set >0 to subsample for faster debugging")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
set_seed(args.seed)

if args.algorithm != "ppo":
    require(args.value_clip is None, "--value-clip only applies to --algorithm ppo")
    require(args.value_coef is None, "--value-coef only applies to --algorithm ppo")
    require(args.entropy_coef is None, "--entropy-coef only applies to --algorithm ppo")
    require(args.gae_lambda is None, "--gae-lambda only applies to --algorithm ppo")
if args.algorithm == "ppo":
    require(args.num_generations is None, "--num-generations only applies to group policy algorithms")
if args.algorithm != "dapo":
    require(args.clip_ratio_low is None, "--clip-ratio-low only applies to --algorithm dapo")
    require(args.clip_ratio_high is None, "--clip-ratio-high only applies to --algorithm dapo")
    require(args.safe_length is None, "--safe-length only applies to --algorithm dapo")
    require(args.length_penalty is None, "--length-penalty only applies to --algorithm dapo")
    require(args.max_resample is None, "--max-resample only applies to --algorithm dapo")

num_generations = resolve_default(args.num_generations, 4)
value_clip = resolve_default(args.value_clip, 0.2)
value_coef = resolve_default(args.value_coef, 0.5)
entropy_coef = resolve_default(args.entropy_coef, 0.0)
gae_lambda = resolve_default(args.gae_lambda, 0.95)
clip_ratio_low = resolve_default(args.clip_ratio_low, 0.2)
clip_ratio_high = resolve_default(args.clip_ratio_high, 0.28)
safe_length = resolve_default(args.safe_length, 0)
length_penalty = resolve_default(args.length_penalty, 0.0)
max_resample = resolve_default(args.max_resample, 5)
inner_epochs = args.inner_epochs
if inner_epochs is None:
    inner_epochs = 1 if args.algorithm == "rloo" else 4
kl_coef = args.kl_coef
if kl_coef is None:
    kl_coef = 0.0 if args.algorithm == "dapo" else 0.1
if args.algorithm == "dapo":
    require(kl_coef == 0, "DAPO removes the KL penalty; set --kl-coef 0 or leave it unset")
    require(args.clip_ratio is None, "DAPO uses --clip-ratio-low/--clip-ratio-high; do not set --clip-ratio")
if args.algorithm == "rloo":
    require(args.clip_ratio is None, "RLOO is an unclipped REINFORCE estimator; do not set --clip-ratio")
clip_ratio_default = 4e-4 if args.algorithm == "gspo" else 0.2
clip_ratio = resolve_default(args.clip_ratio, clip_ratio_default)

tok = load_tokenizer(args.tokenizer)

model_path = require_checkpoint_path(args.checkpoint, args.resume_from, "GRPO training")
validate_checkpoint_tokenizer(model_path, tok)
ref_path = None
if args.algorithm != "dapo":
    ref_path = resolve_reference_path(args.checkpoint, args.resume_from, args.algorithm.upper())
    validate_checkpoint_tokenizer(ref_path, tok)
model_name, model = load_model_checkpoint(model_path, args.model)
print(f"Trainable: {model_path} ({model_name}, {model.num_parameters():,} params)")
if ref_path is not None:
    print(f"Frozen reference: {ref_path}")

train_ds = load_gsm8k(tok, args.seq_len, max_examples=args.max_examples, split="train")
eval_ds = load_gsm8k(tok, args.seq_len, max_examples=args.eval_examples, split="test")
print(f"GSM8K: train={len(train_ds)} test={len(eval_ds)}")


def math_reward(batch, completions, completion_mask):
    return batch_reward(tok, train_ds.answers, batch, completions, completion_mask)


base_config = dict(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    log_every=50,
    eval_every=0,
    save_every=args.save_every or args.max_steps,
    save_dir=args.save_dir or f"checkpoints/{args.algorithm}",
    resume_from=args.resume_from,
    seed=args.seed,
)
if args.algorithm == "ppo":
    tc = PPOTrainConfig(
        ppo_max_new_tokens=args.max_new_tokens,
        ppo_inner_epochs=inner_epochs,
        ppo_kl_coef=kl_coef,
        ppo_clip_ratio=clip_ratio,
        ppo_value_clip=value_clip,
        ppo_value_coef=value_coef,
        ppo_entropy_coef=entropy_coef,
        ppo_lam=gae_lambda,
        **base_config,
    )
elif args.algorithm == "dapo":
    tc = DAPOTrainConfig(
        grpo_num_generations=num_generations,
        grpo_max_new_tokens=args.max_new_tokens,
        grpo_inner_epochs=inner_epochs,
        grpo_kl_coef=kl_coef,
        dapo_clip_ratio_low=clip_ratio_low,
        dapo_clip_ratio_high=clip_ratio_high,
        dapo_safe_length=safe_length,
        dapo_length_penalty=length_penalty,
        dapo_max_resample=max_resample,
        **base_config,
    )
else:
    config_cls = {
        "grpo": GRPOTrainConfig,
        "gspo": GSPOTrainConfig,
        "rloo": RLOOTrainConfig,
    }[args.algorithm]
    policy_kwargs = dict(
        grpo_num_generations=num_generations,
        grpo_max_new_tokens=args.max_new_tokens,
        grpo_inner_epochs=inner_epochs,
        grpo_kl_coef=kl_coef,
    )
    if args.algorithm in {"grpo", "gspo"}:
        policy_kwargs["grpo_clip_ratio"] = clip_ratio
    tc = config_cls(**policy_kwargs, **base_config)
sig = run_signature(tok, {"name": "gsm8k", "split": "train", "algorithm": args.algorithm, "max_examples": args.max_examples}, args.seq_len)
trainer_cls = {
    "ppo": PPOTrainer,
    "grpo": GRPOTrainer,
    "dapo": DAPOTrainer,
    "gspo": GSPOTrainer,
    "rloo": RLOOTrainer,
}[args.algorithm]
if args.algorithm == "dapo":
    trainer = trainer_cls(model, math_reward, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok))
else:
    trainer = trainer_cls(model, math_reward, train_ds, tc, ref_model_path=ref_path, signature=sig, tokenizer_sig=tokenizer_signature(tok))
trainer.train()
model = trainer.model

# Evaluate on the held-out test split — the training-set loop below was an optimistic
# debugging signal, not a paper-safe number.
print(f"\n--- After {args.algorithm.upper()} (held-out GSM8K test) ---")
model.eval()
correct = 0
total = len(eval_ds)
for i in range(total):
    plen = eval_ds.prompt_lens[i]
    prompt_ids = eval_ds.data[i][:plen].unsqueeze(0)
    out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0)
    text = tok.decode(out[0, plen:].tolist())
    predicted = extract_answer(text)
    expected = eval_ds.answers[i]
    hit = gsm8k_reward(text, expected)
    correct += hit
    if i < 5:
        print(f"  Q: {tok.decode(prompt_ids[0].tolist())[:80]}...")
        print(f"  A: {text[:80]}  (predicted={predicted}, expected={expected}, {'OK' if hit else 'WRONG'})\n")

label = "GSM8K test" if args.eval_examples == 0 else f"GSM8K test subset ({total} of full split)"
print(f"{label} accuracy: {correct}/{total} = {correct/total:.1%}")
