"""GRPO on GSM8K math problems.

    python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse
from pathlib import Path
import torch
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT
from minilab.data import load_gsm8k
from minilab.alignment import GRPOTrainer
from minilab.trainer import TrainConfig, run_signature
from minilab.tasks.gsm8k import extract_answer, reward as gsm8k_reward
from minilab.generation import generate

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--save-dir", default="checkpoints/grpo")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--num-generations", type=int, default=4)
p.add_argument("--inner-epochs", type=int, default=4, help="PPO-style inner update epochs per rollout; >1 is required for the clip to be active")
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--max-examples", type=int, default=2000)
p.add_argument("--eval-examples", type=int, default=0, help="0 = full GSM8K test split (paper metric); set >0 to subsample for faster debugging")
p.add_argument("--resume-from", default="")
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)

ref_path = args.checkpoint
if args.resume_from:
    saved = Path(args.resume_from) / "ref_path.txt"
    if saved.exists():
        ref_path = saved.read_text().strip()
assert ref_path, "GRPO requires a frozen reference (KL anchor). Pass --checkpoint or resume from a run that saved ref_path.txt."

model_path = args.resume_from or args.checkpoint
model = GPT.load(model_path)
ref_model = GPT.load(ref_path)
print(f"Trainable: {model_path} ({model.num_parameters():,} params)")
print(f"Frozen reference: {ref_path}")

train_ds = load_gsm8k(tok, args.seq_len, max_examples=args.max_examples, split="train")
eval_ds = load_gsm8k(tok, args.seq_len, max_examples=args.eval_examples, split="test")
print(f"GSM8K: train={len(train_ds)} test={len(eval_ds)}")


def math_reward(batch, completions):
    rewards = [
        gsm8k_reward(tok.decode(completions[b].tolist()), train_ds.answers[batch["idx"][b].item()])
        for b in range(completions.size(0))
    ]
    return torch.tensor(rewards)


tc = TrainConfig(
    max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
    grpo_num_generations=args.num_generations, grpo_max_new_tokens=args.max_new_tokens,
    grpo_inner_epochs=args.inner_epochs,
    log_every=50, eval_every=0, save_every=args.save_every or args.max_steps, save_dir=args.save_dir,
    resume_from=args.resume_from,
)
sig = run_signature(tok, {"name": "gsm8k", "split": "train", "max_examples": args.max_examples}, args.seq_len)
GRPOTrainer(model, math_reward, train_ds, tc, ref_model=ref_model, ref_model_path=ref_path, signature=sig).train()

# Evaluate on the held-out test split — the training-set loop below was an optimistic
# debugging signal, not a paper-safe number.
print("\n--- After GRPO (held-out GSM8K test) ---")
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
