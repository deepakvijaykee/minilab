"""GRPO on GSM8K math problems.

    python scripts/grpo.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000
"""

import argparse
import torch
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT
from minilab.data import load_gsm8k
from minilab.alignment import GRPOTrainer
from minilab.trainer import TrainConfig
from minilab.evaluation import extract_number, accuracy_reward
from minilab.generation import generate

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--save-dir", default="checkpoints/grpo")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-steps", type=int, default=500)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--lr", type=float, default=1e-5)
p.add_argument("--num-generations", type=int, default=4)
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--max-examples", type=int, default=2000)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
model = GPT.load(args.checkpoint)
print(f"Loaded {args.checkpoint} ({model.num_parameters():,} params)")

ds = load_gsm8k(tok, args.seq_len, max_examples=args.max_examples)
print(f"GSM8K: {len(ds)} problems")


def math_reward(batch, completions):
    """Compare extracted numbers from completions against ground truth answers."""
    rewards = []
    for b in range(completions.size(0)):
        text = tok.decode(completions[b].tolist())
        predicted = extract_number(text)
        expected = ds.answers[batch["idx"][b].item()]
        rewards.append(accuracy_reward(predicted, expected) if predicted is not None else 0.0)
    return torch.tensor(rewards)


tc = TrainConfig(
    max_steps=args.max_steps, batch_size=args.batch_size, lr=args.lr,
    grpo_num_generations=args.num_generations, grpo_max_new_tokens=args.max_new_tokens,
    log_every=50, eval_every=0, save_every=args.max_steps, save_dir=args.save_dir,
)
GRPOTrainer(model, math_reward, ds, tc).train()

# Evaluate: generate answers for a few problems and check accuracy
print("\n--- After GRPO ---")
correct = 0
total = min(50, len(ds))
for i in range(total):
    prompt_ids = ds.data[i].unsqueeze(0)
    out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens, temperature=0)
    text = tok.decode(out[0, prompt_ids.size(1):].tolist())
    predicted = extract_number(text)
    expected = ds.answers[i]
    hit = accuracy_reward(predicted, expected) if predicted is not None else 0.0
    correct += hit
    if i < 5:
        print(f"  Q: {tok.decode(prompt_ids[0].tolist())[:80]}...")
        print(f"  A: {text[:80]}  (predicted={predicted}, expected={expected}, {'OK' if hit else 'WRONG'})\n")

print(f"Accuracy: {correct}/{total} = {correct/total:.1%}")
