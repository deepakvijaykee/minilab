"""Evaluate a saved autoregressive LM: perplexity and generation diversity.

    python scripts/evaluate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
"""

import argparse
import torch
from torch.utils.data import DataLoader
from minilab.tokenizers import load_tokenizer
from minilab.data import load_tinystories
from minilab.evaluation import perplexity, distinct_n, self_bleu
from minilab.generation import generate
from minilab.trainer import validate_checkpoint_tokenizer
from common import MODEL_CHOICES, load_model_checkpoint


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--num-samples", type=int, default=50)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
validate_checkpoint_tokenizer(args.checkpoint, tok)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name, model = load_model_checkpoint(args.checkpoint, args.model, device=device)
model.eval()
print(f"Loaded {args.checkpoint} ({model_name}) on {device} ({model.num_parameters():,} params)")

eval_ds = load_tinystories(tok, args.seq_len, split="validation", max_examples=2000)
ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
print(f"Perplexity: {ppl:.1f}")

prompts = ["Once upon a time", "The little", "She was", "One day", "There was a"]
texts = []
for i in range(args.num_samples):
    prompt_text = prompts[i % len(prompts)]
    out = generate(model, torch.tensor([tok.encode(prompt_text)]), max_new_tokens=100, temperature=0.9, top_k=50)
    texts.append(tok.decode(out[0].tolist()))

print(f"Distinct-1: {distinct_n(texts, 1):.3f}")
print(f"Distinct-2: {distinct_n(texts, 2):.3f}")
print(f"Distinct-3: {distinct_n(texts, 3):.3f}")
print(f"Self-BLEU-4: {self_bleu(texts, 4):.3f}")

print("\n--- Samples ---")
for t in texts[:5]:
    print(f"  {t[:150]}\n")
