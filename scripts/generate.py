"""Generate text from a saved GPT checkpoint.

    python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
    python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000 --prompt "Explain gravity."
"""

import argparse
import torch
from minilab.tokenizers import load_tokenizer
from minilab.models.gpt import GPT
from minilab.generation import generate
from minilab.trainer import validate_checkpoint_tokenizer

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--max-new-tokens", type=int, default=200)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--num-samples", type=int, default=3)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
validate_checkpoint_tokenizer(args.checkpoint, tok)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT.load(args.checkpoint, device=device)
model.eval()
print(f"Loaded {args.checkpoint} on {device} ({model.num_parameters():,} params)\n")

prompt_ids = torch.tensor([tok.encode(args.prompt)])
for i in range(args.num_samples):
    out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens,
                   temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    print(f"--- Sample {i + 1} ---")
    print(tok.decode(out[0].tolist()))
    print()
