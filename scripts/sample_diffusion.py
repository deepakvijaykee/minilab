"""Sample from a saved diffusion model checkpoint.

    python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
    python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000 --model sedd
"""

import argparse
import torch
from minilab.checks import require
from minilab.tokenizers import load_tokenizer
from minilab.diffusion import ForwardProcess
from minilab.generation import sample_diffusion_dream, sample_diffusion_semi_ar
from minilab.trainer import validate_checkpoint_tokenizer
from common import DIFFUSION_MODEL_CHOICES, diffusion_sampler, load_diffusion_model_checkpoint, reject_supplied, resolve_default

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="override checkpoint model family")
p.add_argument("--sampler", default="full", choices=["full", "semi_ar", "dream"])
p.add_argument("--prompt", default=None)
p.add_argument("--num-samples", type=int, default=5)
p.add_argument("--seq-len", type=int, default=None)
p.add_argument("--max-new-tokens", type=int, default=None)
p.add_argument("--block-size", type=int, default=None)
p.add_argument("--num-steps", type=int, default=None, help="sampler-specific default if omitted")
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--top-k", type=int, default=None)
p.add_argument("--top-p", type=float, default=None)
p.add_argument("--dream-alg", default=None, choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
p.add_argument("--dream-alg-temp", type=float, default=None)
args = p.parse_args()

if args.sampler == "full":
    reject_supplied(args, ("prompt", "max_new_tokens", "block_size", "top_k", "top_p", "dream_alg", "dream_alg_temp"), (
        "does not apply to --sampler full"
    ))
elif args.sampler == "semi_ar":
    reject_supplied(args, ("seq_len",), "only applies to --sampler full")
    reject_supplied(args, ("top_k", "top_p", "dream_alg", "dream_alg_temp"), "only applies to --sampler dream")
elif args.sampler == "dream":
    reject_supplied(args, ("seq_len", "block_size"), "does not apply to --sampler dream")

seq_len = resolve_default(args.seq_len, 256)
max_new_tokens = resolve_default(args.max_new_tokens, 128)
block_size = resolve_default(args.block_size, 16)
top_k = resolve_default(args.top_k, 0)
top_p = resolve_default(args.top_p, 1.0)
dream_alg = resolve_default(args.dream_alg, "entropy")
dream_alg_temp = resolve_default(args.dream_alg_temp, 0.0)

require(args.num_samples > 0, "--num-samples must be > 0")
require(seq_len > 0, "--seq-len must be > 0")
require(max_new_tokens > 0, "--max-new-tokens must be > 0")
require(block_size > 0, "--block-size must be > 0")
require(args.num_steps is None or args.num_steps > 0, "--num-steps must be > 0")
require(args.temperature >= 0, "--temperature must be >= 0")
require(top_k >= 0, "--top-k must be >= 0")
require(0 < top_p <= 1.0, "--top-p must be in (0, 1]")
require(dream_alg_temp >= 0, "--dream-alg-temp must be >= 0")

tok = load_tokenizer(args.tokenizer)
validate_checkpoint_tokenizer(args.checkpoint, tok)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name, model = load_diffusion_model_checkpoint(args.checkpoint, args.model, device=device)
model.eval()
print(f"Loaded {args.checkpoint} ({model_name}) on {device} ({model.num_parameters():,} params)\n")

# Load the exact forward process that was used at training time. Rebuilding from a
# CLI --schedule flag would silently sample the wrong chain for any non-default run.
fwd = ForwardProcess.load(f"{args.checkpoint}/forward_process.json")
print(f"Forward process: schedule={fwd.schedule} num_timesteps={fwd.num_timesteps}\n")
if args.sampler == "full":
    sampler = diffusion_sampler(model_name)
    if args.num_steps is None:
        samples = sampler(model, fwd, batch_size=args.num_samples, seq_len=seq_len, temperature=args.temperature)
    else:
        samples = sampler(
            model,
            fwd,
            batch_size=args.num_samples,
            seq_len=seq_len,
            num_steps=args.num_steps,
            temperature=args.temperature,
        )
else:
    require(args.prompt, "--prompt is required for semi_ar and dream diffusion sampling")
    prompt_token_ids = tok.encode(args.prompt)
    require(prompt_token_ids, "encoded --prompt is empty")
    prompt_ids = torch.tensor([prompt_token_ids], dtype=torch.long).repeat(args.num_samples, 1)
    if args.sampler == "semi_ar":
        samples = sample_diffusion_semi_ar(
            model,
            fwd,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            num_steps=args.num_steps,
            temperature=args.temperature,
        )
    else:
        samples = sample_diffusion_dream(
            model,
            fwd,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            steps=args.num_steps,
            temperature=args.temperature,
            top_p=top_p,
            top_k=top_k,
            alg=dream_alg,
            alg_temp=dream_alg_temp,
        )

for i in range(args.num_samples):
    s = [t for t in samples[i].tolist() if t < tok.vocab_size]
    print(f"--- Sample {i + 1} ---")
    print(tok.decode(s))
    print()
