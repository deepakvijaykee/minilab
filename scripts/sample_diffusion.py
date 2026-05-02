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
from common import DIFFUSION_MODEL_CHOICES, diffusion_sampler, load_diffusion_model_checkpoint

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="override checkpoint model family")
p.add_argument("--sampler", default="full", choices=["full", "semi_ar", "dream"])
p.add_argument("--prompt", default="")
p.add_argument("--num-samples", type=int, default=5)
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--block-size", type=int, default=16)
p.add_argument("--num-steps", type=int, default=None, help="sampler-specific default if omitted")
p.add_argument("--temperature", type=float, default=1.0)
p.add_argument("--top-k", type=int, default=0)
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--dream-alg", default="entropy", choices=["origin", "maskgit_plus", "topk_margin", "entropy"])
p.add_argument("--dream-alg-temp", type=float, default=0.0)
args = p.parse_args()

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
        samples = sampler(model, fwd, batch_size=args.num_samples, seq_len=args.seq_len, temperature=args.temperature)
    else:
        samples = sampler(
            model,
            fwd,
            batch_size=args.num_samples,
            seq_len=args.seq_len,
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
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            num_steps=args.num_steps,
            temperature=args.temperature,
        )
    else:
        samples = sample_diffusion_dream(
            model,
            fwd,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            steps=args.num_steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            alg=args.dream_alg,
            alg_temp=args.dream_alg_temp,
        )

for i in range(args.num_samples):
    s = [t for t in samples[i].tolist() if t < tok.vocab_size]
    print(f"--- Sample {i + 1} ---")
    print(tok.decode(s))
    print()
