"""Generate text from a saved autoregressive checkpoint.

    python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
    python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000 --prompt "Explain gravity."
"""

import argparse
import torch
from minilab.tokenizers import load_tokenizer
from minilab.generation import (
    generate,
    generate_jacobi,
    generate_mtp_speculative,
    generate_mtp_speculative_cached,
    generate_mtp_tree,
    generate_self_speculative,
    generate_self_speculative_shared,
)
from minilab.trainer import validate_checkpoint_tokenizer
from common import MODEL_CHOICES, load_model_checkpoint


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--max-new-tokens", type=int, default=200)
p.add_argument("--temperature", type=float, default=0.8)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--top-p", type=float, default=1.0)
p.add_argument("--num-samples", type=int, default=3)
p.add_argument(
    "--mode",
    choices=[
        "ar",
        "mtp_speculative",
        "mtp_speculative_cached",
        "mtp_tree",
        "self_speculative",
        "self_speculative_shared",
        "jacobi",
    ],
    default="ar",
)
p.add_argument("--draft-tokens", type=int, default=None)
p.add_argument("--exit-layer", type=int, default=None)
p.add_argument("--tree-width", type=int, default=2)
p.add_argument("--tree-depth", type=int, default=None)
p.add_argument("--max-tree-paths", type=int, default=32)
p.add_argument("--jacobi-block-size", type=int, default=4)
p.add_argument("--jacobi-iterations", type=int, default=4)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)
validate_checkpoint_tokenizer(args.checkpoint, tok)
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name, model = load_model_checkpoint(args.checkpoint, args.model, device=device)
model.eval()
print(f"Loaded {args.checkpoint} ({model_name}) on {device} ({model.num_parameters():,} params)\n")

prompt_ids = torch.tensor([tok.encode(args.prompt)])
for i in range(args.num_samples):
    if args.mode == "ar":
        out = generate(model, prompt_ids, max_new_tokens=args.max_new_tokens,
                       temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
    elif args.mode == "mtp_speculative":
        out = generate_mtp_speculative(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.mode == "mtp_speculative_cached":
        out = generate_mtp_speculative_cached(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.mode == "mtp_tree":
        out = generate_mtp_tree(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            tree_width=args.tree_width,
            tree_depth=args.tree_depth,
            max_tree_paths=args.max_tree_paths,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.mode == "self_speculative":
        draft_tokens = 4 if args.draft_tokens is None else args.draft_tokens
        out = generate_self_speculative(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            exit_layer=args.exit_layer,
            draft_tokens=draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.mode == "self_speculative_shared":
        draft_tokens = 4 if args.draft_tokens is None else args.draft_tokens
        out = generate_self_speculative_shared(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            exit_layer=args.exit_layer,
            draft_tokens=draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    elif args.mode == "jacobi":
        out = generate_jacobi(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=args.jacobi_block_size,
            iterations=args.jacobi_iterations,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        raise ValueError(f"Unknown generation mode: {args.mode}")
    print(f"--- Sample {i + 1} ---")
    print(tok.decode(out[0].tolist()))
    print()
