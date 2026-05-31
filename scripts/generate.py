"""Generate text from a saved autoregressive checkpoint.

    python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/lm/step_5000
    python scripts/generate.py --tokenizer tokenizer.json --checkpoint checkpoints/sft/step_3000 --prompt "Explain gravity."
"""

import argparse
import torch
from minilab.checks import require
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
from common import MODEL_CHOICES, load_model_checkpoint, reject_supplied, resolve_default


_VERIFIED_MODES = {
    "mtp_speculative",
    "mtp_speculative_cached",
    "mtp_tree",
    "self_speculative",
    "self_speculative_shared",
    "jacobi",
}
_DRAFT_TOKEN_MODES = {
    "mtp_speculative",
    "mtp_speculative_cached",
    "self_speculative",
    "self_speculative_shared",
}
_SELF_SPECULATIVE_MODES = {"self_speculative", "self_speculative_shared"}
_TREE_FLAGS = ("tree_width", "tree_depth", "max_tree_paths")
_JACOBI_FLAGS = ("jacobi_block_size", "jacobi_iterations")


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
p.add_argument("--prompt", default="Once upon a time")
p.add_argument("--max-new-tokens", type=int, default=200)
p.add_argument("--temperature", type=float, default=None)
p.add_argument("--top-k", type=int, default=None)
p.add_argument("--top-p", type=float, default=None)
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
p.add_argument("--tree-width", type=int, default=None)
p.add_argument("--tree-depth", type=int, default=None)
p.add_argument("--max-tree-paths", type=int, default=None)
p.add_argument("--jacobi-block-size", type=int, default=None)
p.add_argument("--jacobi-iterations", type=int, default=None)
args = p.parse_args()

if args.mode in _VERIFIED_MODES:
    if args.temperature is not None:
        require(args.temperature == 0, "--temperature must be 0 for exact verified decoding modes")
    reject_supplied(args, ("top_k", "top_p"), "does not apply to exact verified decoding modes")
    temperature = 0.0
    top_k = 0
    top_p = 1.0
else:
    temperature = resolve_default(args.temperature, 0.8)
    top_k = resolve_default(args.top_k, 50)
    top_p = resolve_default(args.top_p, 1.0)

if args.mode not in _DRAFT_TOKEN_MODES:
    reject_supplied(args, ("draft_tokens",), "only applies to MTP or self-speculative decoding modes")
if args.mode not in _SELF_SPECULATIVE_MODES:
    reject_supplied(args, ("exit_layer",), "only applies to self-speculative decoding modes")
if args.mode != "mtp_tree":
    reject_supplied(args, _TREE_FLAGS, "only applies to --mode mtp_tree")
if args.mode != "jacobi":
    reject_supplied(args, _JACOBI_FLAGS, "only applies to --mode jacobi")

tree_width = resolve_default(args.tree_width, 2)
max_tree_paths = resolve_default(args.max_tree_paths, 32)
jacobi_block_size = resolve_default(args.jacobi_block_size, 4)
jacobi_iterations = resolve_default(args.jacobi_iterations, 4)

require(args.max_new_tokens >= 0, "--max-new-tokens must be >= 0")
require(args.num_samples > 0, "--num-samples must be > 0")
require(temperature >= 0, "--temperature must be >= 0")
require(top_k >= 0, "--top-k must be >= 0")
require(0 < top_p <= 1.0, "--top-p must be in (0, 1]")
require(args.draft_tokens is None or args.draft_tokens > 0, "--draft-tokens must be > 0")
require(args.exit_layer is None or args.exit_layer > 0, "--exit-layer must be > 0")
require(tree_width > 0, "--tree-width must be > 0")
require(args.tree_depth is None or args.tree_depth > 0, "--tree-depth must be > 0")
require(max_tree_paths > 0, "--max-tree-paths must be > 0")
require(jacobi_block_size > 0, "--jacobi-block-size must be > 0")
require(jacobi_iterations > 0, "--jacobi-iterations must be > 0")

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
                       temperature=temperature, top_k=top_k, top_p=top_p)
    elif args.mode == "mtp_speculative":
        out = generate_mtp_speculative(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif args.mode == "mtp_speculative_cached":
        out = generate_mtp_speculative_cached(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif args.mode == "mtp_tree":
        out = generate_mtp_tree(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            tree_width=tree_width,
            tree_depth=args.tree_depth,
            max_tree_paths=max_tree_paths,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif args.mode == "self_speculative":
        draft_tokens = 4 if args.draft_tokens is None else args.draft_tokens
        out = generate_self_speculative(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            exit_layer=args.exit_layer,
            draft_tokens=draft_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif args.mode == "self_speculative_shared":
        draft_tokens = 4 if args.draft_tokens is None else args.draft_tokens
        out = generate_self_speculative_shared(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            exit_layer=args.exit_layer,
            draft_tokens=draft_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    elif args.mode == "jacobi":
        out = generate_jacobi(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=jacobi_block_size,
            iterations=jacobi_iterations,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    else:
        raise ValueError(f"Unknown generation mode: {args.mode}")
    print(f"--- Sample {i + 1} ---")
    print(tok.decode(out[0].tolist()))
    print()
