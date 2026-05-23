"""Benchmark autoregressive and verified generation paths for a saved checkpoint.

Example:
    python scripts/benchmark_generation.py --tokenizer tokenizer.json \
        --checkpoint checkpoints/lm/step_5000 --modes ar mtp_speculative_cached
"""

import argparse
import time

import torch

from common import MODEL_CHOICES, load_model_checkpoint
from minilab.generation import (
    generate,
    generate_jacobi,
    generate_mtp_speculative,
    generate_mtp_speculative_cached,
    generate_mtp_tree,
    generate_self_speculative,
    generate_self_speculative_shared,
)
from minilab.tokenizers import load_tokenizer
from minilab.trainer import validate_checkpoint_tokenizer


GENERATION_MODES = [
    "ar",
    "mtp_speculative",
    "mtp_speculative_cached",
    "mtp_tree",
    "self_speculative",
    "self_speculative_shared",
    "jacobi",
]


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _run_generation(model, prompt_ids, args, mode):
    if mode == "ar":
        return generate(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            use_cache=not args.no_cache,
        )
    if mode == "mtp_speculative":
        return generate_mtp_speculative(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    if mode == "mtp_speculative_cached":
        return generate_mtp_speculative_cached(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            draft_tokens=args.draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    if mode == "mtp_tree":
        return generate_mtp_tree(
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
    if mode == "self_speculative":
        draft_tokens = 4 if args.draft_tokens is None else args.draft_tokens
        return generate_self_speculative(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            exit_layer=args.exit_layer,
            draft_tokens=draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    if mode == "self_speculative_shared":
        draft_tokens = 4 if args.draft_tokens is None else args.draft_tokens
        return generate_self_speculative_shared(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            exit_layer=args.exit_layer,
            draft_tokens=draft_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    if mode == "jacobi":
        return generate_jacobi(
            model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=args.jacobi_block_size,
            iterations=args.jacobi_iterations,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    raise ValueError(f"Unknown generation mode: {mode}")


def _time_mode(model, prompt_ids, args, mode, device):
    for _ in range(args.warmup):
        _run_generation(model, prompt_ids, args, mode)
    _sync(device)

    total_seconds = 0.0
    total_new_tokens = 0
    last_out = None
    for _ in range(args.runs):
        start = time.perf_counter()
        last_out = _run_generation(model, prompt_ids, args, mode)
        _sync(device)
        elapsed = time.perf_counter() - start
        total_seconds += elapsed
        total_new_tokens += max(0, last_out.size(1) - prompt_ids.size(1))

    tokens_per_second = total_new_tokens / total_seconds if total_seconds > 0 else 0.0
    return {
        "mode": mode,
        "seconds": total_seconds,
        "new_tokens": total_new_tokens,
        "tokens_per_second": tokens_per_second,
        "output_tokens": 0 if last_out is None else last_out.size(1),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=None, help="override checkpoint model family")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--modes", nargs="+", choices=GENERATION_MODES, default=["ar"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true", help="disable KV cache for ar mode")
    parser.add_argument("--draft-tokens", type=int, default=None)
    parser.add_argument("--exit-layer", type=int, default=None)
    parser.add_argument("--tree-width", type=int, default=2)
    parser.add_argument("--tree-depth", type=int, default=None)
    parser.add_argument("--max-tree-paths", type=int, default=32)
    parser.add_argument("--jacobi-block-size", type=int, default=4)
    parser.add_argument("--jacobi-iterations", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    tok = load_tokenizer(args.tokenizer)
    validate_checkpoint_tokenizer(args.checkpoint, tok)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, model = load_model_checkpoint(args.checkpoint, args.model, device=device)
    model.eval()

    prompt_ids = torch.tensor([tok.encode(args.prompt)], device=device)
    print("mode\tseconds\tnew_tokens\ttokens_per_second\toutput_tokens")
    for mode in args.modes:
        row = _time_mode(model, prompt_ids, args, mode, device)
        print(
            f"{row['mode']}\t{row['seconds']:.4f}\t{row['new_tokens']}\t"
            f"{row['tokens_per_second']:.2f}\t{row['output_tokens']}"
        )


if __name__ == "__main__":
    main()
