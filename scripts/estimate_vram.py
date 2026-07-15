"""Rough VRAM estimator for Minilab laptop-scale runs.

This is a planning tool, not a profiler. It estimates the major training-time
components before launching a run so recipes can be adjusted to fit a local GPU.
"""

import argparse

from common import (
    MODEL_CHOICES,
    build_lm_model,
    lm_model_kwargs,
    resolve_default,
)
from minilab.checks import require
from minilab.diagnostics import optimizer_state_bytes
from minilab.online_rl import ONLINE_RL_REFERENCE_ALGORITHMS
from minilab.models.transformer_utils import DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K_EXPERTS
from minilab.presets import (
    LM_MODEL_PRESETS,
    all_model_preset_choices,
    get_any_model_preset,
)


REFERENCE_METHODS = {
    "dpo",
    "ipo",
    "kto",
} | ONLINE_RL_REFERENCE_ALGORITHMS
GROUP_ROLLOUT_METHODS = {
    "grpo", "drgrpo", "grpo_lite", "gspo", "rloo", "dapo",
    "tpo", "tpo_no_anchor", "group_pg", "vpo",
    "dg", "kondo", "uncertainty_dg", "filtered_dg", "reward_variance_dg",
    "aspo", "r2vpo", "replay_dg", "fresh_dg",
}
ROLLOUT_METHODS = {"ppo"} | GROUP_ROLLOUT_METHODS
METHODS = (
    "pretrain",
    "sft",
    "dpo",
    "ipo",
    "cpo",
    "orpo",
    "repo",
    "simpo",
    "kto",
    "ppo",
    "grpo",
    "drgrpo",
    "grpo_lite",
    "gspo",
    "rloo",
    "dapo",
    "tpo",
    "tpo_no_anchor",
    "group_pg",
    "vpo",
    "dg",
    "kondo",
    "uncertainty_dg",
    "filtered_dg",
    "reward_variance_dg",
    "aspo",
    "r2vpo",
    "replay_dg",
    "fresh_dg",
)
FAMILY_CHOICES = MODEL_CHOICES
MODEL_OR_PRESET_CHOICES = all_model_preset_choices() + FAMILY_CHOICES
TRANSFORMER_FAMILIES = {"gpt"}


def _gb(num_bytes):
    return num_bytes / 1024 ** 3


def _fmt_gb(num_bytes):
    return f"{_gb(num_bytes):.2f} GB"


def _optimizer_bytes(model, optimizer):
    if optimizer == "sgd":
        return 0
    return optimizer_state_bytes(model, optimizer=optimizer, dtype_bytes=4)


def _resolve_spec(args):
    if args.model in LM_MODEL_PRESETS:
        spec = get_any_model_preset(args.model)
    else:
        require(args.model in FAMILY_CHOICES, f"Unknown model or preset: {args.model}")
        spec = {
            "kind": "lm",
            "model": args.model,
            "dim": 256,
            "num_layers": 6,
            "num_heads": 8,
            "seq_len": 256,
        }

    spec["dim"] = resolve_default(args.dim, spec["dim"])
    spec["num_layers"] = resolve_default(args.num_layers, spec["num_layers"])
    require("num_heads" in spec, f"{spec['model']} VRAM estimation requires num_heads")
    spec["num_heads"] = resolve_default(args.num_heads, spec["num_heads"])
    spec["seq_len"] = resolve_default(args.seq_len, spec["seq_len"])
    require(spec["dim"] > 0, "--dim must be > 0")
    require(spec["num_layers"] > 0, "--num-layers must be > 0")
    require(spec["seq_len"] > 0, "--seq-len must be > 0")
    require(spec["num_heads"] > 0, "--num-heads must be > 0")
    return spec


def _build_model(spec, vocab_size):
    kwargs = lm_model_kwargs(
        spec["model"],
        vocab_size=vocab_size,
        dim=spec["dim"],
        num_layers=spec["num_layers"],
        num_heads=spec.get("num_heads"),
        max_seq_len=spec["seq_len"],
        attention="mha",
        position="rope",
        norm="rmsnorm",
        connection="residual",
        ffn="swiglu",
        num_experts=DEFAULT_NUM_EXPERTS,
        top_k_experts=DEFAULT_TOP_K_EXPERTS,
    )
    return build_lm_model(spec["model"], **kwargs)


def _activation_bytes(spec, args):
    batch = args.batch_size
    seq_len = spec["seq_len"]
    if args.method in ROLLOUT_METHODS:
        seq_len += args.max_new_tokens
    if args.method in GROUP_ROLLOUT_METHODS:
        batch *= args.num_generations
    if args.method == "vpo":
        batch *= args.vpo_num_candidates

    checkpoint_factor = 2.5 if args.grad_checkpoint else 7.0
    hidden = batch * seq_len * spec["dim"] * spec["num_layers"] * args.activation_dtype_bytes * checkpoint_factor

    attention = 0
    if spec["model"] in TRANSFORMER_FAMILIES:
        heads = spec["num_heads"]
        attention = batch * spec["num_layers"] * heads * seq_len * seq_len * args.activation_dtype_bytes
        if args.grad_checkpoint:
            attention *= 0.35

    return int(hidden + attention)


def _generation_cache_bytes(spec, args):
    if args.method not in ROLLOUT_METHODS or spec["model"] not in TRANSFORMER_FAMILIES:
        return 0
    seq_len = spec["seq_len"] + args.max_new_tokens
    batch = args.batch_size
    if args.method in GROUP_ROLLOUT_METHODS:
        batch *= args.num_generations
    if args.method == "vpo":
        batch *= args.vpo_num_candidates
    return int(batch * spec["num_layers"] * 2 * seq_len * spec["dim"] * args.activation_dtype_bytes)


def _estimate(args):
    spec = _resolve_spec(args)
    require(args.batch_size > 0, "--batch-size must be > 0")
    require(args.num_generations > 0, "--num-generations must be > 0")
    require(args.vpo_num_candidates > 0, "--vpo-num-candidates must be > 0")
    if args.method == "vpo":
        require(args.vpo_num_candidates > 1, "VPO requires --vpo-num-candidates > 1")
    require(args.max_new_tokens >= 0, "--max-new-tokens must be >= 0")
    require(args.vocab_size > 0, "--vocab-size must be > 0")

    model = _build_model(spec, args.vocab_size)
    params = model.num_parameters()
    param_bytes = params * args.param_dtype_bytes
    gradient_bytes = params * args.grad_dtype_bytes
    optimizer_bytes = _optimizer_bytes(model, args.optimizer)
    activation_bytes = _activation_bytes(spec, args)
    cache_bytes = _generation_cache_bytes(spec, args)
    reference_bytes = param_bytes if args.method in REFERENCE_METHODS else 0

    subtotal = param_bytes + gradient_bytes + optimizer_bytes + activation_bytes + cache_bytes + reference_bytes
    overhead_bytes = int(max(args.overhead_gb * 1024 ** 3, subtotal * args.overhead_ratio))
    total = subtotal + overhead_bytes
    return spec, params, {
        "model params": param_bytes,
        "gradients": gradient_bytes,
        "optimizer states": optimizer_bytes,
        "activations": activation_bytes,
        "generation cache": cache_bytes,
        "reference model": reference_bytes,
        "runtime overhead": overhead_bytes,
        "total estimate": total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODEL_OR_PRESET_CHOICES, default="gpt-10m", help="model family or tiny preset")
    p.add_argument("--method", choices=METHODS, default="pretrain")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-generations", type=int, default=4)
    p.add_argument("--vpo-num-candidates", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--optimizer", choices=["adamw", "lion", "muon", "soft_muon", "sgd"], default="adamw")
    p.add_argument("--grad-checkpoint", action="store_true")
    p.add_argument("--param-dtype-bytes", type=int, default=4, help="model weights are fp32 by default under autocast")
    p.add_argument("--grad-dtype-bytes", type=int, default=4)
    p.add_argument("--activation-dtype-bytes", type=int, default=2)
    p.add_argument("--overhead-gb", type=float, default=0.5)
    p.add_argument("--overhead-ratio", type=float, default=0.10)
    p.add_argument("--budget-gb", type=float, default=8.0)
    args = p.parse_args()

    spec, params, parts = _estimate(args)
    print("Estimated VRAM (rough planning estimate)")
    print(f"- model: {args.model} ({spec['model']}, {params:,} params)")
    print(f"- method: {args.method}")
    print(f"- sequence length: {spec['seq_len']}")
    print(f"- batch size: {args.batch_size}")
    if args.method in GROUP_ROLLOUT_METHODS:
        print(f"- rollout generations: {args.num_generations}")
    if args.method == "vpo":
        print(f"- VPO candidates per set: {args.vpo_num_candidates}")
    if args.method in ROLLOUT_METHODS:
        print(f"- max new tokens: {args.max_new_tokens}")
    print()
    for name, value in parts.items():
        print(f"- {name}: {_fmt_gb(value)}")
    print()
    fits = parts["total estimate"] <= args.budget_gb * 1024 ** 3
    verdict = "yes" if fits else "no"
    print(f"Fits {args.budget_gb:.1f} GB budget: {verdict}")
    print("Note: estimator output is approximate; use actual peak memory after a run for published tables.")


if __name__ == "__main__":
    main()
