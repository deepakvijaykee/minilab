"""Pretrain a diffusion LM (MDLM, SEDD, D3PM, or BlockDiffusionLM).

    python scripts/pretrain_diffusion.py --tokenizer tokenizer.json
    python scripts/pretrain_diffusion.py --tokenizer tokenizer.json --model sedd --schedule log_linear
"""

import argparse
from minilab.checks import require
from minilab.presets import get_diffusion_model_preset, diffusion_model_preset_choices
from minilab.tokenizers import load_tokenizer
from minilab.diffusion import ForwardProcess
from minilab.trainer import DiffusionTrainer, TrainConfig, run_signature, set_seed, tokenizer_signature, validate_checkpoint_tokenizer
from minilab.nn.architecture import MOE_FFNS
from minilab.models.transformer_utils import DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K_EXPERTS
from common import (
    DIFFUSION_MODEL_CHOICES,
    PRETRAIN_DATASET_CHOICES,
    attention_uses_gqa,
    build_diffusion_model,
    diffusion_model_kwargs,
    diffusion_sampler,
    load_pretrain_dataset,
    load_pretrain_eval_dataset,
    load_diffusion_model_checkpoint,
    reject_supplied,
    resolve_pretrain_max_examples,
    resolve_default,
    resolve_save_every,
)


_MODEL_BUILD_FLAGS = (
    "preset", "dim", "num_layers", "num_heads", "num_kv_heads",
    "attention", "ffn", "num_experts", "top_k_experts",
    "block_size", "block_diffusion_unconditional", "antithetic_time_sampling",
)
_BLOCK_DIFFUSION_FLAGS = (
    "block_size", "block_diffusion_unconditional", "antithetic_time_sampling",
)


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--save-dir", default="checkpoints/diffusion")
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="model family for new runs; inferred from checkpoints")
p.add_argument("--preset", choices=diffusion_model_preset_choices(), default=None, help="tiny diffusion model preset for new runs")
p.add_argument("--dataset", choices=PRETRAIN_DATASET_CHOICES, default="tinystories")
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=None)
p.add_argument("--attention", default=None)
p.add_argument("--ffn", default=None)
p.add_argument("--num-experts", type=int, default=None)
p.add_argument("--top-k-experts", type=int, default=None)
p.add_argument("--block-size", type=int, default=None, help="BlockDiffusionLM block size")
p.add_argument(
    "--block-diffusion-unconditional",
    action="store_true",
    default=None,
    help="build BlockDiffusionLM without clean-context cross attention for generic samplers",
)
p.add_argument(
    "--antithetic-time-sampling",
    action="store_true",
    default=None,
    help="use paired block timesteps for BlockDiffusionLM training",
)
p.add_argument("--schedule", default=None, choices=["cosine", "linear", "geometric", "log_linear"])
p.add_argument("--max-steps", type=int, default=5000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max-examples", type=int, default=None)
p.add_argument("--grad-checkpoint", action="store_true")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
preset = get_diffusion_model_preset(args.preset) if args.preset else {}
if args.preset and args.model is not None:
    require(
        args.model == preset["model"],
        f"--model {args.model} conflicts with --preset {args.preset} ({preset['model']})",
    )
model_name = preset.get("model") or args.model or "mdlm"

if args.resume_from:
    reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
    require(args.schedule is None, "--schedule only applies when starting a new model")

set_seed(args.seed)

dim = resolve_default(args.dim, preset.get("dim", 256))
num_layers = resolve_default(args.num_layers, preset.get("num_layers", 6))
num_heads = resolve_default(args.num_heads, preset.get("num_heads", 8))
seq_len = resolve_default(args.seq_len, preset.get("seq_len", 256))
attention = resolve_default(args.attention, "mha")
ffn = resolve_default(args.ffn, "swiglu")
num_experts = resolve_default(args.num_experts, DEFAULT_NUM_EXPERTS)
top_k_experts = resolve_default(args.top_k_experts, DEFAULT_TOP_K_EXPERTS)

if not args.resume_from:
    if args.num_kv_heads is not None:
        require(attention_uses_gqa(attention), "--num-kv-heads only applies to GQA attention variants")
    if args.num_experts is not None or args.top_k_experts is not None:
        require(ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")
    if model_name != "block_diffusion":
        reject_supplied(args, _BLOCK_DIFFUSION_FLAGS, "only applies to --model block_diffusion")

tok = load_tokenizer(args.tokenizer)
mask_id = tok.vocab_size

max_examples = resolve_pretrain_max_examples(args.dataset, args.max_examples, 50000)
train_ds = load_pretrain_dataset(args.dataset, tok, seq_len, "train", max_examples, "diffusion")
eval_ds = (
    None
    if args.dataset == "openwebtext"
    else load_pretrain_eval_dataset(args.dataset, tok, seq_len, 2000, "diffusion")
)
eval_count = "none" if eval_ds is None else len(eval_ds)
print(f"Data: {args.dataset} train={len(train_ds)} eval={eval_count}")

if args.resume_from:
    validate_checkpoint_tokenizer(args.resume_from, tok)
    model_name, model = load_diffusion_model_checkpoint(args.resume_from, args.model)
    fwd = ForwardProcess.load(f"{args.resume_from}/forward_process.json")
    print(f"Resuming from {args.resume_from} ({model_name}, schedule={fwd.schedule})")
else:
    schedule = args.schedule or ("log_linear" if model_name == "sedd" else "cosine")
    model = build_diffusion_model(
        model_name,
        **diffusion_model_kwargs(
            model_name,
            vocab_size=tok.vocab_size + 1,
            mask_token_id=mask_id,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=args.num_kv_heads,
            max_seq_len=seq_len,
            attention=attention,
            ffn=ffn,
            num_experts=num_experts,
            top_k_experts=top_k_experts,
            block_size=args.block_size,
            block_diffusion_unconditional=args.block_diffusion_unconditional,
            antithetic_time_sampling=args.antithetic_time_sampling,
        ),
    )
    fwd = ForwardProcess(mask_id, schedule=schedule)
if model.requires_terminal_mask_prior and not fwd.has_terminal_mask_prior():
    raise ValueError(
        f"{model_name} sampling starts from all [MASK] tokens; choose a schedule with alpha[-1] = 0"
    )
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()
print(f"{model_name.upper()}: {model.num_parameters():,} params")

tc = TrainConfig(
    max_steps=args.max_steps, warmup_steps=args.warmup_steps, batch_size=args.batch_size, lr=args.lr,
    log_every=100, eval_every=500, save_every=resolve_save_every(args.save_every, args.max_steps),
    save_dir=args.save_dir,
    resume_from=args.resume_from, seed=args.seed,
)
sig = run_signature(tok, {"name": args.dataset, "split": "train", "max_examples": max_examples, "mode": "diffusion"}, seq_len)
trainer = DiffusionTrainer(model, fwd, train_ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok), eval_dataset=eval_ds)
trainer.train()
model = trainer.model

print("\n--- Samples ---")
model.eval()
if model.supports_unconditional_diffusion_sampling():
    sample_steps = min(256, fwd.num_timesteps)
    samples = diffusion_sampler(model_name)(model, fwd, batch_size=4, seq_len=seq_len, num_steps=sample_steps)
    for i in range(4):
        s = [t for t in samples[i].tolist() if t < tok.vocab_size]
        print(f"  {tok.decode(s)[:120]}")
else:
    print("  skipped: model requires clean x_0 context for reverse scoring")
