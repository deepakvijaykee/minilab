"""Response-only supervised fine-tuning for diffusion LMs.

    python scripts/sft_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
"""

import argparse

import torch

from minilab.checks import require
from minilab.data import load_alpaca_diffusion, load_dolly_diffusion
from minilab.diffusion import ForwardProcess
from minilab.generation import infill
from minilab.models.transformer_utils import DEFAULT_NUM_EXPERTS, DEFAULT_TOP_K_EXPERTS
from minilab.nn.architecture import MOE_FFNS
from minilab.tokenizers import load_tokenizer
from minilab.trainer import (
    DiffusionSFTTrainer,
    TrainConfig,
    run_signature,
    set_seed,
    tokenizer_signature,
    validate_checkpoint_tokenizer,
)
from common import (
    DIFFUSION_MODEL_CHOICES,
    attention_uses_gqa,
    build_diffusion_model,
    diffusion_model_kwargs,
    load_diffusion_model_checkpoint,
    reject_supplied,
    resolve_default,
    resolve_save_every,
)


DATASETS = {"alpaca": load_alpaca_diffusion, "dolly": load_dolly_diffusion}
_MODEL_BUILD_FLAGS = (
    "dim", "num_layers", "num_heads", "num_kv_heads",
    "attention", "ffn", "num_experts", "top_k_experts",
    "block_size", "block_diffusion_unconditional", "antithetic_time_sampling",
)
_BLOCK_DIFFUSION_FLAGS = (
    "block_size", "block_diffusion_unconditional", "antithetic_time_sampling",
)


p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", default="")
p.add_argument("--save-dir", default="checkpoints/diffusion_sft")
p.add_argument("--model", default=None, choices=DIFFUSION_MODEL_CHOICES, help="model family for new runs; inferred from checkpoints")
p.add_argument("--dataset", default="alpaca", choices=list(DATASETS))
p.add_argument("--dim", type=int, default=None)
p.add_argument("--num-layers", type=int, default=None)
p.add_argument("--num-heads", type=int, default=None)
p.add_argument("--num-kv-heads", type=int, default=None, help="KV heads for GQA; defaults to num_heads")
p.add_argument("--seq-len", type=int, default=256)
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
p.add_argument("--max-steps", type=int, default=3000)
p.add_argument("--warmup-steps", type=int, default=100)
p.add_argument("--save-every", type=int, default=0, help="periodic save interval (0 = save once at end)")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--lr", type=float, default=1e-4)
p.add_argument("--max-examples", type=int, default=10000)
p.add_argument("--sample-new-tokens", type=int, default=80)
p.add_argument("--grad-checkpoint", action="store_true")
p.add_argument("--resume-from", default="")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
model_name = args.model or "mdlm"
require(not (args.checkpoint and args.resume_from), (
    "Diffusion SFT accepts --checkpoint or --resume-from, not both"
))

loading_model = bool(args.resume_from or args.checkpoint)
if loading_model:
    reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
    require(args.schedule is None, "--schedule only applies when starting a new model")

set_seed(args.seed)

dim = resolve_default(args.dim, 256)
num_layers = resolve_default(args.num_layers, 6)
num_heads = resolve_default(args.num_heads, 8)
attention = resolve_default(args.attention, "mha")
ffn = resolve_default(args.ffn, "swiglu")
num_experts = resolve_default(args.num_experts, DEFAULT_NUM_EXPERTS)
top_k_experts = resolve_default(args.top_k_experts, DEFAULT_TOP_K_EXPERTS)

if not loading_model:
    if args.num_kv_heads is not None:
        require(attention_uses_gqa(attention), "--num-kv-heads only applies to GQA attention variants")
    if args.num_experts is not None or args.top_k_experts is not None:
        require(ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")
    if model_name != "block_diffusion":
        reject_supplied(args, _BLOCK_DIFFUSION_FLAGS, "only applies to --model block_diffusion")

tok = load_tokenizer(args.tokenizer)
mask_id = tok.vocab_size

if args.resume_from:
    validate_checkpoint_tokenizer(args.resume_from, tok)
    model_name, model = load_diffusion_model_checkpoint(args.resume_from, args.model)
    fwd = ForwardProcess.load(f"{args.resume_from}/forward_process.json")
    print(f"Resuming from {args.resume_from} ({model_name}, schedule={fwd.schedule})")
elif args.checkpoint:
    validate_checkpoint_tokenizer(args.checkpoint, tok)
    model_name, model = load_diffusion_model_checkpoint(args.checkpoint, args.model)
    fwd = ForwardProcess.load(f"{args.checkpoint}/forward_process.json")
    print(f"Loaded {args.checkpoint} ({model_name}, {model.num_parameters():,} params)")
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
            max_seq_len=args.seq_len,
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
    print(f"New {model_name.upper()} ({model.num_parameters():,} params, schedule={fwd.schedule})")

if model.requires_terminal_mask_prior and not fwd.has_terminal_mask_prior():
    raise ValueError(
        f"{model_name} infill starts masked response positions from [MASK]; choose a schedule with alpha[-1] = 0"
    )
if args.grad_checkpoint:
    model.gradient_checkpointing_enable()

ds = DATASETS[args.dataset](tok, args.seq_len, max_examples=args.max_examples)
print(f"{args.dataset}: {len(ds)} diffusion SFT examples")

tc = TrainConfig(
    max_steps=args.max_steps,
    warmup_steps=args.warmup_steps,
    batch_size=args.batch_size,
    lr=args.lr,
    log_every=100,
    eval_every=0,
    save_every=resolve_save_every(args.save_every, args.max_steps),
    save_dir=args.save_dir,
    resume_from=args.resume_from,
    seed=args.seed,
)
sig = run_signature(
    tok,
    {"name": args.dataset, "split": "train", "max_examples": args.max_examples, "mode": "diffusion_sft"},
    args.seq_len,
)
trainer = DiffusionSFTTrainer(model, fwd, ds, tc, signature=sig, tokenizer_sig=tokenizer_signature(tok))
trainer.train()
model = trainer.model

print("\n--- After Diffusion SFT ---")
model.eval()
sample_steps = min(128, fwd.num_timesteps)
if not model.supports_unconditional_diffusion_sampling():
    print("  skipped: model requires clean x_0 context for reverse scoring")
else:
    for q in ["Give three tips for staying healthy.", "What is the capital of France?", "Explain gravity."]:
        prompt = tok.encode(q)[: max(1, args.seq_len - args.sample_new_tokens)]
        gen_len = min(args.sample_new_tokens, args.seq_len - len(prompt))
        tokens = prompt + [mask_id] * gen_len
        mask = [False] * len(prompt) + [True] * gen_len
        filled = infill(
            model,
            fwd,
            torch.tensor([tokens], dtype=torch.long),
            torch.tensor([mask], dtype=torch.bool),
            num_steps=sample_steps,
            temperature=0.7,
        )[0]
        answer = [t for t in filled[len(prompt) : len(prompt) + gen_len].tolist() if t < tok.vocab_size]
        print(f"  Q: {q}")
        print(f"  A: {tok.decode(answer)[:120]}\n")
