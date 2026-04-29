"""Response-only supervised fine-tuning for diffusion LMs.

    python scripts/sft_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
"""

import argparse

import torch

from minilab.checks import require
from minilab.data import load_alpaca_diffusion, load_dolly_diffusion
from minilab.diffusion import ForwardProcess
from minilab.generation import infill
from minilab.nn.architecture import GQA_ATTENTIONS, MOE_FFNS, resolve_deepseek_v4_attention
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
    diffusion_config_class,
    diffusion_model_class,
    load_diffusion_model_checkpoint,
)


DATASETS = {"alpaca": load_alpaca_diffusion, "dolly": load_dolly_diffusion}
_MODEL_BUILD_FLAGS = (
    "dim", "num_layers", "num_heads", "num_kv_heads",
    "attention", "ffn", "num_experts", "top_k_experts",
)


def _flag(name):
    return "--" + name.replace("_", "-")


def _reject_supplied(args, names, reason):
    for name in names:
        require(getattr(args, name) is None, f"{_flag(name)} {reason}")


def _resolve(value, default):
    return default if value is None else value


def _attention_uses_gqa(attention):
    return resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS


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
p.add_argument("--schedule", default="", choices=["", "cosine", "linear", "geometric", "log_linear"])
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

loading_model = bool(args.resume_from or args.checkpoint)
if loading_model:
    _reject_supplied(args, _MODEL_BUILD_FLAGS, "only applies when starting a new model")
    require(args.schedule == "", "--schedule only applies when starting a new model")

set_seed(args.seed)

dim = _resolve(args.dim, 256)
num_layers = _resolve(args.num_layers, 6)
num_heads = _resolve(args.num_heads, 8)
attention = _resolve(args.attention, "mha")
ffn = _resolve(args.ffn, "swiglu")
num_experts = _resolve(args.num_experts, 8)
top_k_experts = _resolve(args.top_k_experts, 2)

if not loading_model:
    if args.num_kv_heads is not None:
        require(_attention_uses_gqa(attention), "--num-kv-heads only applies to GQA attention variants")
    if args.num_experts is not None or args.top_k_experts is not None:
        require(ffn in MOE_FFNS, "--num-experts and --top-k-experts only apply to MoE FFNs")

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
    cls = diffusion_model_class(model_name)
    cfg_cls = diffusion_config_class(model_name)
    schedule = args.schedule or ("log_linear" if model_name == "sedd" else "cosine")
    config = cfg_cls(
        vocab_size=tok.vocab_size + 1,
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_kv_heads=args.num_kv_heads,
        max_seq_len=args.seq_len,
        attention=attention,
        ffn=ffn,
        num_experts=num_experts,
        top_k_experts=top_k_experts,
        mask_token_id=mask_id,
    )
    model = cls(config)
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
    save_every=args.save_every or args.max_steps,
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
