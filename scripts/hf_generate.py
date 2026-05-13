"""Generate text from a curated sub-1B Hugging Face causal LM.

Examples:
    python scripts/hf_generate.py --model smollm2-135m-instruct --prompt "Explain gravity"
    python scripts/hf_generate.py --model qwen3-0.6b --device cuda --dtype bfloat16
"""

import argparse

import torch

from hf_common import model_dtype_kwargs, require_transformers, resolve_device, resolve_dtype, resolve_hf_spec
from minilab.hf_presets import print_hf_model_presets


p = argparse.ArgumentParser()
p.add_argument("--model", default="smollm2-135m-instruct", help="curated alias or Hugging Face repo id")
p.add_argument("--prompt", default="Explain gravity in one paragraph.")
p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
p.add_argument("--dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
p.add_argument("--max-new-tokens", type=int, default=128)
p.add_argument("--temperature", type=float, default=0.7)
p.add_argument("--top-p", type=float, default=0.95)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--trust-remote-code", action="store_true")
p.add_argument("--list-presets", action="store_true")
args = p.parse_args()

if args.list_presets:
    print_hf_model_presets()
    raise SystemExit(0)

_, AutoModelForCausalLM, AutoTokenizer = require_transformers()
spec, dtype_name = resolve_hf_spec(args.model, args.dtype)
device = resolve_device(args.device)
if args.dtype is None and device == "cpu":
    dtype_name = "float32"
torch_dtype = resolve_dtype(dtype_name)
torch.manual_seed(args.seed)

tok = AutoTokenizer.from_pretrained(spec["repo"], trust_remote_code=args.trust_remote_code)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    spec["repo"],
    **model_dtype_kwargs(torch_dtype),
    trust_remote_code=args.trust_remote_code,
)
model.to(device)
model.eval()

inputs = tok(args.prompt, return_tensors="pt").to(device)
do_sample = args.temperature > 0

if device == "cuda":
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=do_sample,
        temperature=args.temperature if do_sample else None,
        top_p=args.top_p if do_sample else None,
        top_k=args.top_k if do_sample else None,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )

text = tok.decode(out[0], skip_special_tokens=True)
print(text)

if device == "cuda":
    torch.cuda.synchronize()
    print(
        f"\n[metrics] peak_allocated={torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GB "
        f"peak_reserved={torch.cuda.max_memory_reserved() / 1024 ** 3:.2f} GB"
    )
