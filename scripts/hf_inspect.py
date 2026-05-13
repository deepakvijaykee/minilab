"""Inspect a curated sub-1B Hugging Face causal LM.

Examples:
    python scripts/hf_inspect.py --model smollm2-135m
    python scripts/hf_inspect.py --list-presets
"""

import argparse

import torch

from hf_common import (
    format_gb,
    model_dtype_kwargs,
    model_memory_bytes,
    model_parameter_count,
    require_transformers,
    resolve_dtype,
    resolve_hf_spec,
)
from minilab.hf_presets import hf_model_preset_choices, print_hf_model_presets


p = argparse.ArgumentParser()
p.add_argument("--model", default="smollm2-135m", help="curated alias or Hugging Face repo id")
p.add_argument("--dtype", default=None, choices=["auto", "float32", "float16", "bfloat16"])
p.add_argument("--trust-remote-code", action="store_true")
p.add_argument("--load", action="store_true", help="also instantiate weights to measure local parameter memory")
p.add_argument("--list-presets", action="store_true")
args = p.parse_args()

if args.list_presets:
    print_hf_model_presets()
    raise SystemExit(0)

AutoConfig, AutoModelForCausalLM, AutoTokenizer = require_transformers()
spec, dtype_name = resolve_hf_spec(args.model, args.dtype)
torch_dtype = resolve_dtype(dtype_name)

print(f"Alias: {spec['alias'] or '(custom)'}")
print(f"Repo: {spec['repo']}")
print(f"Family: {spec['family']}")
print(f"Preset params: {spec['params']}")
print(f"Default dtype: {spec['default_dtype']}")
print(f"Recipe max seq len: {spec['max_seq_len']}")
print(f"Role: {spec['role']}")

cfg = AutoConfig.from_pretrained(spec["repo"], trust_remote_code=args.trust_remote_code)
print(f"Architecture: {getattr(cfg, 'architectures', None)}")
print(f"Model type: {getattr(cfg, 'model_type', None)}")
print(f"HF max positions: {getattr(cfg, 'max_position_embeddings', None)}")
print(f"Vocab size: {getattr(cfg, 'vocab_size', None)}")

tok = AutoTokenizer.from_pretrained(spec["repo"], trust_remote_code=args.trust_remote_code)
print(f"Tokenizer class: {type(tok).__name__}")
print(f"Tokenizer vocab: {len(tok)}")
print(f"EOS token: {tok.eos_token!r}")
print(f"PAD token: {tok.pad_token!r}")

if args.load:
    model = AutoModelForCausalLM.from_pretrained(
        spec["repo"],
        **model_dtype_kwargs(torch_dtype),
        trust_remote_code=args.trust_remote_code,
    )
    params = model_parameter_count(model)
    memory = model_memory_bytes(model)
    print(f"Loaded params: {params:,}")
    print(f"Parameter+buffer memory: {format_gb(memory)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
else:
    print("Weights not loaded. Pass --load to instantiate the model.")

print(f"Curated presets: {hf_model_preset_choices()}")
