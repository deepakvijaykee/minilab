"""Import a compatible Hugging Face causal LM into Minilab's native GPT format.

This keeps training on the existing Minilab trainers:

    python scripts/import_hf.py --model smollm2-135m --save-dir checkpoints/imported/smollm2-135m
    python scripts/sft.py --tokenizer checkpoints/imported/smollm2-135m/tokenizer.json \
      --checkpoint checkpoints/imported/smollm2-135m
"""

import argparse
import json
from pathlib import Path

import torch

from hf_common import model_dtype_kwargs, require_transformers, resolve_hf_spec
from minilab.checks import require
from minilab.hf_presets import print_hf_model_presets
from minilab.models.gpt import GPT, GPTConfig
from minilab.tokenizers.hf import HFTokenizer
from minilab.trainer import tokenizer_signature


def _rope_base(hf_config):
    data = hf_config.to_dict()
    rope = data.get("rope_parameters") or data.get("rope_scaling") or {}
    return float(rope.get("rope_theta") or data.get("rope_theta") or 10000.0)


def _native_config(hf_config, max_seq_len):
    data = hf_config.to_dict()
    model_type = data.get("model_type")
    require(model_type == "llama", (
        f"Only Llama-compatible HF models are currently importable, got model_type={model_type!r}. "
        "SmolLM2 is supported; Qwen3/Gemma need separate mapping validation."
    ))
    require(data.get("hidden_act", "silu") == "silu", "HF import currently requires hidden_act='silu'")
    require(not data.get("attention_bias", False), "HF import currently requires attention_bias=false")
    require(not data.get("mlp_bias", False), "HF import currently requires mlp_bias=false")
    require(data.get("tie_word_embeddings", True), "HF import currently requires tied input/output embeddings")

    dim = int(data["hidden_size"])
    heads = int(data["num_attention_heads"])
    kv_heads = int(data.get("num_key_value_heads") or heads)
    intermediate = int(data["intermediate_size"])
    require(dim % heads == 0, "HF hidden_size must be divisible by num_attention_heads")
    require(intermediate > 0, "HF intermediate_size must be positive")

    return GPTConfig(
        vocab_size=int(data["vocab_size"]),
        dim=dim,
        num_layers=int(data["num_hidden_layers"]),
        num_heads=heads,
        num_kv_heads=kv_heads,
        max_seq_len=max_seq_len,
        dropout=0.0,
        ffn_mult=intermediate / dim,
        norm_eps=float(data.get("rms_norm_eps", 1e-6)),
        attention="gqa" if kv_heads != heads else "mha",
        position="rope",
        norm="rmsnorm",
        ffn="swiglu",
        rope_base=_rope_base(hf_config),
    )


def _map_llama_state(hf_state, native_model):
    native_state = native_model.state_dict()
    mapped = {
        "tok_emb.weight": hf_state["model.embed_tokens.weight"],
        "lm_head.weight": hf_state.get("lm_head.weight", hf_state["model.embed_tokens.weight"]),
        "ln_f.weight": hf_state["model.norm.weight"],
    }

    num_layers = native_model.config.num_layers
    for i in range(num_layers):
        src = f"model.layers.{i}"
        dst = f"blocks.{i}"
        mapped.update({
            f"{dst}.attn_norm.weight": hf_state[f"{src}.input_layernorm.weight"],
            f"{dst}.ffn_norm.weight": hf_state[f"{src}.post_attention_layernorm.weight"],
            f"{dst}.attn.q_proj.weight": hf_state[f"{src}.self_attn.q_proj.weight"],
            f"{dst}.attn.k_proj.weight": hf_state[f"{src}.self_attn.k_proj.weight"],
            f"{dst}.attn.v_proj.weight": hf_state[f"{src}.self_attn.v_proj.weight"],
            f"{dst}.attn.out.weight": hf_state[f"{src}.self_attn.o_proj.weight"],
            f"{dst}.ffn.w1.weight": hf_state[f"{src}.mlp.gate_proj.weight"],
            f"{dst}.ffn.w2.weight": hf_state[f"{src}.mlp.up_proj.weight"],
            f"{dst}.ffn.w3.weight": hf_state[f"{src}.mlp.down_proj.weight"],
        })

    missing = sorted(set(native_state) - set(mapped))
    missing = [key for key in missing if not key.endswith("pos_enc.inv_freq")]
    require(not missing, f"HF import mapping is missing native weights: {missing[:8]}")

    for key, tensor in mapped.items():
        require(key in native_state, f"Mapped unexpected native key: {key}")
        require(tuple(tensor.shape) == tuple(native_state[key].shape), (
            f"Shape mismatch for {key}: HF {tuple(tensor.shape)} vs native {tuple(native_state[key].shape)}"
        ))
        native_state[key] = tensor.detach().cpu().to(native_state[key].dtype)

    native_model.load_state_dict(native_state)


@torch.no_grad()
def _verify_logits(hf_model, native_model, tokenizer, device, text):
    ids = torch.tensor([tokenizer.encode(text)], dtype=torch.long, device=device)
    hf_model.to(device).eval()
    native_model.to(device).eval()
    hf_logits = hf_model(input_ids=ids).logits.float().cpu()
    native_logits, _ = native_model(ids)
    diff = (hf_logits - native_logits.float().cpu()).abs()
    return {
        "prompt": text,
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
    }


p = argparse.ArgumentParser()
p.add_argument("--model", default="smollm2-135m", help="curated alias or HF repo id")
p.add_argument("--save-dir", default="", help="output Minilab checkpoint directory")
p.add_argument("--max-seq-len", type=int, default=None, help="native context length; defaults to curated recipe length")
p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
p.add_argument("--verify", action="store_true", help="compare HF and native logits on a short prompt after import")
p.add_argument("--verify-text", default="The capital of France is")
p.add_argument("--trust-remote-code", action="store_true")
p.add_argument("--list-presets", action="store_true")
args = p.parse_args()

if args.list_presets:
    print_hf_model_presets()
    raise SystemExit(0)

if args.device == "cuda" and not torch.cuda.is_available():
    raise ValueError("CUDA requested but torch.cuda.is_available() is false")

AutoConfig, AutoModelForCausalLM, AutoTokenizer = require_transformers()
spec, _ = resolve_hf_spec(args.model, "float32")
save_dir = Path(args.save_dir or f"checkpoints/imported/{spec['alias'] or spec['repo'].replace('/', '__')}")
max_seq_len = args.max_seq_len or int(spec["max_seq_len"])
require(max_seq_len > 0, "--max-seq-len must be > 0")

hf_config = AutoConfig.from_pretrained(spec["repo"], trust_remote_code=args.trust_remote_code)
native_config = _native_config(hf_config, max_seq_len)
hf_model = AutoModelForCausalLM.from_pretrained(
    spec["repo"],
    trust_remote_code=args.trust_remote_code,
    **model_dtype_kwargs(torch.float32),
)
native_model = GPT(native_config)
_map_llama_state(hf_model.state_dict(), native_model)

save_dir.mkdir(parents=True, exist_ok=True)
tokenizer_dir = save_dir / "hf_tokenizer"
hf_tokenizer = AutoTokenizer.from_pretrained(spec["repo"], trust_remote_code=args.trust_remote_code)
hf_tokenizer.save_pretrained(tokenizer_dir)
minilab_tokenizer = HFTokenizer.from_pretrained(tokenizer_dir)
minilab_tokenizer.path = tokenizer_dir.name
minilab_tokenizer._set_state_base_dir(save_dir)

native_model.save(save_dir)
minilab_tokenizer.save(save_dir / "tokenizer.json")

run_meta = {
    "tokenizer_signature": tokenizer_signature(minilab_tokenizer),
    "source": {
        "kind": "huggingface",
        "repo": spec["repo"],
        "alias": spec["alias"],
        "model_type": hf_config.to_dict().get("model_type"),
    },
}
(save_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

import_meta = {
    "source_repo": spec["repo"],
    "source_alias": spec["alias"],
    "native_model": "gpt",
    "native_config": native_config.to_dict(),
    "tokenizer": "tokenizer.json",
}
if args.verify:
    import_meta["verification"] = _verify_logits(
        hf_model,
        native_model,
        minilab_tokenizer,
        args.device,
        args.verify_text,
    )
(save_dir / "import_meta.json").write_text(json.dumps(import_meta, indent=2) + "\n")

print(f"Imported {spec['repo']} -> {save_dir}")
print(f"Tokenizer: {save_dir / 'tokenizer.json'}")
print(f"Checkpoint: {save_dir}")
if args.verify:
    v = import_meta["verification"]
    print(f"Logit check: max_abs_diff={v['max_abs_diff']:.6g} mean_abs_diff={v['mean_abs_diff']:.6g}")
