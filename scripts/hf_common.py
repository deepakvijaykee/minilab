from importlib.metadata import PackageNotFoundError, version

import torch

from minilab.hf_cache import configure_hf_cache
from minilab.hf_presets import resolve_hf_model


_DTYPES = {
    "auto": "auto",
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def require_transformers():
    configure_hf_cache()
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Hugging Face support requires optional dependencies. "
            "Install with: python -m pip install -e \".[hf]\""
        ) from exc
    return AutoConfig, AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(name):
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype: {name}. Available: {tuple(_DTYPES)}")
    return _DTYPES[name]


def resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    return device


def resolve_hf_spec(model, dtype):
    spec = resolve_hf_model(model)
    if dtype is None:
        dtype = spec["default_dtype"]
    return spec, dtype


def model_dtype_kwargs(torch_dtype):
    try:
        major = int(version("transformers").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        major = 4
    if major >= 5:
        return {"dtype": torch_dtype}
    return {"torch_dtype": torch_dtype}


def model_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def model_memory_bytes(model):
    total = 0
    for param in model.parameters():
        total += param.numel() * param.element_size()
    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()
    return total


def format_gb(num_bytes):
    return f"{num_bytes / 1024 ** 3:.2f} GB"
