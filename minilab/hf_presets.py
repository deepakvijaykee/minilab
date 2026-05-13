from copy import deepcopy

from minilab.checks import require


HF_MODEL_PRESETS = {
    "smollm2-135m": {
        "repo": "HuggingFaceTB/SmolLM2-135M",
        "params": "135M",
        "family": "SmolLM2",
        "default_dtype": "float16",
        "max_seq_len": 512,
        "role": "smallest HF pretrained baseline",
    },
    "smollm2-135m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "params": "135M",
        "family": "SmolLM2",
        "default_dtype": "float16",
        "max_seq_len": 512,
        "role": "tiny instruction-tuned HF baseline",
    },
    "gemma3-270m": {
        "repo": "google/gemma-3-270m",
        "params": "270M",
        "family": "Gemma 3",
        "default_dtype": "bfloat16",
        "max_seq_len": 512,
        "role": "small modern pretrained baseline",
    },
    "gemma3-270m-it": {
        "repo": "google/gemma-3-270m-it",
        "params": "270M",
        "family": "Gemma 3",
        "default_dtype": "bfloat16",
        "max_seq_len": 512,
        "role": "small modern instruction-tuned baseline",
    },
    "smollm2-360m": {
        "repo": "HuggingFaceTB/SmolLM2-360M",
        "params": "360M",
        "family": "SmolLM2",
        "default_dtype": "float16",
        "max_seq_len": 512,
        "role": "lightweight pretrained HF baseline",
    },
    "smollm2-360m-instruct": {
        "repo": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "params": "360M",
        "family": "SmolLM2",
        "default_dtype": "float16",
        "max_seq_len": 512,
        "role": "lightweight instruction-tuned HF baseline",
    },
    "qwen3-0.6b": {
        "repo": "Qwen/Qwen3-0.6B",
        "params": "0.6B",
        "family": "Qwen3",
        "default_dtype": "bfloat16",
        "max_seq_len": 512,
        "role": "main modern sub-1B HF baseline",
    },
    "qwen3-0.6b-base": {
        "repo": "Qwen/Qwen3-0.6B-Base",
        "params": "0.6B",
        "family": "Qwen3",
        "default_dtype": "bfloat16",
        "max_seq_len": 512,
        "role": "main modern sub-1B pretrained baseline",
    },
}


def hf_model_preset_choices():
    return tuple(HF_MODEL_PRESETS)


def get_hf_model_preset(name):
    require(name in HF_MODEL_PRESETS, f"Unknown HF preset: {name}. Available: {hf_model_preset_choices()}")
    return deepcopy(HF_MODEL_PRESETS[name])


def resolve_hf_model(name_or_repo):
    if name_or_repo in HF_MODEL_PRESETS:
        preset = get_hf_model_preset(name_or_repo)
        preset["alias"] = name_or_repo
        return preset
    return {
        "alias": "",
        "repo": name_or_repo,
        "params": "unknown",
        "family": "custom",
        "default_dtype": "float16",
        "max_seq_len": 512,
        "role": "custom Hugging Face model",
    }


def print_hf_model_presets():
    for alias, preset in HF_MODEL_PRESETS.items():
        print(f"{alias:24s} {preset['repo']:40s} {preset['params']:>6s}  {preset['role']}")
