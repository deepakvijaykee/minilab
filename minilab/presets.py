from copy import deepcopy

from minilab.checks import require


LM_MODEL_PRESETS = {
    "gpt-10m": {
        "model": "gpt",
        "dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "seq_len": 512,
        "use_case": "main tiny GPT for tokenizer, pretraining, SFT, and preference runs",
    },
    "gpt-25m": {
        "model": "gpt",
        "dim": 384,
        "num_layers": 8,
        "num_heads": 8,
        "seq_len": 512,
        "use_case": "larger laptop GPT for more realistic alignment experiments",
    },
    "gpt-60m": {
        "model": "gpt",
        "dim": 512,
        "num_layers": 12,
        "num_heads": 8,
        "seq_len": 1024,
        "use_case": "stretch GPT for users with more headroom",
    },
    "mamba-25m": {
        "model": "mamba",
        "dim": 512,
        "num_layers": 12,
        "seq_len": 512,
        "use_case": "SSM baseline for laptop-scale sequence model comparisons",
    },
}


DIFFUSION_MODEL_PRESETS = {
    "mdlm-25m": {
        "model": "mdlm",
        "dim": 384,
        "num_layers": 8,
        "num_heads": 8,
        "seq_len": 512,
        "use_case": "small masked diffusion LM for laptop-scale diffusion experiments",
    },
}


def lm_model_preset_choices():
    return tuple(LM_MODEL_PRESETS)


def diffusion_model_preset_choices():
    return tuple(DIFFUSION_MODEL_PRESETS)


def all_model_preset_choices():
    return lm_model_preset_choices() + diffusion_model_preset_choices()


def get_lm_model_preset(name):
    require(name in LM_MODEL_PRESETS, f"Unknown LM preset: {name}. Available: {lm_model_preset_choices()}")
    return deepcopy(LM_MODEL_PRESETS[name])


def get_diffusion_model_preset(name):
    require(
        name in DIFFUSION_MODEL_PRESETS,
        f"Unknown diffusion preset: {name}. Available: {diffusion_model_preset_choices()}",
    )
    return deepcopy(DIFFUSION_MODEL_PRESETS[name])


def get_any_model_preset(name):
    if name in LM_MODEL_PRESETS:
        preset = get_lm_model_preset(name)
        preset["kind"] = "lm"
        return preset
    if name in DIFFUSION_MODEL_PRESETS:
        preset = get_diffusion_model_preset(name)
        preset["kind"] = "diffusion"
        return preset
    raise ValueError(f"Unknown model preset: {name}. Available: {all_model_preset_choices()}")
