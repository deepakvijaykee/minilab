from pathlib import Path

from minilab.checks import require
from minilab.data import load_openwebtext, load_text8, load_tinystories, load_wikitext
from minilab.evaluation import perplexity
from minilab import generation as _generation
from minilab import models as _models
from minilab.nn.architecture import GQA_ATTENTIONS, resolve_deepseek_v4_attention
from minilab.registry import get_model, get_sampler
from minilab.trainer import LMTrainer, set_seed, tokenizer_signature
from torch.utils.data import DataLoader


# These imports execute model and sampler decorators before registry lookups below.
_REGISTRY_IMPORTS = (_models, _generation)

MODEL_CHOICES = ("gpt", "hymba", "hybrid", "mamba", "mamba2", "xlstm", "byte_latent")
DIFFUSION_MODEL_CHOICES = ("mdlm", "sedd", "d3pm", "block_diffusion")
_DIFFUSION_SAMPLER_NAMES = {
    "mdlm": "ancestral",
    "sedd": "sedd_analytical",
    "d3pm": "d3pm_ancestral",
    "block_diffusion": "ancestral",
}
PRETRAIN_DATASET_CHOICES = ("tinystories", "text8", "wikitext", "openwebtext")
PRETRAIN_EVAL_DATASET_CHOICES = ("tinystories", "text8", "wikitext")


def _require_choice(name, choices, kind):
    require(name in choices, f"Unknown {kind}: {name}. Available: {choices}")


def model_class(name):
    _require_choice(name, MODEL_CHOICES, "model")
    return get_model(name)


def build_lm_model(name, **config_kwargs):
    cls = model_class(name)
    return cls(cls.config_class(**config_kwargs))


def _base_lm_kwargs(vocab_size, dim, num_layers, max_seq_len):
    return {
        "vocab_size": vocab_size,
        "dim": dim,
        "num_layers": num_layers,
        "max_seq_len": max_seq_len,
    }


def _require_num_heads(model_name, num_heads):
    require(num_heads is not None, f"{model_name} requires num_heads")
    return num_heads


def lm_model_kwargs(
    model_name,
    *,
    vocab_size,
    dim,
    num_layers,
    max_seq_len,
    num_heads=None,
    num_kv_heads=None,
    attention=None,
    position=None,
    norm=None,
    rope_base=None,
    rope_local_base=None,
    rope_global_base=None,
    rope_scaling_factor=None,
    rope_original_max_seq_len=None,
    rope_partial_rotary_factor=None,
    yarn_beta_fast=None,
    yarn_beta_slow=None,
    local_attention_window=None,
    qwen3_next_full_attention_interval=None,
    attention_k_eq_v=None,
    per_layer_embedding_dim=None,
    final_logit_softcap=None,
    connection=None,
    ffn=None,
    num_experts=None,
    top_k_experts=None,
    post_norm=None,
    mtp_depth=None,
    mtp_loss_weight=None,
):
    kwargs = _base_lm_kwargs(vocab_size, dim, num_layers, max_seq_len)
    if model_name == "gpt":
        kwargs["num_heads"] = _require_num_heads(model_name, num_heads)
        _update_supplied(kwargs, {
            "num_kv_heads": num_kv_heads,
            "attention": attention,
            "position": position,
            "norm": norm,
            "connection": connection,
            "ffn": ffn,
            "num_experts": num_experts,
            "top_k_experts": top_k_experts,
            "post_norm": post_norm,
            "rope_base": rope_base,
            "rope_local_base": rope_local_base,
            "rope_global_base": rope_global_base,
            "rope_scaling_factor": rope_scaling_factor,
            "rope_original_max_seq_len": rope_original_max_seq_len,
            "rope_partial_rotary_factor": rope_partial_rotary_factor,
            "yarn_beta_fast": yarn_beta_fast,
            "yarn_beta_slow": yarn_beta_slow,
            "local_attention_window": local_attention_window,
            "qwen3_next_full_attention_interval": qwen3_next_full_attention_interval,
            "attention_k_eq_v": attention_k_eq_v,
            "per_layer_embedding_dim": per_layer_embedding_dim,
            "final_logit_softcap": final_logit_softcap,
            "mtp_depth": mtp_depth,
            "mtp_loss_weight": mtp_loss_weight,
        })
    elif model_name in {"hybrid", "hymba"}:
        kwargs["num_heads"] = _require_num_heads(model_name, num_heads)
        _update_supplied(kwargs, {
            "num_kv_heads": num_kv_heads,
            "attention": attention,
            "position": position,
            "norm": norm,
            "ffn": ffn,
            "num_experts": num_experts,
            "top_k_experts": top_k_experts,
            "post_norm": post_norm,
            "rope_base": rope_base,
            "rope_scaling_factor": rope_scaling_factor,
            "rope_original_max_seq_len": rope_original_max_seq_len,
            "rope_partial_rotary_factor": rope_partial_rotary_factor,
            "yarn_beta_fast": yarn_beta_fast,
            "yarn_beta_slow": yarn_beta_slow,
            "local_attention_window": local_attention_window,
            "qwen3_next_full_attention_interval": qwen3_next_full_attention_interval,
            "final_logit_softcap": final_logit_softcap,
        })
    elif model_name in {"mamba", "mamba2"}:
        return kwargs
    elif model_name == "byte_latent":
        kwargs["num_heads"] = _require_num_heads(model_name, num_heads)
        _update_supplied(kwargs, {
            "attention": attention,
            "norm": norm,
            "ffn": ffn,
        })
    elif model_name == "xlstm":
        kwargs["num_heads"] = _require_num_heads(model_name, num_heads)
    else:
        raise ValueError(f"Unhandled model family: {model_name}")
    return kwargs


def _update_supplied(target, fields):
    target.update({name: value for name, value in fields.items() if value is not None})


def diffusion_model_class(name):
    _require_choice(name, DIFFUSION_MODEL_CHOICES, "diffusion model")
    return get_model(name)


def build_diffusion_model(name, **config_kwargs):
    cls = diffusion_model_class(name)
    return cls(cls.config_class(**config_kwargs))


def diffusion_model_kwargs(
    model_name,
    *,
    vocab_size,
    mask_token_id,
    dim,
    num_layers,
    num_heads,
    num_kv_heads,
    max_seq_len,
    attention,
    ffn,
    num_experts,
    top_k_experts,
    block_size=None,
    block_diffusion_unconditional=None,
    antithetic_time_sampling=None,
):
    kwargs = {
        "vocab_size": vocab_size,
        "dim": dim,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "max_seq_len": max_seq_len,
        "attention": attention,
        "ffn": ffn,
        "num_experts": num_experts,
        "top_k_experts": top_k_experts,
        "mask_token_id": mask_token_id,
    }
    if model_name == "block_diffusion":
        kwargs.update(
            block_size=resolve_default(block_size, 32),
            cross_attention=not resolve_default(block_diffusion_unconditional, False),
            antithetic_time_sampling=resolve_default(antithetic_time_sampling, False),
        )
    return kwargs


def diffusion_sampler(name):
    _require_choice(name, DIFFUSION_MODEL_CHOICES, "diffusion model")
    return get_sampler(_DIFFUSION_SAMPLER_NAMES[name])


def flag_name(name):
    return "--" + name.replace("_", "-")


def reject_supplied(args, names, reason):
    supplied = vars(args)
    for name in names:
        require(supplied[name] is None, f"{flag_name(name)} {reason}")


def resolve_default(value, default):
    return default if value is None else value


def load_pretrain_dataset(name, tok, seq_len, split, max_examples, mode):
    if name == "tinystories":
        return load_tinystories(tok, seq_len, split=split, max_examples=max_examples, mode=mode)
    if name == "text8":
        require(max_examples == 0, "--max-examples does not apply to text8; pass 0 to use the standard split")
        return load_text8(tok, seq_len, split=split, mode=mode)
    if name == "wikitext":
        return load_wikitext(tok, seq_len, split=split, max_examples=max_examples, mode=mode)
    if name == "openwebtext":
        require(split == "train", "OpenWebText loader only supports the train split")
        return load_openwebtext(tok, seq_len, max_examples=max_examples, mode=mode)
    raise ValueError(f"Unknown pretraining dataset: {name}")


def resolve_pretrain_max_examples(name, supplied, default):
    if name == "text8":
        require(supplied in {None, 0}, "--max-examples does not apply to text8; use the standard split")
        return 0
    return default if supplied is None else supplied


def load_pretrain_eval_dataset(name, tok, seq_len, max_examples, mode):
    require(name != "openwebtext", "evaluation requires a validation split; OpenWebText loader only supports train")
    eval_max_examples = 0 if name == "text8" else max_examples
    return load_pretrain_dataset(name, tok, seq_len, "validation", eval_max_examples, mode)


def attention_uses_gqa(attention):
    return attention in {"gemma3", "gemma4", "qwen3_next"} or (
        resolve_deepseek_v4_attention(attention, 0) in GQA_ATTENTIONS
    )


def require_checkpoint_path(checkpoint, resume_from, context):
    require(not (checkpoint and resume_from), f"{context} accepts --checkpoint or --resume-from, not both")
    path = resume_from or checkpoint
    require(path, f"{context} requires --checkpoint or --resume-from")
    return path


def _checkpoint_name_by_type(choices):
    return {get_model(name).__name__: name for name in choices}


def _checkpoint_model_name(path, choices, kind):
    type_path = Path(path) / "model_type.txt"
    require(type_path.exists(), f"Missing {type_path}; checkpoint must declare its model family")
    saved_type = type_path.read_text().strip()
    name_by_type = _checkpoint_name_by_type(choices)
    require(
        saved_type in name_by_type,
        f"Checkpoint declares unsupported {kind} family {saved_type!r}; expected one of {tuple(name_by_type)}",
    )
    return name_by_type[saved_type]


def load_model_checkpoint(path, requested_model=None, device="cpu"):
    model_name = requested_model or _checkpoint_model_name(path, MODEL_CHOICES, "language model")
    return model_name, model_class(model_name).load(path, device=device)


def load_diffusion_model_checkpoint(path, requested_model=None, device="cpu"):
    model_name = requested_model or _checkpoint_model_name(
        path,
        DIFFUSION_MODEL_CHOICES,
        "diffusion model",
    )
    return model_name, diffusion_model_class(model_name).load(path, device=device)


def compare_lm_variants(
    variants,
    tok,
    train_ds,
    eval_ds,
    train_config,
    signature,
    *,
    seed,
    dim,
    num_layers,
    num_heads,
    seq_len,
):
    results = []
    gpt_cls = model_class("gpt")
    for name, config_fields in variants:
        print(f"\n=== {name} ===")
        set_seed(seed)
        cfg = gpt_cls.config_class(
            vocab_size=tok.vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=seq_len,
            **config_fields,
        )
        model = gpt_cls(cfg)
        print(f"  {model.num_parameters():,} params")
        trainer = LMTrainer(
            model,
            train_ds,
            train_config,
            signature=signature,
            tokenizer_sig=tokenizer_signature(tok),
            eval_dataset=eval_ds,
        )
        trainer.train()
        eval_loss = trainer.evaluate()
        model.eval()
        ppl = perplexity(model, DataLoader(eval_ds, batch_size=32))
        results.append((name, model.num_parameters(), eval_loss, ppl))

    print(f"\n{'Variant':<15} {'Params':>10} {'Loss':>10} {'PPL':>10}")
    print("-" * 48)
    for name, params, loss, ppl in results:
        print(f"{name:<15} {params:>10,} {loss:>10.4f} {ppl:>10.1f}")
