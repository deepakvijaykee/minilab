from pathlib import Path

from minilab.checks import require
from minilab.data import load_openwebtext, load_text8, load_tinystories, load_wikitext
from minilab.evaluation import perplexity
from minilab.generation import sample_d3pm, sample_diffusion, sample_sedd
from minilab.models.block_diffusion import BlockDiffusionLM
from minilab.models.byte_latent import ByteLatentLM
from minilab.models.hymba import HymbaLM
from minilab.models.gpt import GPT, GPTConfig
from minilab.models.hybrid import HybridLM
from minilab.models.mamba import MambaLM
from minilab.models.mamba2 import Mamba2LM
from minilab.models.xlstm import XLSTMLM
from minilab.models.d3pm import D3PM
from minilab.models.mdlm import MDLM
from minilab.models.sedd import SEDD
from minilab.nn.architecture import GQA_ATTENTIONS, resolve_deepseek_v4_attention
from minilab.trainer import LMTrainer, set_seed, tokenizer_signature
from torch.utils.data import DataLoader


_MODEL_CLASSES = {
    "gpt": GPT,
    "hymba": HymbaLM,
    "hybrid": HybridLM,
    "mamba": MambaLM,
    "mamba2": Mamba2LM,
    "xlstm": XLSTMLM,
    "byte_latent": ByteLatentLM,
}
_DIFFUSION_MODEL_CLASSES = {"mdlm": MDLM, "sedd": SEDD, "d3pm": D3PM, "block_diffusion": BlockDiffusionLM}
_DIFFUSION_SAMPLER_BY_MODEL = {
    "mdlm": sample_diffusion,
    "sedd": sample_sedd,
    "d3pm": sample_d3pm,
    "block_diffusion": sample_diffusion,
}
MODEL_CHOICES = tuple(_MODEL_CLASSES)
DIFFUSION_MODEL_CHOICES = tuple(_DIFFUSION_MODEL_CLASSES)
PRETRAIN_DATASET_CHOICES = ("tinystories", "text8", "wikitext", "openwebtext")
_MODEL_NAME_BY_CHECKPOINT_TYPE = {cls.__name__: name for name, cls in _MODEL_CLASSES.items()}
_DIFFUSION_NAME_BY_CHECKPOINT_TYPE = {
    cls.__name__: name for name, cls in _DIFFUSION_MODEL_CLASSES.items()
}


def _lookup(table, name, kind):
    require(name in table, f"Unknown {kind}: {name}. Available: {tuple(table)}")
    return table[name]


def model_class(name):
    return _lookup(_MODEL_CLASSES, name, "model")


def build_lm_model(name, **config_kwargs):
    cls = model_class(name)
    return cls(cls.config_class(**config_kwargs))


def diffusion_model_class(name):
    return _lookup(_DIFFUSION_MODEL_CLASSES, name, "diffusion model")


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
    return _lookup(_DIFFUSION_SAMPLER_BY_MODEL, name, "diffusion model")


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


def _checkpoint_model_name(path, name_by_type, kind):
    type_path = Path(path) / "model_type.txt"
    require(type_path.exists(), f"Missing {type_path}; checkpoint must declare its model family")
    saved_type = type_path.read_text().strip()
    require(
        saved_type in name_by_type,
        f"Checkpoint declares unsupported {kind} family {saved_type!r}; expected one of {tuple(name_by_type)}",
    )
    return name_by_type[saved_type]


def load_model_checkpoint(path, requested_model=None, device="cpu"):
    model_name = requested_model or _checkpoint_model_name(path, _MODEL_NAME_BY_CHECKPOINT_TYPE, "language model")
    return model_name, model_class(model_name).load(path, device=device)


def load_diffusion_model_checkpoint(path, requested_model=None, device="cpu"):
    model_name = requested_model or _checkpoint_model_name(
        path,
        _DIFFUSION_NAME_BY_CHECKPOINT_TYPE,
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
    for name, config_fields in variants:
        print(f"\n=== {name} ===")
        set_seed(seed)
        cfg = GPTConfig(
            vocab_size=tok.vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            max_seq_len=seq_len,
            **config_fields,
        )
        model = GPT(cfg)
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
