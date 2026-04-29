from pathlib import Path

from minilab.checks import require
from minilab.evaluation import perplexity
from minilab.generation import sample_d3pm, sample_diffusion, sample_sedd
from minilab.models.gpt import GPT, GPTConfig
from minilab.models.mamba import MambaLM
from minilab.models.d3pm import D3PM, D3PMConfig
from minilab.models.mdlm import MDLM, MDLMConfig
from minilab.models.sedd import SEDD, SEDDConfig
from minilab.nn.architecture import GQA_ATTENTIONS, resolve_deepseek_v4_attention
from minilab.trainer import LMTrainer, set_seed, tokenizer_signature
from torch.utils.data import DataLoader


_MODEL_CLASSES = {"gpt": GPT, "mamba": MambaLM}
_DIFFUSION_MODEL_CLASSES = {"mdlm": MDLM, "sedd": SEDD, "d3pm": D3PM}
_DIFFUSION_CONFIG_CLASSES = {"mdlm": MDLMConfig, "sedd": SEDDConfig, "d3pm": D3PMConfig}
_DIFFUSION_SAMPLER_BY_MODEL = {"mdlm": sample_diffusion, "sedd": sample_sedd, "d3pm": sample_d3pm}
MODEL_CHOICES = tuple(_MODEL_CLASSES)
DIFFUSION_MODEL_CHOICES = tuple(_DIFFUSION_MODEL_CLASSES)
_MODEL_NAME_BY_CHECKPOINT_TYPE = {cls.__name__: name for name, cls in _MODEL_CLASSES.items()}
_DIFFUSION_NAME_BY_CHECKPOINT_TYPE = {
    cls.__name__: name for name, cls in _DIFFUSION_MODEL_CLASSES.items()
}


def _lookup(table, name, kind):
    require(name in table, f"Unknown {kind}: {name}. Available: {tuple(table)}")
    return table[name]


def model_class(name):
    return _lookup(_MODEL_CLASSES, name, "model")


def diffusion_model_class(name):
    return _lookup(_DIFFUSION_MODEL_CLASSES, name, "diffusion model")


def diffusion_config_class(name):
    return _lookup(_DIFFUSION_CONFIG_CLASSES, name, "diffusion model")


def diffusion_sampler(name):
    return _lookup(_DIFFUSION_SAMPLER_BY_MODEL, name, "diffusion model")


def flag_name(name):
    return "--" + name.replace("_", "-")


def reject_supplied(args, names, reason):
    for name in names:
        require(getattr(args, name) is None, f"{flag_name(name)} {reason}")


def resolve_default(value, default):
    return default if value is None else value


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
