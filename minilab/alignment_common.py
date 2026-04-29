import json
from pathlib import Path

import torch
import torch.nn.functional as F

from minilab.base import BaseModel, unwrap_model
from minilab.checks import require


def _validated_reference_path(ref_model_path, algorithm):
    require(ref_model_path is not None, f"{algorithm} requires a frozen reference checkpoint")
    ref_model_path = str(ref_model_path).strip()
    require(ref_model_path, f"{algorithm} requires a frozen reference checkpoint")
    path = Path(ref_model_path).expanduser().resolve()
    require(path.exists(), f"{algorithm} frozen reference checkpoint does not exist: {path}")
    missing = [name for name in ("config.json", "model.pt") if not (path / name).exists()]
    require(not missing, f"{algorithm} frozen reference checkpoint {path} is missing: {missing}")
    return str(path)


def _resume_reference_path(resume_from, algorithm):
    path = Path(resume_from) / "ref_path.txt"
    require(path.exists(), f"{algorithm} resume is missing {path}; cannot restore frozen reference")
    ref_path = path.read_text().strip()
    require(ref_path, f"{algorithm} resume has empty frozen reference path in {path}")
    return _validated_reference_path(ref_path, algorithm)


def resolve_reference_path(checkpoint, resume_from, algorithm):
    if resume_from:
        return _resume_reference_path(resume_from, algorithm)
    return _validated_reference_path(checkpoint, algorithm)


def _trainer_reference_path(ref_model_path, config, algorithm):
    ref_path = _validated_reference_path(ref_model_path, algorithm)
    if config.resume_from:
        saved_ref_path = _resume_reference_path(config.resume_from, algorithm)
        require(ref_path == saved_ref_path, (
            f"{algorithm} resume reference mismatch: checkpoint expects {saved_ref_path}, "
            f"caller supplied {ref_path}"
        ))
    return ref_path


def _validate_reference_tokenizer(ref_model_path, tokenizer_sig, algorithm):
    require(tokenizer_sig, f"{algorithm} requires tokenizer_sig to validate the frozen reference tokenizer")
    meta_path = Path(ref_model_path) / "run_meta.json"
    require(meta_path.exists(), (
        f"{algorithm} frozen reference checkpoint is missing {meta_path}; "
        "cannot validate tokenizer identity"
    ))
    saved_meta = json.loads(meta_path.read_text())
    require(isinstance(saved_meta, dict), (
        f"{algorithm} frozen reference run metadata must be a JSON object"
    ))
    require("tokenizer_signature" in saved_meta, (
        f"{algorithm} frozen reference checkpoint is missing tokenizer_signature in {meta_path}"
    ))
    require(saved_meta["tokenizer_signature"] == tokenizer_sig, (
        f"{algorithm} frozen reference tokenizer mismatch: "
        f"saved={saved_meta['tokenizer_signature'][:12]}... current={tokenizer_sig[:12]}..."
    ))


def _load_reference_model(model, ref_model_path, device, algorithm):
    model = unwrap_model(model)
    require(isinstance(model, BaseModel), (
        f"{algorithm} requires a BaseModel trainable model so the frozen reference "
        f"can be loaded from the validated checkpoint path"
    ))
    ref_model = type(model).load(ref_model_path, device=device).eval()
    require(ref_model.config.to_dict() == model.config.to_dict(), (
        f"{algorithm} frozen reference config does not match the trainable model config"
    ))
    for p in ref_model.parameters():
        p.requires_grad = False
    return ref_model


def _seq_logp(model, input_ids, labels):
    token_logp, mask = _token_logp(model, input_ids, labels)
    return (token_logp * mask).sum(dim=-1)


def _seq_avg_logp(model, input_ids, labels):
    token_logp, mask = _token_logp(model, input_ids, labels)
    counts = mask.sum(dim=-1)
    require((counts > 0).all(), "sequence average log-probability requires at least one supervised token")
    return (token_logp * mask).sum(dim=-1) / counts


def _token_logp(model, input_ids, labels):
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits, dim=-1)
    mask = (labels != -100).float()
    safe_targets = labels.where(mask.bool(), torch.zeros_like(labels))
    token_logp = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return token_logp, mask


def _log1mexp(logp):
    eps = torch.finfo(logp.dtype).eps
    return torch.log1p(-logp.exp().clamp(max=1.0 - eps))


def _kto_kl_batch(batch):
    input_ids = batch["input_ids"]
    prompt_lens = batch["prompt_len"]
    response_lens = batch["response_len"]
    B, T = input_ids.shape
    require(B > 1, "KTO KL estimate requires batch_size > 1")
    kl_ids = torch.zeros_like(input_ids)
    kl_labels = torch.full_like(input_ids, -100)
    for b in range(B):
        source = (b - 1) % B
        prompt_len = int(prompt_lens[b].item())
        source_prompt_len = int(prompt_lens[source].item())
        response_len = min(int(response_lens[source].item()), T - prompt_len)
        require(response_len > 0, "KTO KL batch requires at least one response token after truncation")
        full = torch.cat([
            input_ids[b, :prompt_len],
            input_ids[source, source_prompt_len : source_prompt_len + response_len],
        ])
        kl_ids[b, : full.size(0)] = full
        for pos in range(prompt_len - 1, full.size(0) - 1):
            kl_labels[b, pos] = full[pos + 1]
    return kl_ids, kl_labels


def _masked_response_mean(values, mask, context):
    token_counts = mask.sum(dim=-1)
    require((token_counts > 0).all(), f"{context} requires at least one generated token per response")
    return ((values * mask).sum(dim=-1) / token_counts).mean()


def _whiten_masked(values, mask, eps=1e-8):
    selected = values[mask.bool()]
    require(selected.numel() > 0, "masked whitening requires at least one value")
    mean = selected.mean()
    std = selected.std(unbiased=False)
    whitened = (values - mean) / (std + eps)
    return whitened * mask


def _generation_context_token_logp(model, input_ids, labels):
    """Token log-probs under the same cropped context contract used by generate()."""
    max_ctx = _model_max_seq_len(model, "GRPO token log-prob scoring")
    if input_ids.size(1) <= max_ctx:
        return _token_logp(model, input_ids, labels)

    mask = labels != -100
    token_logp = None
    safe_targets = labels.where(mask, torch.zeros_like(labels))
    for pos in range(input_ids.size(1)):
        active = mask[:, pos]
        if not active.any():
            continue
        start = max(0, pos + 1 - max_ctx)
        context = input_ids[active, start : pos + 1]
        logits, _ = model(context)
        selected = F.log_softmax(logits[:, -1], dim=-1).gather(
            -1,
            safe_targets[active, pos].unsqueeze(-1),
        ).squeeze(-1)
        if token_logp is None:
            token_logp = torch.zeros(labels.shape, device=input_ids.device, dtype=selected.dtype)
        token_logp[active, pos] = selected

    if token_logp is None:
        token_logp = torch.zeros(labels.shape, device=input_ids.device, dtype=torch.float32)
    return token_logp, mask.float()


def _model_max_seq_len(model, context):
    model = unwrap_model(model)
    require(isinstance(model, BaseModel), f"{context} requires a BaseModel")
    max_seq_len = model.config.max_seq_len
    require(max_seq_len > 0, f"{context} requires model.config.max_seq_len > 0")
    return max_seq_len


def _diffusion_loss_per_example(model, fwd, x_0, loss_mask, t, z_t, mask):
    core = unwrap_model(model)
    require(isinstance(core, BaseModel), "diffusion preference tuning requires a BaseModel")
    output = model(z_t, t)
    return core.compute_loss_per_example(
        output,
        x_0,
        mask,
        t,
        fwd,
        loss_mask=loss_mask,
        normalization="none",
    )
