import torch

from minilab.base import BaseModel, unwrap_model
from minilab.checks import require, require_finite_number, require_integer
from minilab.config import BaseConfig
from minilab.diffusion_sampling import (
    absorbing_unmask_probability as _absorbing_unmask_probability,
    d3pm_reverse_timesteps,
    dream_remask_step,
    dream_sample_tokens,
    dream_transfer_count,
    sample_categorical,
    sample_clean_logits,
    sample_logits,
    sedd_absorbing_step_probs,
)
from minilab.models.d3pm import absorbing_posterior_log_probs
from minilab.models.diffusion_base import DiffusionModelConfig, validate_infill_tokens
from minilab.registry import register_sampler


def _apply_repetition_penalty(logits, ids, repetition_penalty):
    if repetition_penalty == 1.0:
        return logits
    logits = logits.clone()
    for b in range(ids.size(0)):
        seen = ids[b].unique()
        seen_logits = logits[b, seen]
        logits[b, seen] = torch.where(
            seen_logits < 0,
            seen_logits * repetition_penalty,
            seen_logits / repetition_penalty,
        )
    return logits


def _apply_top_k_top_p(logits, top_k, top_p):
    if 0 < top_k < logits.size(-1):
        logits = logits.clone()
        cutoff = logits.topk(top_k).values[:, -1:]
        logits[logits < cutoff] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = logits.sort(descending=True)
        sorted_probs = sorted_logits.softmax(-1)
        remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
        sorted_logits[remove] = float("-inf")
        filtered_logits = torch.full_like(logits, float("-inf"))
        logits = filtered_logits.scatter(-1, sorted_idx, sorted_logits)
    return logits


def _sample_next_token(logits, temperature, top_k=0, top_p=1.0):
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = _apply_top_k_top_p(logits / temperature, top_k, top_p)
    return torch.multinomial(logits.softmax(-1), 1)


def _require_greedy_verified_decode(
    prompt_ids,
    max_new_tokens,
    max_ctx,
    context,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
):
    require_integer(max_new_tokens, "max_new_tokens")
    require_integer(top_k, "top_k")
    require(prompt_ids.size(0) == 1, f"{context} currently supports batch_size=1")
    require(prompt_ids.size(1) > 0, f"{context} requires a non-empty prompt")
    require(max_new_tokens >= 0, "max_new_tokens must be >= 0")
    require(prompt_ids.size(1) + max_new_tokens <= max_ctx, (
        f"{context} requires prompt length + max_new_tokens <= max_seq_len"
    ))
    require(temperature == 0, f"{context} currently supports exact greedy decoding only")
    require(top_k == 0, f"{context} does not support top_k")
    require(top_p == 1.0, f"{context} does not support top_p")
    require(repetition_penalty == 1.0, f"{context} does not support repetition_penalty")


def _validate_prompt_token_ids(prompt_ids, vocab_size, context):
    require(torch.is_tensor(prompt_ids), f"{context} prompt_ids must be a tensor")
    require(prompt_ids.dim() == 2, f"{context} prompt_ids must have shape (batch, seq)")
    require(prompt_ids.dtype == torch.long, f"{context} prompt_ids must use dtype torch.long")
    require(prompt_ids.size(1) > 0, f"{context} requires a non-empty prompt")
    require(type(vocab_size) is int and vocab_size > 0, (
        f"{context} requires model.config.vocab_size to be a positive integer"
    ))
    require((prompt_ids >= 0).all() and (prompt_ids < vocab_size).all(), (
        f"{context} prompt token ids must be in [0, {vocab_size})"
    ))


def _config_vocab_size(model_core, context):
    require(isinstance(model_core.config, BaseConfig), (
        f"{context} requires model.config to inherit BaseConfig"
    ))
    config_state = model_core.config.to_dict()
    require("vocab_size" in config_state, (
        f"{context} requires model.config.vocab_size to validate prompt token ids"
    ))
    return config_state["vocab_size"]


def _stop_requested(ids, prompt_len, stop_texts, tokenizer):
    if not stop_texts:
        return False
    tail = tokenizer.decode(ids[0, prompt_len:].tolist())
    return any(text in tail for text in stop_texts)


def _append_verified_greedy(model, ids, draft):
    require(draft.dim() == 2 and draft.size(0) == ids.size(0), "draft tokens must match batch size")
    require(draft.size(1) > 0, "draft verification requires at least one draft token")
    prefix_len = ids.size(1)
    verify_ids = torch.cat([ids, draft], dim=1)
    logits, _ = model(verify_ids)
    return _append_verified_greedy_from_logits(ids, draft, logits, prefix_len)


def _append_verified_greedy_from_logits(ids, draft, logits, prefix_len):
    accepted = 0
    for pos in range(draft.size(1)):
        greedy = logits[:, prefix_len + pos - 1].argmax(dim=-1, keepdim=True)
        if torch.equal(greedy, draft[:, pos : pos + 1]):
            accepted += 1
        else:
            break

    if accepted == 0:
        fallback = logits[:, prefix_len - 1].argmax(dim=-1, keepdim=True)
        return torch.cat([ids, fallback], dim=1), 1
    return torch.cat([ids, draft[:, :accepted]], dim=1), accepted


def _cached_greedy_acceptance(prefix_logits, draft_logits, draft):
    accepted = 0
    for pos in range(draft.size(1)):
        logits = prefix_logits[:, -1] if pos == 0 else draft_logits[:, pos - 1]
        greedy = logits.argmax(dim=-1, keepdim=True)
        if torch.equal(greedy, draft[:, pos : pos + 1]):
            accepted += 1
        else:
            break
    return accepted


def _append_verified_greedy_cached(model, ids, draft, cache, prefix_logits):
    require(draft.dim() == 2 and draft.size(0) == ids.size(0), "draft tokens must match batch size")
    require(draft.size(1) > 0, "cached draft verification requires at least one draft token")
    draft_logits, draft_cache, draft_hidden = model.forward_cached(draft, cache, return_hidden=True)
    accepted = _cached_greedy_acceptance(prefix_logits, draft_logits, draft)

    if accepted == 0:
        fallback = prefix_logits[:, -1].argmax(dim=-1, keepdim=True)
        logits, cache, hidden = model.forward_cached(fallback, cache, return_hidden=True)
        return torch.cat([ids, fallback], dim=1), cache, logits, hidden, 1
    if accepted == draft.size(1):
        return torch.cat([ids, draft], dim=1), draft_cache, draft_logits, draft_hidden, accepted

    accepted_tokens = draft[:, :accepted]
    logits, cache, hidden = model.forward_cached(accepted_tokens, cache, return_hidden=True)
    return torch.cat([ids, accepted_tokens], dim=1), cache, logits, hidden, accepted


def _candidate_tree_paths(draft_logits, tree_width, tree_depth, max_tree_paths):
    require_integer(tree_width, "tree_width")
    require_integer(tree_depth, "tree_depth")
    require_integer(max_tree_paths, "max_tree_paths")
    require(tree_width > 0, "tree_width must be > 0")
    require(tree_depth > 0, "tree_depth must be > 0")
    require(max_tree_paths > 0, "max_tree_paths must be > 0")
    require(draft_logits, "tree candidate generation requires at least one draft logit")
    require(draft_logits[0].size(0) == 1, "tree candidate generation currently supports batch_size=1")

    paths = [[]]
    for logits in draft_logits[:tree_depth]:
        width = min(tree_width, logits.size(-1))
        top_tokens = logits.topk(width, dim=-1).indices[0].tolist()
        expanded = []
        for path in paths:
            for token in top_tokens:
                expanded.append([*path, token])
                if len(expanded) >= max_tree_paths:
                    break
            if len(expanded) >= max_tree_paths:
                break
        paths = expanded
    return torch.tensor(paths, dtype=torch.long, device=draft_logits[0].device)


def _append_verified_tree_greedy(model, ids, paths):
    require(ids.size(0) == 1, "tree verification currently supports batch_size=1")
    require(paths.dim() == 2 and paths.size(0) > 0 and paths.size(1) > 0, (
        "tree verification requires non-empty candidate paths"
    ))

    prefix_len = ids.size(1)
    verify_ids = torch.cat([ids.expand(paths.size(0), -1), paths], dim=1)
    logits, _ = model(verify_ids)
    best_path = 0
    best_accepted = 0
    for path_idx in range(paths.size(0)):
        accepted = 0
        for pos in range(paths.size(1)):
            greedy = logits[path_idx : path_idx + 1, prefix_len + pos - 1].argmax(
                dim=-1,
                keepdim=True,
            )
            if torch.equal(greedy, paths[path_idx : path_idx + 1, pos : pos + 1]):
                accepted += 1
            else:
                break
        if accepted > best_accepted:
            best_path = path_idx
            best_accepted = accepted

    if best_accepted == 0:
        fallback = logits[0:1, prefix_len - 1].argmax(dim=-1, keepdim=True)
        return torch.cat([ids, fallback], dim=1), 1
    return torch.cat([ids, paths[best_path : best_path + 1, :best_accepted]], dim=1), best_accepted


def _fill_remaining_masks(model, z, mask_id, batch_size):
    still_masked = z == mask_id
    if not still_masked.any():
        return z
    predictions = sample_clean_logits(
        model(z, torch.zeros(batch_size, device=z.device)),
        mask_id,
        temperature=0,
    )
    return torch.where(still_masked, predictions, z)


def _sample_clean_logits_reverse(model, fwd, batch_size, seq_len, num_steps, temperature, cache_interval=1):
    device = next(model.parameters()).device
    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    logits = None
    for i in range(num_steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]
        if i % cache_interval == 0:
            logits = model(z, t_now.expand(batch_size))
        predictions = sample_clean_logits(logits, mask_id, temperature)
        unmask_prob = _absorbing_unmask_probability(fwd, t_now, t_next)

        unmask = torch.rand(batch_size, seq_len, device=device) < unmask_prob
        z = torch.where((z == mask_id) & unmask, predictions, z)

    return _fill_remaining_masks(model, z, mask_id, batch_size)


@torch.no_grad()
def generate(
    model,
    prompt_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
    use_cache=True,
):
    """Autoregressive sampling. temperature=0 for greedy.
    stop_texts: list of strings that trigger early stopping (batch_size=1 only, requires tokenizer)."""
    _require_eval_model(model, "generate")
    model_core = _require_base_model(model, "generate")
    _validate_prompt_token_ids(prompt_ids, _config_vocab_size(model_core, "generate"), "generate")
    for name, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("repetition_penalty", repetition_penalty),
    ):
        require_finite_number(value, name)
    require_integer(max_new_tokens, "max_new_tokens")
    require_integer(top_k, "top_k")
    require(max_new_tokens >= 0, "max_new_tokens must be >= 0")
    require(temperature >= 0, "temperature must be >= 0")
    require(top_k >= 0, "top_k must be >= 0")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(repetition_penalty > 0, "repetition_penalty must be > 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
        require(prompt_ids.size(0) == 1, "stop_texts only supported for batch_size=1")
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    if max_new_tokens == 0:
        return ids
    max_ctx = model_core.config.max_seq_len
    prompt_len = ids.size(1)
    can_use_cache = (
        use_cache
        and model_core.supports_kv_cache()
        and prompt_len + max_new_tokens <= max_ctx
    )
    if can_use_cache:
        return _generate_cached(
            model_core,
            ids,
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            stop_texts,
            tokenizer,
            prompt_len,
        )

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -max_ctx:])
        logits = logits[:, -1]

        logits = _apply_repetition_penalty(logits, ids, repetition_penalty)
        next_id = _sample_next_token(logits, temperature, top_k, top_p)

        ids = torch.cat([ids, next_id], dim=1)

        if stop_texts:
            tail = tokenizer.decode(ids[0, prompt_len:].tolist())
            if any(s in tail for s in stop_texts):
                break

    return ids


def _generate_cached(
    model,
    ids,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    stop_texts,
    tokenizer,
    prompt_len,
):
    logits, cache = model.forward_cached(ids)
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1]
        next_logits = _apply_repetition_penalty(next_logits, ids, repetition_penalty)
        next_id = _sample_next_token(next_logits, temperature, top_k, top_p)
        ids = torch.cat([ids, next_id], dim=1)

        if stop_texts:
            tail = tokenizer.decode(ids[0, prompt_len:].tolist())
            if any(s in tail for s in stop_texts):
                break
        if ids.size(1) == prompt_len + max_new_tokens:
            break
        logits, cache = model.forward_cached(next_id, cache)
    return ids


@torch.no_grad()
def generate_mtp_speculative(
    model,
    prompt_ids,
    max_new_tokens=100,
    draft_tokens=None,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
):
    """Greedy draft/verify decoding using parallel MTP heads."""
    _require_eval_model(model, "generate_mtp_speculative")
    model_core = _require_base_model(model, "generate_mtp_speculative")
    _validate_prompt_token_ids(
        prompt_ids,
        _config_vocab_size(model_core, "generate_mtp_speculative"),
        "generate_mtp_speculative",
    )
    require_finite_number(top_p, "top_p")
    require_integer(top_k, "top_k")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
    _require_greedy_verified_decode(
        prompt_ids,
        max_new_tokens,
        model_core.config.max_seq_len,
        "generate_mtp_speculative",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    require(model_core.supports_parallel_mtp_drafting(), (
        "generate_mtp_speculative requires a GPT checkpoint trained with mtp_mode='parallel'"
    ))
    if draft_tokens is None:
        draft_tokens = model_core.config.mtp_depth + 1
    require_integer(draft_tokens, "draft_tokens")
    require(0 < draft_tokens <= model_core.config.mtp_depth + 1, (
        "draft_tokens must be in [1, mtp_depth + 1]"
    ))

    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    prompt_len = ids.size(1)
    while ids.size(1) < prompt_len + max_new_tokens:
        remaining = prompt_len + max_new_tokens - ids.size(1)
        draft_logits = model_core.mtp_draft_logits(ids)
        draft = torch.cat(
            [logits.argmax(dim=-1, keepdim=True) for logits in draft_logits[: min(draft_tokens, remaining)]],
            dim=1,
        )
        ids, _ = _append_verified_greedy(model_core, ids, draft)
        if _stop_requested(ids, prompt_len, stop_texts, tokenizer):
            break
    return ids[:, : prompt_len + max_new_tokens]


@torch.no_grad()
def generate_mtp_speculative_cached(
    model,
    prompt_ids,
    max_new_tokens=100,
    draft_tokens=None,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
):
    """Greedy MTP draft/verify decoding that reuses the verifier KV cache."""
    _require_eval_model(model, "generate_mtp_speculative_cached")
    model_core = _require_base_model(model, "generate_mtp_speculative_cached")
    _validate_prompt_token_ids(
        prompt_ids,
        _config_vocab_size(model_core, "generate_mtp_speculative_cached"),
        "generate_mtp_speculative_cached",
    )
    require_finite_number(top_p, "top_p")
    require_integer(top_k, "top_k")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
    _require_greedy_verified_decode(
        prompt_ids,
        max_new_tokens,
        model_core.config.max_seq_len,
        "generate_mtp_speculative_cached",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    require(model_core.supports_parallel_mtp_drafting(), (
        "generate_mtp_speculative_cached requires a GPT checkpoint trained with mtp_mode='parallel'"
    ))
    require(model_core.supports_cached_parallel_mtp_drafting(), (
        "generate_mtp_speculative_cached requires a cache-compatible residual GPT checkpoint"
    ))
    if draft_tokens is None:
        draft_tokens = model_core.config.mtp_depth + 1
    require_integer(draft_tokens, "draft_tokens")
    require(0 < draft_tokens <= model_core.config.mtp_depth + 1, (
        "draft_tokens must be in [1, mtp_depth + 1]"
    ))

    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    prompt_len = ids.size(1)
    if max_new_tokens == 0:
        return ids

    logits, cache, hidden = model_core.forward_cached(ids, return_hidden=True)
    while ids.size(1) < prompt_len + max_new_tokens:
        remaining = prompt_len + max_new_tokens - ids.size(1)
        draft_logits = model_core.mtp_draft_logits_from_hidden(logits, hidden)
        draft = torch.cat(
            [logits.argmax(dim=-1, keepdim=True) for logits in draft_logits[: min(draft_tokens, remaining)]],
            dim=1,
        )
        ids, cache, logits, hidden, _ = _append_verified_greedy_cached(
            model_core,
            ids,
            draft,
            cache,
            logits,
        )
        if _stop_requested(ids, prompt_len, stop_texts, tokenizer):
            break
    return ids[:, : prompt_len + max_new_tokens]


@torch.no_grad()
def generate_mtp_tree(
    model,
    prompt_ids,
    max_new_tokens=100,
    tree_width=2,
    tree_depth=None,
    max_tree_paths=32,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
):
    """Greedy Medusa-style MTP tree proposals with exact batched verification."""
    _require_eval_model(model, "generate_mtp_tree")
    model_core = _require_base_model(model, "generate_mtp_tree")
    _validate_prompt_token_ids(
        prompt_ids,
        _config_vocab_size(model_core, "generate_mtp_tree"),
        "generate_mtp_tree",
    )
    require_finite_number(top_p, "top_p")
    require_integer(top_k, "top_k")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    require_integer(tree_width, "tree_width")
    require_integer(max_tree_paths, "max_tree_paths")
    require(tree_width > 0, "tree_width must be > 0")
    require(max_tree_paths > 0, "max_tree_paths must be > 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
    _require_greedy_verified_decode(
        prompt_ids,
        max_new_tokens,
        model_core.config.max_seq_len,
        "generate_mtp_tree",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    require(model_core.supports_parallel_mtp_drafting(), (
        "generate_mtp_tree requires a GPT checkpoint trained with mtp_mode='parallel'"
    ))
    if tree_depth is None:
        tree_depth = model_core.config.mtp_depth + 1
    require_integer(tree_depth, "tree_depth")
    require(tree_depth > 0, "tree_depth must be > 0")
    require(tree_depth <= model_core.config.mtp_depth + 1, (
        "tree_depth must be in [1, mtp_depth + 1]"
    ))

    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    prompt_len = ids.size(1)
    while ids.size(1) < prompt_len + max_new_tokens:
        remaining = prompt_len + max_new_tokens - ids.size(1)
        draft_logits = model_core.mtp_draft_logits(ids)
        current_depth = min(tree_depth, remaining, len(draft_logits))
        paths = _candidate_tree_paths(draft_logits, tree_width, current_depth, max_tree_paths)
        ids, _ = _append_verified_tree_greedy(model_core, ids, paths)
        if _stop_requested(ids, prompt_len, stop_texts, tokenizer):
            break
    return ids[:, : prompt_len + max_new_tokens]


@torch.no_grad()
def generate_self_speculative(
    model,
    prompt_ids,
    max_new_tokens=100,
    exit_layer=None,
    draft_tokens=4,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
):
    """LayerSkip-style greedy self-speculative decoding with full-model verification."""
    _require_eval_model(model, "generate_self_speculative")
    model_core = _require_base_model(model, "generate_self_speculative")
    _validate_prompt_token_ids(
        prompt_ids,
        _config_vocab_size(model_core, "generate_self_speculative"),
        "generate_self_speculative",
    )
    require_finite_number(top_p, "top_p")
    require_integer(top_k, "top_k")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    require_integer(draft_tokens, "draft_tokens")
    require(draft_tokens > 0, "draft_tokens must be > 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
    _require_greedy_verified_decode(
        prompt_ids,
        max_new_tokens,
        model_core.config.max_seq_len,
        "generate_self_speculative",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    require(model_core.supports_early_exit(), "generate_self_speculative requires GPT early-exit support")
    if exit_layer is None:
        exit_layer = max(1, model_core.config.num_layers // 2)
    require_integer(exit_layer, "exit_layer")
    require(0 < exit_layer < model_core.config.num_layers, "exit_layer must be in [1, num_layers)")

    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    prompt_len = ids.size(1)
    while ids.size(1) < prompt_len + max_new_tokens:
        draft = ids.new_empty((ids.size(0), 0))
        draft_ids = ids
        for _ in range(min(draft_tokens, prompt_len + max_new_tokens - ids.size(1))):
            logits = model_core.early_exit_logits(draft_ids, exit_layer)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            draft = torch.cat([draft, next_id], dim=1)
            draft_ids = torch.cat([draft_ids, next_id], dim=1)
        ids, _ = _append_verified_greedy(model_core, ids, draft)
        if _stop_requested(ids, prompt_len, stop_texts, tokenizer):
            break
    return ids[:, : prompt_len + max_new_tokens]


@torch.no_grad()
def generate_self_speculative_shared(
    model,
    prompt_ids,
    max_new_tokens=100,
    exit_layer=None,
    draft_tokens=4,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
):
    """LayerSkip-style greedy self-speculative decoding with shared verification activations."""
    _require_eval_model(model, "generate_self_speculative_shared")
    model_core = _require_base_model(model, "generate_self_speculative_shared")
    _validate_prompt_token_ids(
        prompt_ids,
        _config_vocab_size(model_core, "generate_self_speculative_shared"),
        "generate_self_speculative_shared",
    )
    require_finite_number(top_p, "top_p")
    require_integer(top_k, "top_k")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    require_integer(draft_tokens, "draft_tokens")
    require(draft_tokens > 0, "draft_tokens must be > 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
    _require_greedy_verified_decode(
        prompt_ids,
        max_new_tokens,
        model_core.config.max_seq_len,
        "generate_self_speculative_shared",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )
    require(model_core.supports_early_exit(), (
        "generate_self_speculative_shared requires GPT early-exit state support"
    ))
    require(model_core.supports_hidden_continuation(), (
        "generate_self_speculative_shared requires residual GPT hidden continuation without per-layer embeddings"
    ))
    if exit_layer is None:
        exit_layer = max(1, model_core.config.num_layers // 2)
    require_integer(exit_layer, "exit_layer")
    require(0 < exit_layer < model_core.config.num_layers, "exit_layer must be in [1, num_layers)")

    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    prompt_len = ids.size(1)
    while ids.size(1) < prompt_len + max_new_tokens:
        draft = ids.new_empty((ids.size(0), 0))
        draft_ids = ids
        for _ in range(min(draft_tokens, prompt_len + max_new_tokens - ids.size(1))):
            logits = model_core.early_exit_logits(draft_ids, exit_layer)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            draft = torch.cat([draft, next_id], dim=1)
            draft_ids = torch.cat([draft_ids, next_id], dim=1)

        prefix_len = ids.size(1)
        verify_ids = torch.cat([ids, draft], dim=1)
        _, hidden = model_core.early_exit_state(verify_ids, exit_layer)
        full_logits = model_core.continue_from_hidden(hidden, exit_layer)
        ids, _ = _append_verified_greedy_from_logits(ids, draft, full_logits, prefix_len)
        if _stop_requested(ids, prompt_len, stop_texts, tokenizer):
            break
    return ids[:, : prompt_len + max_new_tokens]


@torch.no_grad()
def generate_jacobi(
    model,
    prompt_ids,
    max_new_tokens=100,
    block_size=4,
    iterations=4,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
    repetition_penalty=1.0,
    stop_texts=None,
    tokenizer=None,
):
    """Greedy Jacobi-style parallel drafts with full-model verification."""
    _require_eval_model(model, "generate_jacobi")
    model_core = _require_base_model(model, "generate_jacobi")
    _validate_prompt_token_ids(
        prompt_ids,
        _config_vocab_size(model_core, "generate_jacobi"),
        "generate_jacobi",
    )
    require_finite_number(top_p, "top_p")
    require_integer(top_k, "top_k")
    require_integer(block_size, "block_size")
    require_integer(iterations, "iterations")
    require(block_size > 0, "block_size must be > 0")
    require(iterations > 0, "iterations must be > 0")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
    _require_greedy_verified_decode(
        prompt_ids,
        max_new_tokens,
        model_core.config.max_seq_len,
        "generate_jacobi",
        temperature,
        top_k,
        top_p,
        repetition_penalty,
    )

    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    prompt_len = ids.size(1)
    while ids.size(1) < prompt_len + max_new_tokens:
        remaining = prompt_len + max_new_tokens - ids.size(1)
        current_block = min(block_size, remaining)
        logits, _ = model_core(ids)
        first = logits[:, -1].argmax(dim=-1, keepdim=True)
        draft = first.expand(ids.size(0), current_block).clone()
        prefix_len = ids.size(1)
        for _ in range(iterations):
            trial = torch.cat([ids, draft], dim=1)
            logits, _ = model_core(trial)
            next_tokens = [
                logits[:, prefix_len + pos - 1].argmax(dim=-1, keepdim=True)
                for pos in range(current_block)
            ]
            updated = torch.cat(next_tokens, dim=1)
            if torch.equal(updated, draft):
                break
            draft = updated
        ids, _ = _append_verified_greedy(model_core, ids, draft)
        if _stop_requested(ids, prompt_len, stop_texts, tokenizer):
            break
    return ids[:, : prompt_len + max_new_tokens]


@register_sampler("ancestral")
@torch.no_grad()
def sample_diffusion(model, fwd, batch_size, seq_len, num_steps=None, temperature=1.0):
    """Standard absorbing-mask reverse process: predict clean tokens, then unmask."""
    _require_eval_model(model, "sample_diffusion")
    _require_sampler_contract(model, fwd, "clean_logits", "sample_diffusion")
    _require_terminal_mask_prior(fwd, "sample_diffusion")
    if num_steps is None:
        num_steps = min(256, fwd.num_timesteps)
    for name, value in (
        ("temperature", temperature),
    ):
        require_finite_number(value, name)
    require_integer(batch_size, "batch_size")
    require_integer(seq_len, "seq_len")
    require_integer(num_steps, "num_steps")
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(0 < num_steps <= fwd.num_timesteps, "num_steps must be in [1, fwd.num_timesteps]")
    require(temperature >= 0, "temperature must be >= 0")
    return _sample_clean_logits_reverse(model, fwd, batch_size, seq_len, num_steps, temperature)


@register_sampler("ddpm_cache")
@torch.no_grad()
def sample_diffusion_cached(model, fwd, batch_size, seq_len, num_steps=None, temperature=1.0, cache_interval=4):
    """Reuse clean-token predictions for cache_interval steps. ~cache_interval x fewer forward passes."""
    _require_eval_model(model, "sample_diffusion_cached")
    _require_sampler_contract(model, fwd, "clean_logits", "sample_diffusion_cached")
    _require_terminal_mask_prior(fwd, "sample_diffusion_cached")
    if num_steps is None:
        num_steps = min(256, fwd.num_timesteps)
    for name, value in (
        ("temperature", temperature),
    ):
        require_finite_number(value, name)
    require_integer(batch_size, "batch_size")
    require_integer(seq_len, "seq_len")
    require_integer(num_steps, "num_steps")
    require_integer(cache_interval, "cache_interval")
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(0 < num_steps <= fwd.num_timesteps, "num_steps must be in [1, fwd.num_timesteps]")
    require(temperature >= 0, "temperature must be >= 0")
    require(cache_interval > 0, "cache_interval must be > 0")
    return _sample_clean_logits_reverse(
        model,
        fwd,
        batch_size,
        seq_len,
        num_steps,
        temperature,
        cache_interval=cache_interval,
    )


@register_sampler("sedd_analytical")
@torch.no_grad()
def sample_sedd(model, fwd, batch_size, seq_len, num_steps=None, temperature=1.0):
    """SEDD analytical sampler for absorbing noise (Lou et al. 2024, Algorithm 4).
    The model emits log score ratios; the absorbing graph combines staggered
    scores with the transposed transition kernel for each sigma step."""
    _require_eval_model(model, "sample_sedd")
    _require_sampler_contract(model, fwd, "sedd_log_scores", "sample_sedd")
    if num_steps is None:
        num_steps = min(256, fwd.num_timesteps)
    for name, value in (
        ("temperature", temperature),
    ):
        require_finite_number(value, name)
    require_integer(batch_size, "batch_size")
    require_integer(seq_len, "seq_len")
    require_integer(num_steps, "num_steps")
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(0 < num_steps <= fwd.num_timesteps, "num_steps must be in [1, fwd.num_timesteps]")
    require(temperature >= 0, "temperature must be >= 0")
    device = next(model.parameters()).device

    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    eps = 1.0 / fwd.num_timesteps
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

    for i in range(num_steps):
        masked = z == mask_id
        if not masked.any():
            break

        t_now, t_next = timesteps[i], timesteps[i + 1]
        log_scores = model(z, t_now.expand(batch_size))
        sigma_now = fwd.get_sigma(t_now.unsqueeze(0)).to(device)
        sigma_next = fwd.get_sigma(t_next.unsqueeze(0)).to(device)
        probs = sedd_absorbing_step_probs(log_scores, z, sigma_now - sigma_next, mask_id, temperature)
        z = sample_categorical(probs)

    still_masked = z == mask_id
    if still_masked.any():
        t_eps = timesteps[-1]
        log_scores = model(z, t_eps.expand(batch_size))
        sigma = fwd.get_sigma(t_eps.unsqueeze(0)).to(device)
        probs = sedd_absorbing_step_probs(log_scores, z, sigma, mask_id, temperature, drop_mask=True)
        z = torch.where(still_masked, sample_categorical(probs), z)

    return z


@register_sampler("d3pm_ancestral")
@torch.no_grad()
def sample_d3pm(model, fwd, batch_size, seq_len, num_steps=None, temperature=1.0):
    """Absorbing-chain ancestral sampler for D3PM.

    The model predicts clean-token logits. The sampler combines those logits with
    the absorbing posterior for each chosen interval, so it can use the full chain
    or a smaller number of skip steps."""
    _require_eval_model(model, "sample_d3pm")
    _require_sampler_contract(model, fwd, "d3pm_x0_logits", "sample_d3pm")
    if num_steps is None:
        num_steps = min(256, fwd.num_timesteps)
    for name, value in (
        ("temperature", temperature),
    ):
        require_finite_number(value, name)
    require_integer(batch_size, "batch_size")
    require_integer(seq_len, "seq_len")
    require_integer(num_steps, "num_steps")
    require(0 < num_steps <= fwd.num_timesteps, (
        "D3PM num_steps must be in [1, fwd.num_timesteps]"
    ))
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(temperature >= 0, "temperature must be >= 0")
    device = next(model.parameters()).device
    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = d3pm_reverse_timesteps(fwd, num_steps, device)

    for i in range(num_steps):
        masked = z == mask_id
        if not masked.any():
            break

        t_now, t_prev = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(batch_size))
        log_probs = absorbing_posterior_log_probs(
            logits, z, t_now.expand(batch_size), t_prev.expand(batch_size), fwd, mask_id
        )
        predictions = sample_logits(log_probs, temperature)
        z = torch.where(masked, predictions, z)

    still_masked = z == mask_id
    require(not still_masked.any(), "D3PM reverse chain left masked tokens at t=0")
    return z


@torch.no_grad()
def infill(model, fwd, tokens, mask_positions, num_steps=None, temperature=1.0):
    """Fill masked positions while keeping context fixed. Unique to diffusion models."""
    _require_eval_model(model, "infill")
    require_finite_number(temperature, "temperature")
    require(temperature >= 0, "temperature must be >= 0")
    parameterization = _reverse_parameterization(model)
    if num_steps is None:
        num_steps = min(128, fwd.num_timesteps)
    require_integer(num_steps, "num_steps")
    require(num_steps > 0, "num_steps must be > 0")
    model_config = _require_absorbing_forward_process(model, fwd, "infill")
    validate_infill_tokens(tokens, mask_positions.to(tokens.device), model_config, "infill")
    device = next(model.parameters()).device
    if not mask_positions.any():
        return tokens.to(device)
    if parameterization == "clean_logits":
        _require_terminal_mask_prior(fwd, "infill")
        return _infill_clean_logits(model, fwd, tokens, mask_positions, num_steps, temperature)
    if parameterization == "sedd_log_scores":
        return _infill_sedd(model, fwd, tokens, mask_positions, num_steps, temperature)
    if parameterization == "d3pm_x0_logits":
        return _infill_d3pm(model, fwd, tokens, mask_positions, num_steps, temperature)
    raise ValueError(f"infill does not support reverse parameterization: {parameterization!r}")


@register_sampler("semi_ar")
@torch.no_grad()
def sample_diffusion_semi_ar(
    model,
    fwd,
    prompt_ids,
    max_new_tokens,
    block_size=16,
    num_steps=None,
    temperature=1.0,
):
    """Semi-autoregressive diffusion sampling by denoising masked blocks.

    Each new block starts as [MASK] while all previous blocks are fixed context.
    The block itself is denoised with the model-specific infill sampler, so the
    reverse-process contract remains MDLM/SEDD/D3PM-specific.
    """
    _require_eval_model(model, "sample_diffusion_semi_ar")
    model_config = _require_absorbing_forward_process(model, fwd, "sample_diffusion_semi_ar")
    require(prompt_ids.dim() == 2, "prompt_ids must have shape (batch, seq)")
    require(prompt_ids.dtype == torch.long, "prompt_ids must contain integer token ids")
    for name, value in (
        ("temperature", temperature),
    ):
        require_finite_number(value, name)
    require_integer(max_new_tokens, "max_new_tokens")
    require_integer(block_size, "block_size")
    require(max_new_tokens >= 0, "max_new_tokens must be >= 0")
    require(block_size > 0, "block_size must be > 0")
    require(temperature >= 0, "temperature must be >= 0")
    require(prompt_ids.size(1) + max_new_tokens <= model_config.max_seq_len, (
        f"semi-AR sampling supports at most {model_config.max_seq_len} tokens, "
        f"got {prompt_ids.size(1) + max_new_tokens}"
    ))

    device = next(model.parameters()).device
    tokens = prompt_ids.to(device)
    context_mask = torch.zeros_like(tokens, dtype=torch.bool)
    validate_infill_tokens(tokens, context_mask, model_config, "sample_diffusion_semi_ar")
    if max_new_tokens == 0:
        return tokens

    remaining = max_new_tokens
    mask_id = fwd.mask_token_id
    while remaining > 0:
        current = min(block_size, remaining)
        block = torch.full((tokens.size(0), current), mask_id, device=device, dtype=torch.long)
        tokens = torch.cat([tokens, block], dim=1)
        mask_positions = torch.zeros_like(tokens, dtype=torch.bool)
        mask_positions[:, -current:] = True
        tokens = infill(model, fwd, tokens, mask_positions, num_steps=num_steps, temperature=temperature)
        remaining -= current
    return tokens


@register_sampler("dream")
@torch.no_grad()
def sample_diffusion_dream(
    model,
    fwd,
    prompt_ids,
    max_new_tokens,
    steps=None,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    alg="entropy",
    alg_temp=0.0,
):
    """Dream-style masked generation from a prompt.

    All generation slots start as [MASK]. Each reverse step predicts clean
    tokens for the whole sequence and transfers a dynamic share of still-masked
    generation slots according to Dream's confidence policy.
    """
    _require_eval_model(model, "sample_diffusion_dream")
    model_config = _require_sampler_contract(model, fwd, "clean_logits", "sample_diffusion_dream")
    require(prompt_ids.dim() == 2, "prompt_ids must have shape (batch, seq)")
    require(prompt_ids.dtype == torch.long, "prompt_ids must contain integer token ids")
    for name, value in (
        ("temperature", temperature),
        ("top_p", top_p),
        ("alg_temp", alg_temp),
    ):
        require_finite_number(value, name)
    require_integer(max_new_tokens, "max_new_tokens")
    require_integer(top_k, "top_k")
    require(max_new_tokens >= 0, "max_new_tokens must be >= 0")
    require(temperature >= 0, "temperature must be >= 0")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    require(alg in {"origin", "maskgit_plus", "topk_margin", "entropy"}, f"unknown Dream alg: {alg}")
    require(alg_temp >= 0, "alg_temp must be >= 0")
    if steps is None:
        steps = max(1, max_new_tokens)
    require_integer(steps, "steps")
    require(steps > 0, "steps must be > 0")
    require(prompt_ids.size(1) + max_new_tokens <= model_config.max_seq_len, (
        f"Dream sampling supports at most {model_config.max_seq_len} tokens, "
        f"got {prompt_ids.size(1) + max_new_tokens}"
    ))
    device = next(model.parameters()).device
    tokens = prompt_ids.to(device)
    validate_infill_tokens(tokens, torch.zeros_like(tokens, dtype=torch.bool), model_config, "sample_diffusion_dream")
    if max_new_tokens == 0:
        return tokens

    mask_id = fwd.mask_token_id
    masked = torch.full((tokens.size(0), max_new_tokens), mask_id, device=device, dtype=torch.long)
    x = torch.cat([tokens, masked], dim=1)
    prompt_index = torch.zeros_like(x, dtype=torch.bool)
    prompt_index[:, : tokens.size(1)] = True
    eps = 1.0 / fwd.num_timesteps
    timesteps = torch.linspace(1.0, eps, steps + 1, device=device)
    for i in range(steps):
        mask_index = (x == mask_id) & (~prompt_index)
        if not mask_index.any():
            break
        t_now, t_next = timesteps[i], timesteps[i + 1]
        logits = model(x, t_now.expand(x.size(0)))
        final_step = i == steps - 1
        if alg == "origin":
            logits = logits.clone()
            logits[:, :, mask_id] = float("-inf")
            _, x0 = dream_sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k, alg=alg)
            p_transfer = 1.0 if final_step else 1.0 - t_next / t_now
            transfer_index = (torch.rand(x.shape, device=device) < p_transfer) & mask_index
            x = torch.where(transfer_index, x0, x)
        else:
            transfer_count = dream_transfer_count(mask_index, t_now, t_next, final_step)
            x, _, _ = dream_remask_step(
                logits,
                x,
                mask_id,
                transfer_count,
                prompt_index=prompt_index,
                block_end=x.size(1),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                alg=alg,
                alg_temp=alg_temp,
            )
    require(not ((x == mask_id) & (~prompt_index)).any(), "Dream sampling left masked generation tokens")
    return x


def _require_base_model(model, context):
    model = unwrap_model(model)
    require(isinstance(model, BaseModel), f"{context} requires a BaseModel")
    return model


def _require_diffusion_model(model, context):
    model = _require_base_model(model, context)
    require(isinstance(model.config, DiffusionModelConfig), f"{context} requires a DiffusionModelConfig")
    return model


def _reverse_parameterization(model, context="sampler"):
    model = _require_diffusion_model(model, context)
    require(model.reverse_parameterization is not None, f"{context} requires model.reverse_parameterization")
    return model.reverse_parameterization


def _require_eval_model(model, context):
    require(not unwrap_model(model).training, f"{context} expects model.eval() at the call boundary")


def _require_sampler_contract(model, fwd, expected_parameterization, context):
    model_core = _require_diffusion_model(model, context)
    actual = model_core.reverse_parameterization
    require(actual is not None, f"{context} requires model.reverse_parameterization")
    require(
        actual == expected_parameterization,
        f"{context} requires reverse_parameterization={expected_parameterization!r}, got {actual!r}",
    )
    return _require_absorbing_forward_process(model, fwd, context)


def _require_terminal_mask_prior(fwd, context):
    require(fwd.has_terminal_mask_prior(), (
        f"{context} starts reverse sampling from all [MASK] tokens and therefore "
        "requires alpha[-1] = 0 so q(x_T | x_0) matches that terminal prior"
    ))


def _require_absorbing_forward_process(model, fwd, context):
    require(fwd.process_type == "absorbing", f"{context} currently supports only the absorbing forward process")
    model = _require_diffusion_model(model, context)
    model_config = model.config
    require(model.supports_unconditional_diffusion_sampling(), (
        f"{context} requires a diffusion model that can score reverse steps without clean x_0 context"
    ))
    require(
        model_config.mask_token_id == fwd.mask_token_id,
        f"{context} model and forward process must use the same mask_token_id",
    )
    if model.requires_terminal_mask_prior:
        require(fwd.has_terminal_mask_prior(), (
            f"{context} requires a forward process with alpha[-1] = 0 "
            "so q(x_T | x_0) matches the all-mask terminal prior"
        ))
    return model_config


def _infill_clean_logits(model, fwd, tokens, mask_positions, num_steps, temperature):
    require(num_steps <= fwd.num_timesteps, "clean-logits infill num_steps must be <= fwd.num_timesteps")
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    mask_positions = mask_positions.to(device)
    B = tokens.size(0)
    mask_id = fwd.mask_token_id

    z = tokens.clone()
    z[mask_positions] = mask_id
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        masked = mask_positions & (z == mask_id)
        if not masked.any():
            break
        t_now = timesteps[i]
        predictions = sample_clean_logits(model(z, t_now.expand(B)), mask_id, temperature)

        unmask_prob = _absorbing_unmask_probability(fwd, t_now, timesteps[i + 1])

        unmask = torch.rand_like(tokens, dtype=torch.float) < unmask_prob
        z = torch.where(masked & unmask, predictions, z)

    still_masked = (z == mask_id) & mask_positions
    if still_masked.any():
        logits = model(z, torch.zeros(B, device=device))
        z = torch.where(still_masked, sample_clean_logits(logits, mask_id, temperature=0), z)

    return z


def _infill_sedd(model, fwd, tokens, mask_positions, num_steps, temperature):
    require(num_steps <= fwd.num_timesteps, "SEDD infill num_steps must be <= fwd.num_timesteps")
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    mask_positions = mask_positions.to(device)
    B = tokens.size(0)
    mask_id = fwd.mask_token_id

    z = tokens.clone()
    z[mask_positions] = mask_id
    eps = 1.0 / fwd.num_timesteps
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

    for i in range(num_steps):
        masked = (z == mask_id) & mask_positions
        if not masked.any():
            break

        t_now, t_next = timesteps[i], timesteps[i + 1]
        log_scores = model(z, t_now.expand(B))
        sigma_now = fwd.get_sigma(t_now.unsqueeze(0)).to(device)
        sigma_next = fwd.get_sigma(t_next.unsqueeze(0)).to(device)
        probs = sedd_absorbing_step_probs(log_scores, z, sigma_now - sigma_next, mask_id, temperature)
        z = torch.where(mask_positions, sample_categorical(probs), tokens)

    still_masked = (z == mask_id) & mask_positions
    if still_masked.any():
        t_eps = timesteps[-1]
        log_scores = model(z, t_eps.expand(B))
        sigma = fwd.get_sigma(t_eps.unsqueeze(0)).to(device)
        probs = sedd_absorbing_step_probs(log_scores, z, sigma, mask_id, temperature, drop_mask=True)
        z = torch.where(still_masked, sample_categorical(probs), z)
    return z


def _infill_d3pm(model, fwd, tokens, mask_positions, num_steps, temperature):
    require(num_steps <= fwd.num_timesteps, "D3PM infill num_steps must be <= fwd.num_timesteps")
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    mask_positions = mask_positions.to(device)
    B = tokens.size(0)
    mask_id = fwd.mask_token_id

    z = tokens.clone()
    z[mask_positions] = mask_id
    timesteps = d3pm_reverse_timesteps(fwd, num_steps, device)
    for i in range(num_steps):
        masked = (z == mask_id) & mask_positions
        if not masked.any():
            break

        t_now, t_prev = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(B))
        log_probs = absorbing_posterior_log_probs(logits, z, t_now.expand(B), t_prev.expand(B), fwd, mask_id)
        predictions = sample_logits(log_probs, temperature)
        z = torch.where(masked, predictions, z)

    still_masked = (z == mask_id) & mask_positions
    require(not still_masked.any(), "D3PM infill reverse chain left masked tokens at t=0")
    return z
