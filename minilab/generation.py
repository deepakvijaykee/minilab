import torch

from minilab.base import BaseModel, unwrap_model
from minilab.checks import require, require_finite_number, require_integer
from minilab.config import BaseConfig


def _require_eval_model(model, context):
    require(not unwrap_model(model).training, f"{context} expects model.eval() at the call boundary")


def _require_base_model(model, context):
    model = unwrap_model(model)
    require(isinstance(model, BaseModel), f"{context} requires a BaseModel")
    return model


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
