import torch

from minilab.base import BaseModel, unwrap_model
from minilab.checks import require
from minilab.models.diffusion_base import DiffusionModelConfig, validate_infill_tokens
from minilab.models.d3pm import absorbing_posterior_log_probs
from minilab.registry import register_sampler


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, stop_texts=None, tokenizer=None):
    """Autoregressive sampling. temperature=0 for greedy.
    stop_texts: list of strings that trigger early stopping (batch_size=1 only, requires tokenizer)."""
    _require_eval_model(model, "generate")
    require(prompt_ids.dim() == 2, "prompt_ids must have shape (batch, seq)")
    require(prompt_ids.size(1) > 0, "generate requires a non-empty prompt")
    require(max_new_tokens >= 0, "max_new_tokens must be >= 0")
    require(temperature >= 0, "temperature must be >= 0")
    require(top_k >= 0, "top_k must be >= 0")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(repetition_penalty > 0, "repetition_penalty must be > 0")
    if stop_texts:
        require(tokenizer is not None, "stop_texts requires tokenizer")
        require(prompt_ids.size(0) == 1, "stop_texts only supported for batch_size=1")
    model_core = _require_base_model(model, "generate")
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    max_ctx = model_core.config.max_seq_len
    prompt_len = ids.size(1)

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -max_ctx:])
        logits = logits[:, -1]

        if repetition_penalty != 1.0:
            for b in range(ids.size(0)):
                seen = ids[b].unique()
                seen_logits = logits[b, seen]
                logits[b, seen] = torch.where(
                    seen_logits < 0,
                    seen_logits * repetition_penalty,
                    seen_logits / repetition_penalty,
                )

        if temperature == 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if 0 < top_k < logits.size(-1):
                cutoff = logits.topk(top_k).values[:, -1:]
                logits[logits < cutoff] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                sorted_probs = sorted_logits.softmax(-1)
                remove = sorted_probs.cumsum(-1) - sorted_probs > top_p
                sorted_logits[remove] = float("-inf")
                filtered_logits = torch.full_like(logits, float("-inf"))
                logits = filtered_logits.scatter(-1, sorted_idx, sorted_logits)
            next_id = torch.multinomial(logits.softmax(-1), 1)

        ids = torch.cat([ids, next_id], dim=1)

        if stop_texts:
            tail = tokenizer.decode(ids[0, prompt_len:].tolist())
            if any(s in tail for s in stop_texts):
                break

    return ids


@register_sampler("ancestral")
@torch.no_grad()
def sample_diffusion(model, fwd, batch_size, seq_len, num_steps=None, temperature=1.0):
    """Standard absorbing-mask reverse process: predict clean tokens, then unmask."""
    _require_eval_model(model, "sample_diffusion")
    _require_sampler_contract(model, fwd, "clean_logits", "sample_diffusion")
    _require_terminal_mask_prior(fwd, "sample_diffusion")
    if num_steps is None:
        num_steps = min(256, fwd.num_timesteps)
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(0 < num_steps <= fwd.num_timesteps, "num_steps must be in [1, fwd.num_timesteps]")
    require(temperature >= 0, "temperature must be >= 0")
    device = next(model.parameters()).device

    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(batch_size))
        logits[:, :, mask_id] = float("-inf")
        if temperature == 0:
            predictions = logits.argmax(-1)
        else:
            if temperature > 0:
                logits = logits / temperature
            predictions = torch.multinomial(logits.softmax(-1).view(-1, logits.size(-1)), 1).view(batch_size, seq_len)

        alpha_now = fwd.get_alpha(t_now.unsqueeze(0)).item()
        alpha_next = fwd.get_alpha(t_next.unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next

        unmask = torch.rand(batch_size, seq_len, device=device) < unmask_prob
        z = torch.where((z == mask_id) & unmask, predictions, z)

    still_masked = z == mask_id
    if still_masked.any():
        logits = model(z, torch.zeros(batch_size, device=device))
        logits[:, :, mask_id] = float("-inf")
        z = torch.where(still_masked, logits.argmax(-1), z)

    return z


@register_sampler("ddpm_cache")
@torch.no_grad()
def sample_diffusion_cached(model, fwd, batch_size, seq_len, num_steps=None, temperature=1.0, cache_interval=4):
    """Reuse clean-token predictions for cache_interval steps. ~cache_interval x fewer forward passes."""
    _require_eval_model(model, "sample_diffusion_cached")
    _require_sampler_contract(model, fwd, "clean_logits", "sample_diffusion_cached")
    _require_terminal_mask_prior(fwd, "sample_diffusion_cached")
    if num_steps is None:
        num_steps = min(256, fwd.num_timesteps)
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(0 < num_steps <= fwd.num_timesteps, "num_steps must be in [1, fwd.num_timesteps]")
    require(temperature >= 0, "temperature must be >= 0")
    require(cache_interval > 0, "cache_interval must be > 0")
    device = next(model.parameters()).device

    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    logits = None
    for i in range(num_steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]

        if i % cache_interval == 0:
            logits = model(z, t_now.expand(batch_size))
            logits[:, :, mask_id] = float("-inf")
            if temperature > 0:
                logits = logits / temperature

        if temperature == 0:
            predictions = logits.argmax(-1)
        else:
            predictions = torch.multinomial(logits.softmax(-1).view(-1, logits.size(-1)), 1).view(batch_size, seq_len)

        alpha_now = fwd.get_alpha(t_now.unsqueeze(0)).item()
        alpha_next = fwd.get_alpha(t_next.unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next

        unmask = torch.rand(batch_size, seq_len, device=device) < unmask_prob
        z = torch.where((z == mask_id) & unmask, predictions, z)

    still_masked = z == mask_id
    if still_masked.any():
        logits = model(z, torch.zeros(batch_size, device=device))
        logits[:, :, mask_id] = float("-inf")
        z = torch.where(still_masked, logits.argmax(-1), z)

    return z


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
        probs = _sedd_absorbing_step_probs(log_scores, z, sigma_now - sigma_next, mask_id, temperature)
        z = _sample_categorical(probs)

    still_masked = z == mask_id
    if still_masked.any():
        t_eps = timesteps[-1]
        log_scores = model(z, t_eps.expand(batch_size))
        sigma = fwd.get_sigma(t_eps.unsqueeze(0)).to(device)
        probs = _sedd_absorbing_step_probs(log_scores, z, sigma, mask_id, temperature, drop_mask=True)
        z = torch.where(still_masked, _sample_categorical(probs), z)

    return z


def _sedd_absorbing_step_probs(log_scores, z, dsigma, mask_id, temperature, drop_mask=False):
    require(temperature >= 0, "temperature must be >= 0")
    if temperature == 0:
        scaled_log_scores = log_scores.double()
    else:
        scaled_log_scores = (log_scores / temperature).double()
    dsigma = dsigma.to(scaled_log_scores.device, dtype=scaled_log_scores.dtype).view(1, 1)
    require((dsigma >= 0).all(), "SEDD sigma step must be non-negative")
    log_weights = (
        _sedd_absorbing_log_staggered_scores(scaled_log_scores, dsigma, mask_id)
        + _absorbing_transposed_transition_log_probs(z, dsigma, mask_id, log_scores.size(-1))
    )
    if drop_mask:
        log_weights[:, :, mask_id] = float("-inf")
    has_support = torch.isfinite(log_weights).any(dim=-1, keepdim=True)
    require(has_support.all(), "SEDD categorical step has no valid support")
    if temperature == 0:
        greedy = torch.zeros_like(log_weights)
        greedy.scatter_(-1, log_weights.argmax(dim=-1, keepdim=True), 1.0)
        return greedy
    log_probs = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
    return log_probs.exp()


def _sedd_absorbing_log_staggered_scores(log_scores, dsigma, mask_id):
    log_staggered = log_scores + dsigma[..., None]
    mask_log_a = log_scores[:, :, mask_id] + dsigma
    mask_log_b = torch.logsumexp(log_scores, dim=-1) + torch.expm1(dsigma).log()
    delta = mask_log_b - mask_log_a
    positive = delta < 0
    safe_delta = torch.where(positive, delta, torch.full_like(delta, float("-inf")))
    mask_log_staggered = torch.where(
        positive,
        mask_log_a + torch.log1p(-safe_delta.exp()),
        torch.full_like(mask_log_a, float("-inf")),
    )
    log_staggered[:, :, mask_id] = mask_log_staggered
    return log_staggered


def _absorbing_transposed_transition_log_probs(z, dsigma, mask_id, vocab_size):
    log_transition = torch.full(
        (*z.shape, vocab_size),
        float("-inf"),
        device=z.device,
        dtype=dsigma.dtype,
    )
    log_decay = -dsigma.squeeze()
    log_unmask = torch.log1p(-(-dsigma).exp()).squeeze()
    masked = z == mask_id
    observed = ~masked
    if observed.any():
        observed_rows = log_transition[observed]
        observed_rows.scatter_(1, z[observed].unsqueeze(-1), log_decay.expand(observed_rows.size(0), 1))
        log_transition[observed] = observed_rows
    if masked.any():
        masked_rows = log_transition[masked]
        masked_rows[:] = log_unmask
        masked_rows[:, mask_id] = 0.0
        log_transition[masked] = masked_rows
    return log_transition


def _sample_categorical(probs):
    flat = probs.reshape(-1, probs.size(-1))
    row_sum = flat.sum(dim=-1, keepdim=True)
    require((row_sum > 0).all(), "categorical sampler received a zero-probability row")
    normalized = flat / row_sum
    return torch.multinomial(normalized, 1).view(probs.shape[:-1])


def _sample_logits(logits, temperature):
    if temperature == 0:
        return logits.argmax(-1)
    probs = (logits / temperature).softmax(-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])


def _sample_clean_logits(logits, mask_id, temperature):
    logits = logits.clone()
    logits[:, :, mask_id] = float("-inf")
    return _sample_logits(logits, temperature)


def _d3pm_reverse_timesteps(fwd, num_steps, device):
    step = torch.arange(num_steps + 1, device=device, dtype=torch.long)
    offset = torch.div(
        step * fwd.num_timesteps + num_steps // 2,
        num_steps,
        rounding_mode="floor",
    )
    idx = fwd.num_timesteps - offset
    require((idx[:-1] > idx[1:]).all(), (
        "D3PM reverse timesteps must be strictly descending grid indices"
    ))
    return idx.float() / fwd.num_timesteps


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
    require(0 < num_steps <= fwd.num_timesteps, (
        "D3PM num_steps must be in [1, fwd.num_timesteps]"
    ))
    require(batch_size > 0 and seq_len > 0, "batch_size and seq_len must be > 0")
    require(temperature >= 0, "temperature must be >= 0")
    device = next(model.parameters()).device
    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = _d3pm_reverse_timesteps(fwd, num_steps, device)

    for i in range(num_steps):
        masked = z == mask_id
        if not masked.any():
            break

        t_now, t_prev = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(batch_size))
        log_probs = absorbing_posterior_log_probs(
            logits, z, t_now.expand(batch_size), t_prev.expand(batch_size), fwd, mask_id
        )
        predictions = _sample_logits(log_probs, temperature)
        z = torch.where(masked, predictions, z)

    still_masked = z == mask_id
    if still_masked.any():
        logits = model(z, torch.zeros(batch_size, device=device))
        z = torch.where(still_masked, _sample_clean_logits(logits, mask_id, temperature=0), z)
    return z


@torch.no_grad()
def infill(model, fwd, tokens, mask_positions, num_steps=None, temperature=1.0):
    """Fill masked positions while keeping context fixed. Unique to diffusion models."""
    _require_eval_model(model, "infill")
    require(temperature >= 0, "temperature must be >= 0")
    require(tokens.shape == mask_positions.shape, "tokens and mask_positions must have the same shape")
    require(mask_positions.dtype == torch.bool, "mask_positions must be a bool tensor")
    parameterization = _reverse_parameterization(model)
    if num_steps is None:
        num_steps = min(128, fwd.num_timesteps)
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
    require(not model.training, f"{context} expects model.eval() at the call boundary")


def _require_sampler_contract(model, fwd, expected_parameterization, context):
    actual = _reverse_parameterization(model, context)
    require(
        actual == expected_parameterization,
        f"{context} requires reverse_parameterization={expected_parameterization!r}, got {actual!r}",
    )
    _require_absorbing_forward_process(model, fwd, context)


def _require_terminal_mask_prior(fwd, context):
    require(fwd.has_terminal_mask_prior(), (
        f"{context} starts reverse sampling from all [MASK] tokens and therefore "
        "requires alpha[-1] = 0 so q(x_T | x_0) matches that terminal prior"
    ))


def _require_absorbing_forward_process(model, fwd, context):
    require(fwd.process_type == "absorbing", f"{context} currently supports only the absorbing forward process")
    model = _require_diffusion_model(model, context)
    model_config = model.config
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
        predictions = _sample_clean_logits(model(z, t_now.expand(B)), mask_id, temperature)

        alpha_now = fwd.get_alpha(t_now.unsqueeze(0)).item()
        alpha_next = fwd.get_alpha(timesteps[i + 1].unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next

        unmask = torch.rand_like(tokens, dtype=torch.float) < unmask_prob
        z = torch.where(masked & unmask, predictions, z)

    still_masked = (z == mask_id) & mask_positions
    if still_masked.any():
        logits = model(z, torch.zeros(B, device=device))
        z = torch.where(still_masked, _sample_clean_logits(logits, mask_id, temperature=0), z)

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
        probs = _sedd_absorbing_step_probs(log_scores, z, sigma_now - sigma_next, mask_id, temperature)
        z = torch.where(mask_positions, _sample_categorical(probs), tokens)

    still_masked = (z == mask_id) & mask_positions
    if still_masked.any():
        t_eps = timesteps[-1]
        log_scores = model(z, t_eps.expand(B))
        sigma = fwd.get_sigma(t_eps.unsqueeze(0)).to(device)
        probs = _sedd_absorbing_step_probs(log_scores, z, sigma, mask_id, temperature, drop_mask=True)
        z = torch.where(still_masked, _sample_categorical(probs), z)
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
    timesteps = _d3pm_reverse_timesteps(fwd, num_steps, device)
    for i in range(num_steps):
        masked = (z == mask_id) & mask_positions
        if not masked.any():
            break

        t_now, t_prev = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(B))
        log_probs = absorbing_posterior_log_probs(logits, z, t_now.expand(B), t_prev.expand(B), fwd, mask_id)
        predictions = _sample_logits(log_probs, temperature)
        z = torch.where(masked, predictions, z)

    still_masked = (z == mask_id) & mask_positions
    if still_masked.any():
        logits = model(z, torch.zeros(B, device=device))
        z = torch.where(still_masked, _sample_clean_logits(logits, mask_id, temperature=0), z)
    return z
