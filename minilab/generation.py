import math

import torch

from minilab.registry import register_sampler


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0, stop_texts=None, tokenizer=None):
    """Autoregressive sampling. temperature=0 for greedy.
    stop_texts: list of strings that trigger early stopping (batch_size=1 only, requires tokenizer)."""
    assert not model.training, "generate expects model.eval() at the call boundary"
    if stop_texts:
        assert tokenizer is not None, "stop_texts requires tokenizer"
        assert prompt_ids.size(0) == 1, "stop_texts only supported for batch_size=1"
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    max_ctx = model.config.max_seq_len
    prompt_len = ids.size(1)

    for _ in range(max_new_tokens):
        logits, _ = model(ids[:, -max_ctx:])
        logits = logits[:, -1]

        if repetition_penalty != 1.0:
            for b in range(ids.size(0)):
                seen = ids[b].unique()
                logits[b, seen] /= repetition_penalty

        if temperature == 0:
            next_id = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if 0 < top_k < logits.size(-1):
                cutoff = logits.topk(top_k).values[:, -1:]
                logits[logits < cutoff] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(descending=True)
                cum_probs = sorted_logits.softmax(-1).cumsum(-1)
                remove = cum_probs > top_p
                remove[..., 0] = False
                sorted_logits[remove] = float("-inf")
                logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
            next_id = torch.multinomial(logits.softmax(-1), 1)

        ids = torch.cat([ids, next_id], dim=1)

        if stop_texts:
            tail = tokenizer.decode(ids[0, prompt_len:].tolist())
            if any(s in tail for s in stop_texts):
                break

    return ids


@register_sampler("ancestral")
@torch.no_grad()
def sample_diffusion(model, fwd, batch_size, seq_len, num_steps=256, temperature=1.0):
    """Standard reverse process: predict clean tokens, unmask probabilistically."""
    assert not model.training, "sample_diffusion expects model.eval() at the call boundary"
    device = next(model.parameters()).device

    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(batch_size))
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
        z = torch.where(still_masked, logits.argmax(-1), z)

    return z


@register_sampler("ddpm_cache")
@torch.no_grad()
def sample_diffusion_cached(model, fwd, batch_size, seq_len, num_steps=256, temperature=1.0, cache_interval=4):
    """Reuse cached predictions for cache_interval steps. ~cache_interval x fewer forward passes."""
    assert not model.training, "sample_diffusion_cached expects model.eval() at the call boundary"
    device = next(model.parameters()).device

    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    logits = None
    for i in range(num_steps):
        t_now, t_next = timesteps[i], timesteps[i + 1]

        if i % cache_interval == 0:
            logits = model(z, t_now.expand(batch_size))
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
        z = torch.where(still_masked, logits.argmax(-1), z)

    return z


@register_sampler("sedd_analytical")
@torch.no_grad()
def sample_sedd(model, fwd, batch_size, seq_len, num_steps=256, temperature=1.0):
    """SEDD analytical sampler for absorbing noise (Lou et al. 2024, Algorithm 4).
    Masked positions unmask with the schedule-implied probability
    (alpha_next - alpha_now) / (1 - alpha_now); the token is drawn from
    softmax(scores) over non-mask vocab (scores are log p(x_0=j|z_t) / p(x_0=mask|z_t))."""
    assert not model.training, "sample_sedd expects model.eval() at the call boundary"
    device = next(model.parameters()).device

    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        masked = z == mask_id
        if not masked.any():
            break

        t_now, t_next = timesteps[i], timesteps[i + 1]
        scores = model(z, t_now.expand(batch_size))
        if temperature > 0:
            scores = scores / temperature
        scores[:, :, mask_id] = float("-inf")

        alpha_now = fwd.get_alpha(t_now.unsqueeze(0)).item()
        alpha_next = fwd.get_alpha(t_next.unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next

        predictions = torch.multinomial(scores.softmax(-1).view(-1, scores.size(-1)), 1).view(batch_size, seq_len)
        unmask = torch.rand(batch_size, seq_len, device=device) < unmask_prob
        z = torch.where(masked & unmask, predictions, z)

    still_masked = z == mask_id
    if still_masked.any():
        scores = model(z, torch.zeros(batch_size, device=device))
        scores[:, :, mask_id] = float("-inf")
        z = torch.where(still_masked, scores.argmax(-1), z)

    return z


@register_sampler("d3pm_ancestral")
@torch.no_grad()
def sample_d3pm(model, fwd, batch_size, seq_len, num_steps=256, temperature=1.0):
    """Absorbing-chain ancestral sampler for D3PM.

    The D3PM head predicts a distribution over the full vocabulary (including
    [MASK]) because training matches p_θ(·|z_t) against a two-atom posterior on
    {x_0, mask}. For reverse sampling we want x_0, so we mask [MASK] before
    softmax and then unmask with the absorbing-kernel schedule probability
    (alpha_next - alpha_now) / (1 - alpha_now). This derivation is closed-form
    for the absorbing process, so skip-step sampling (num_steps < num_timesteps)
    remains faithful to the trained chain."""
    assert not model.training, "sample_d3pm expects model.eval() at the call boundary"
    device = next(model.parameters()).device
    mask_id = fwd.mask_token_id
    z = torch.full((batch_size, seq_len), mask_id, device=device, dtype=torch.long)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        masked = z == mask_id
        if not masked.any():
            break

        t_now, t_next = timesteps[i], timesteps[i + 1]
        logits = model(z, t_now.expand(batch_size))
        logits[:, :, mask_id] = float("-inf")
        if temperature == 0:
            predictions = logits.argmax(-1)
        else:
            logits = logits / temperature
            predictions = torch.multinomial(logits.softmax(-1).view(-1, logits.size(-1)), 1).view(batch_size, seq_len)

        alpha_now = fwd.get_alpha(t_now.unsqueeze(0)).item()
        alpha_next = fwd.get_alpha(t_next.unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next

        unmask = torch.rand(batch_size, seq_len, device=device) < unmask_prob
        z = torch.where(masked & unmask, predictions, z)

    still_masked = z == mask_id
    if still_masked.any():
        logits = model(z, torch.zeros(batch_size, device=device))
        logits[:, :, mask_id] = float("-inf")
        z = torch.where(still_masked, logits.argmax(-1), z)
    return z


@torch.no_grad()
def infill(model, fwd, tokens, mask_positions, num_steps=128, temperature=1.0):
    """Fill masked positions while keeping context fixed. Unique to diffusion models."""
    assert not model.training, "infill expects model.eval() at the call boundary"
    device = tokens.device
    B = tokens.size(0)

    z = tokens.clone()
    z[mask_positions] = fwd.mask_token_id
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in range(num_steps):
        t_now = timesteps[i]
        logits = model(z, t_now.expand(B))
        if temperature > 0:
            logits = logits / temperature
        predictions = torch.multinomial(logits.softmax(-1).view(-1, logits.size(-1)), 1).view_as(tokens)

        alpha_now = fwd.get_alpha(t_now.unsqueeze(0)).item()
        alpha_next = fwd.get_alpha(timesteps[i + 1].unsqueeze(0)).item()
        unmask_prob = (alpha_next - alpha_now) / (1.0 - alpha_now) if alpha_now < 1.0 else alpha_next

        unmask = torch.rand_like(tokens, dtype=torch.float) < unmask_prob
        should_update = mask_positions & (z == fwd.mask_token_id) & unmask
        z = torch.where(should_update, predictions, z)

    still_masked = (z == fwd.mask_token_id) & mask_positions
    if still_masked.any():
        logits = model(z, torch.zeros(B, device=device))
        z = torch.where(still_masked, logits.argmax(-1), z)

    return z
