import torch

from minilab.registry import register_sampler


@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens=100, temperature=1.0, top_k=50, top_p=1.0, repetition_penalty=1.0):
    """Autoregressive sampling. temperature=0 for greedy. repetition_penalty > 1.0 penalizes seen tokens."""
    model.eval()
    device = next(model.parameters()).device
    ids = prompt_ids.to(device)
    max_ctx = model.config.max_seq_len

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

    return ids


@register_sampler("ancestral")
@torch.no_grad()
def sample_diffusion(model, fwd, batch_size, seq_len, num_steps=256, temperature=1.0):
    """Standard reverse process: predict clean tokens, unmask probabilistically."""
    device = next(model.parameters()).device
    model.eval()

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
    device = next(model.parameters()).device
    model.eval()

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


@torch.no_grad()
def infill(model, fwd, tokens, mask_positions, num_steps=128, temperature=1.0):
    """Fill masked positions while keeping context fixed. Unique to diffusion models."""
    device = tokens.device
    model.eval()
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
