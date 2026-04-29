import torch

from minilab.checks import require


def sedd_absorbing_step_probs(log_scores, z, dsigma, mask_id, temperature, drop_mask=False):
    require(temperature >= 0, "temperature must be >= 0")
    log_scores = log_scores.scatter(
        -1,
        z.unsqueeze(-1),
        torch.zeros_like(log_scores[..., :1]),
    )
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


def sample_categorical(probs):
    flat = probs.reshape(-1, probs.size(-1))
    row_sum = flat.sum(dim=-1, keepdim=True)
    require((row_sum > 0).all(), "categorical sampler received a zero-probability row")
    normalized = flat / row_sum
    return torch.multinomial(normalized, 1).view(probs.shape[:-1])


def sample_logits(logits, temperature):
    if temperature == 0:
        return logits.argmax(-1)
    probs = (logits / temperature).softmax(-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(logits.shape[:-1])


def sample_clean_logits(logits, mask_id, temperature):
    logits = logits.clone()
    logits[:, :, mask_id] = float("-inf")
    return sample_logits(logits, temperature)


def d3pm_reverse_timesteps(fwd, num_steps, device):
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
