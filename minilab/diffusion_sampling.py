import torch
import torch.nn.functional as F

from minilab.checks import require


def sedd_absorbing_step_probs(log_scores, z, dsigma, mask_id, temperature, drop_mask=False):
    require(temperature >= 0, "temperature must be >= 0")
    require(log_scores.ndim == 3, "SEDD log_scores must have shape (batch, seq, vocab)")
    require(z.shape == log_scores.shape[:2], "SEDD current tokens must match log_scores batch and sequence shape")
    log_scores = log_scores.scatter(
        -1,
        z.unsqueeze(-1),
        torch.zeros_like(log_scores[..., :1]),
    )
    if temperature == 0:
        scaled_log_scores = log_scores.double()
    else:
        scaled_log_scores = (log_scores / temperature).double()
    dsigma = _sedd_batch_step(dsigma, z.size(0), scaled_log_scores.device, scaled_log_scores.dtype)
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


def _sedd_batch_step(dsigma, batch_size, device, dtype):
    dsigma = torch.as_tensor(dsigma, device=device, dtype=dtype)
    if dsigma.ndim == 0 or dsigma.numel() == 1:
        return dsigma.reshape(1, 1).expand(batch_size, 1)
    if dsigma.shape == (batch_size,):
        return dsigma.view(batch_size, 1)
    if dsigma.shape == (batch_size, 1):
        return dsigma
    raise ValueError("SEDD sigma step must be scalar or have shape (batch,) / (batch, 1)")


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
    log_decay = (-dsigma).expand_as(z)
    log_unmask = torch.log1p(-(-dsigma).exp()).expand_as(z)
    log_transition.scatter_(-1, z.unsqueeze(-1), log_decay.unsqueeze(-1))
    masked = z == mask_id
    if masked.any():
        masked_rows = log_unmask[masked].unsqueeze(-1).expand(-1, vocab_size).clone()
        masked_rows[:, mask_id] = 0.0
        log_transition[masked] = masked_rows
    return log_transition


def sample_categorical(probs):
    require(torch.isfinite(probs).all(), "categorical sampler received non-finite probabilities")
    require((probs >= 0).all(), "categorical sampler received negative probabilities")
    flat = probs.reshape(-1, probs.size(-1))
    row_sum = flat.sum(dim=-1, keepdim=True)
    require((row_sum > 0).all(), "categorical sampler received a zero-probability row")
    normalized = flat / row_sum
    return torch.multinomial(normalized, 1).view(probs.shape[:-1])


def sample_logits(logits, temperature):
    require(temperature >= 0, "temperature must be >= 0")
    valid_or_masked = torch.isfinite(logits) | torch.isneginf(logits)
    require(valid_or_masked.all(), "logits sampler received non-finite logits outside masked support")
    require(torch.isfinite(logits).any(dim=-1).all(), "logits sampler received a row with no valid support")
    if temperature == 0:
        return logits.argmax(-1)
    probs = (logits / temperature).softmax(-1)
    return sample_categorical(probs)


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


def llada_add_gumbel_noise(logits, temperature):
    """LLaDA reference sampler's low-precision Gumbel-max transform."""
    require(temperature >= 0, "temperature must be >= 0")
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def llada_transfer_counts(mask_index, steps):
    """Number of masked tokens to transfer per reverse step.

    Matches the LLaDA reference schedule: each row's masked-token count is split
    as evenly as possible across the block's reverse steps, with the remainder
    assigned to the earliest steps.
    """
    require(mask_index.dtype == torch.bool, "mask_index must be bool")
    require(steps > 0, "steps must be > 0")
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    counts = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.long) + base
    for i in range(mask_num.size(0)):
        counts[i, : int(remainder[i].item())] += 1
    return counts


def _select_transfer_positions(
    confidence,
    eligible,
    counts,
    *,
    stochastic=False,
    temperature=0.0,
):
    transfer_index = torch.zeros_like(eligible, dtype=torch.bool)
    for row in range(eligible.size(0)):
        k = int(counts[row].item())
        if k == 0:
            continue
        eligible_row = eligible[row]
        require(k <= int(eligible_row.sum().item()), "num_transfer_tokens exceeds masked token count")
        row_confidence = confidence[row].masked_fill(~eligible_row, float("-inf"))
        require(torch.isfinite(row_confidence[eligible_row]).all(), (
            "confidence scores must be finite for eligible transfer tokens"
        ))
        if stochastic and temperature > 0:
            probs = torch.softmax(row_confidence / temperature, dim=0)
            select_index = torch.multinomial(probs, num_samples=k, replacement=False)
        else:
            _, select_index = torch.topk(row_confidence, k=k)
        transfer_index[row, select_index] = True
    return transfer_index


def dream_transfer_count(mask_index, t_now, t_next, final_step=False):
    """Number of Dream masked tokens to transfer for one reverse step, per row."""
    require(mask_index.dtype == torch.bool, "mask_index must be bool")
    require(mask_index.dim() == 2, "mask_index must have shape (batch, seq)")
    remaining = mask_index.sum(dim=1).long()
    if final_step:
        return remaining
    t_now = _dream_row_times(t_now, mask_index.size(0), mask_index.device, "t_now")
    t_next = _dream_row_times(t_next, mask_index.size(0), mask_index.device, "t_next")
    require(((0 <= t_next) & (t_next < t_now) & (t_now <= 1)).all(), (
        "Dream timesteps must satisfy 0 <= t_next < t_now <= 1"
    ))
    return (remaining.to(t_now.dtype) * (1.0 - t_next / t_now)).long()


def _dream_row_times(value, batch_size, device, name):
    if torch.is_tensor(value):
        value = value.to(device=device, dtype=torch.float64)
        if value.numel() == 1:
            return value.reshape(1).expand(batch_size)
        require(value.shape == (batch_size,), f"Dream {name} must be scalar or shape (batch,)")
        return value
    return torch.full((batch_size,), float(value), device=device, dtype=torch.float64)


def llada_remask_step(
    logits,
    x,
    mask_id,
    num_transfer_tokens,
    *,
    prompt_index=None,
    block_end=None,
    temperature=0.0,
    remasking="low_confidence",
    eos_token_id=None,
    eot_token_id=None,
):
    """One LLaDA reverse step with low-confidence or random remasking."""
    require(logits.shape[:2] == x.shape, "logits and x must share batch/sequence shape")
    require(num_transfer_tokens.shape == (x.size(0),), "num_transfer_tokens must have shape (batch,)")
    require(remasking in {"low_confidence", "random"}, "remasking must be 'low_confidence' or 'random'")
    if eos_token_id is not None:
        logits = logits.clone()
        logits[:, :, eos_token_id] = float("-inf")
    if eot_token_id is not None:
        logits = logits.clone()
        logits[:, :, eot_token_id] = float("-inf")
    noisy_logits = llada_add_gumbel_noise(logits, temperature)
    x0 = noisy_logits.argmax(dim=-1)

    if remasking == "low_confidence":
        probs = F.softmax(logits, dim=-1)
        confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        confidence = torch.rand(x.shape, device=x.device)
    mask_index = x == mask_id
    if prompt_index is not None:
        require(prompt_index.shape == x.shape, "prompt_index must match x")
        mask_index = mask_index & (~prompt_index)
    if block_end is not None:
        require(0 < block_end <= x.size(1), "block_end must be in (0, seq_len]")
        inside_block = torch.arange(x.size(1), device=x.device) < block_end
        mask_index = mask_index & inside_block.unsqueeze(0)
        confidence[:, block_end:] = float("-inf")
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))
    transfer_index = _select_transfer_positions(confidence, mask_index, num_transfer_tokens)
    return torch.where(transfer_index, x0, x), transfer_index, confidence


def dream_top_p_logits(logits, top_p):
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    if top_p == 1:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probs > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, remove)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def dream_top_k_logits(logits, top_k):
    require(top_k >= 0, "top_k must be >= 0")
    if top_k == 0 or top_k >= logits.size(-1):
        return logits
    cutoff = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)


def dream_sample_tokens(
    logits,
    *,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    alg="origin",
):
    """Dream token proposal and confidence policies.

    `alg` matches the Dream generator names: `origin` assigns random transfer
    order, `maskgit_plus` uses sampled-token probability, `topk_margin` uses
    top1-top2 margin, and `entropy` uses negative entropy.
    """
    require(temperature >= 0, "temperature must be >= 0")
    require(0 < top_p <= 1, "top_p must be in (0, 1]")
    require(top_k >= 0, "top_k must be >= 0")
    require(alg in {"origin", "maskgit_plus", "topk_margin", "entropy"}, f"unknown Dream alg: {alg}")
    logits = dream_top_k_logits(dream_top_p_logits(logits, top_p), top_k)
    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        x0 = torch.multinomial(probs.reshape(-1, probs.size(-1)), 1).view(logits.shape[:-1])
        confidence = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        probs = F.softmax(logits, dim=-1)
        confidence, x0 = probs.max(dim=-1)

    if alg == "origin":
        confidence = torch.rand(logits.shape[:-1], device=logits.device, dtype=probs.dtype)
    elif alg == "topk_margin":
        top2 = torch.topk(probs, k=2, dim=-1).values
        confidence = top2[..., 0] - top2[..., 1]
    elif alg == "entropy":
        confidence = torch.sum(probs * torch.log(probs.clamp_min(1e-10)), dim=-1)

    return confidence, x0


def _dream_transfer_counts(num_transfer_tokens, batch_size, device):
    if torch.is_tensor(num_transfer_tokens):
        num_transfer_tokens = num_transfer_tokens.to(device=device, dtype=torch.long)
        if num_transfer_tokens.numel() == 1:
            counts = num_transfer_tokens.reshape(1).expand(batch_size)
        else:
            require(num_transfer_tokens.shape == (batch_size,), (
                "Dream num_transfer_tokens must be a scalar count or shape (batch,)"
            ))
            counts = num_transfer_tokens
    else:
        counts = torch.full((batch_size,), int(num_transfer_tokens), device=device, dtype=torch.long)
    require((counts >= 0).all(), "Dream num_transfer_tokens must be >= 0")
    return counts


def dream_remask_step(
    logits,
    x,
    mask_id,
    num_transfer_tokens,
    *,
    prompt_index=None,
    block_end=None,
    temperature=0.0,
    top_p=1.0,
    top_k=0,
    alg="origin",
    alg_temp=0.0,
):
    """One Dream reverse step over currently masked tokens."""
    require(logits.shape[:2] == x.shape, "logits and x must share batch/sequence shape")
    require(alg_temp >= 0, "alg_temp must be >= 0")
    transfer_counts = _dream_transfer_counts(num_transfer_tokens, x.size(0), x.device)
    logits = logits.clone()
    logits[:, :, mask_id] = float("-inf")
    confidence, x0 = dream_sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
    )
    mask_index = x == mask_id
    if prompt_index is not None:
        require(prompt_index.shape == x.shape, "prompt_index must match x")
        mask_index = mask_index & (~prompt_index)
    if block_end is not None:
        require(0 < block_end <= x.size(1), "block_end must be in (0, seq_len]")
        inside_block = torch.arange(x.size(1), device=x.device) < block_end
        mask_index = mask_index & inside_block.unsqueeze(0)
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))
    transfer_index = _select_transfer_positions(
        confidence,
        mask_index,
        transfer_counts,
        stochastic=alg != "origin",
        temperature=alg_temp,
    )
    return torch.where(transfer_index, x0, x), transfer_index, confidence
