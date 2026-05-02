import torch
import torch.nn.functional as F

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
        confidence[:, block_end:] = float("-inf")
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))
    transfer_index = torch.zeros_like(x, dtype=torch.bool)
    for row in range(x.size(0)):
        k = int(num_transfer_tokens[row].item())
        if k == 0:
            continue
        require(k <= int(mask_index[row].sum().item()), "num_transfer_tokens exceeds masked token count")
        _, select_index = torch.topk(confidence[row], k=k)
        transfer_index[row, select_index] = True
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
    alg_temp=0.0,
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
    require(alg_temp >= 0, "alg_temp must be >= 0")
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

    if alg != "origin" and alg_temp > 0:
        u = torch.rand_like(confidence).clamp_(1e-6, 1 - 1e-6)
        confidence = confidence + alg_temp * (-torch.log(-torch.log(u)))
    return confidence, x0


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
    require(num_transfer_tokens.shape == (x.size(0),), "num_transfer_tokens must have shape (batch,)")
    logits = logits.clone()
    logits[:, :, mask_id] = float("-inf")
    confidence, x0 = dream_sample_tokens(
        logits,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=alg,
        alg_temp=alg_temp,
    )
    mask_index = x == mask_id
    if prompt_index is not None:
        require(prompt_index.shape == x.shape, "prompt_index must match x")
        mask_index = mask_index & (~prompt_index)
    if block_end is not None:
        require(0 < block_end <= x.size(1), "block_end must be in (0, seq_len]")
        confidence[:, block_end:] = float("-inf")
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, confidence, torch.full_like(confidence, float("-inf")))
    transfer_index = torch.zeros_like(x, dtype=torch.bool)
    for row in range(x.size(0)):
        k = int(num_transfer_tokens[row].item())
        if k == 0:
            continue
        require(k <= int(mask_index[row].sum().item()), "num_transfer_tokens exceeds masked token count")
        _, select_index = torch.topk(confidence[row], k=k)
        transfer_index[row, select_index] = True
    return torch.where(transfer_index, x0, x), transfer_index, confidence
