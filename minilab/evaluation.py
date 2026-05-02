import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from minilab.checks import require


@torch.no_grad()
def perplexity(model, dataloader):
    total_nll, total_tokens = _lm_nll_sum(model, dataloader, "perplexity")
    return math.exp(total_nll / total_tokens)


@torch.no_grad()
def bits_per_character(model, dataloader):
    total_nll, total_chars = _lm_nll_sum(model, dataloader, "bits_per_character")
    return total_nll / (math.log(2.0) * total_chars)


def _lm_nll_sum(model, dataloader, context):
    require(not model.training, f"{context} expects model.eval() at the call boundary")
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(ids)
        n = (labels != -100).sum().item()
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        total_tokens += n
    require(total_tokens > 0, f"{context} received no valid target tokens")
    return total_nll, total_tokens


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt, continuation):
    """Log-probability of `continuation` under an autoregressive model."""
    require(not model.training, "sequence_logprob expects model.eval() at the call boundary")
    prompt_ids = tokenizer.encode(prompt)
    continuation_ids = tokenizer.encode(continuation)
    require(len(prompt_ids) > 0, "sequence_logprob requires a non-empty prompt")
    require(len(continuation_ids) > 0, "sequence_logprob requires a non-empty continuation")
    ids = prompt_ids + continuation_ids
    max_seq_len = model.config.max_seq_len
    require(len(ids) <= max_seq_len + 1, (
        f"sequence_logprob exact scoring needs at most {max_seq_len + 1} tokens, got {len(ids)}"
    ))

    device = next(model.parameters()).device
    input_ids = torch.tensor([ids[:-1]], device=device, dtype=torch.long)
    targets = torch.tensor(ids[1:], device=device, dtype=torch.long)
    logits, _ = model(input_ids)
    log_probs = F.log_softmax(logits[0], dim=-1)
    start = len(prompt_ids) - 1
    end = start + len(continuation_ids)
    token_log_probs = log_probs[start:end].gather(-1, targets[start:end].unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item())


@torch.no_grad()
def multiple_choice_accuracy(model, tokenizer, examples, normalize=False):
    """Evaluate examples shaped as {"prompt", "choices", "answer"}."""
    require(not model.training, "multiple_choice_accuracy expects model.eval() at the call boundary")
    total = 0
    correct = 0
    for ex in examples:
        choices = ex["choices"]
        answer = ex["answer"]
        require(len(choices) > 1, "multiple-choice examples require at least two choices")
        require(0 <= answer < len(choices), "multiple-choice answer index is out of range")
        scores = []
        for choice in choices:
            score = sequence_logprob(model, tokenizer, ex["prompt"], choice)
            if normalize:
                score /= max(1, len(tokenizer.encode(choice)))
            scores.append(score)
        pred = max(range(len(scores)), key=lambda i: scores[i])
        total += 1
        correct += int(pred == answer)
    require(total > 0, "multiple_choice_accuracy received no examples")
    return {"accuracy": correct / total, "correct": correct, "total": total}


@torch.no_grad()
def generation_task_accuracy(
    model,
    tokenizer,
    dataset,
    reward_fn,
    max_new_tokens=64,
    temperature=0.0,
    top_k=0,
    top_p=1.0,
):
    """Evaluate PromptDataset-like rows with `answers` and a task reward function."""
    require(not model.training, "generation_task_accuracy expects model.eval() at the call boundary")
    from minilab.data import AnsweredPromptDataset

    require(isinstance(dataset, AnsweredPromptDataset), (
        "generation_task_accuracy requires an AnsweredPromptDataset"
    ))
    answers = dataset.answers
    require(len(answers) == len(dataset), "generation_task_accuracy answers must match dataset length")
    from minilab.generation import generate

    rewards = []
    device = next(model.parameters()).device
    for idx in range(len(dataset)):
        row = dataset[idx]
        prompt_len = int(row["prompt_len"].item())
        prompt_ids = row["prompt_ids"][:prompt_len].unsqueeze(0).to(device)
        out = generate(
            model,
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
        completion = tokenizer.decode(out[0, prompt_len:].tolist())
        rewards.append(float(reward_fn(completion, answers[idx])))
    require(rewards, "generation_task_accuracy received no examples")
    mean_reward = sum(rewards) / len(rewards)
    return {"accuracy": mean_reward, "mean_reward": mean_reward, "total": len(rewards)}


def save_eval_results(results, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def distinct_n(texts, n=2):
    require(n > 0, "distinct_n requires n > 0")
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def self_bleu(texts, n=4):
    require(n > 0, "self_bleu requires n > 0")
    scores = []
    for i, hyp in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        scores.append(_bleu_single(hyp, refs, n))
    return sum(scores) / len(scores) if scores else 0.0


def _bleu_single(hypothesis, references, max_n):
    hyp_tokens = hypothesis.split()
    if not hyp_tokens or not references:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
        max_ref = Counter()
        for ref in references:
            ref_tokens = ref.split()
            ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
            for ng in ref_ngrams:
                max_ref[ng] = max(max_ref[ng], ref_ngrams[ng])
        clipped = sum(min(hyp_ngrams[ng], max_ref[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())
        precisions.append(clipped / total if total > 0 else 0.0)
    if any(p == 0 for p in precisions):
        return 0.0
    ref_lens = [len(ref.split()) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - len(hyp_tokens)), ref_len))
    brevity_penalty = 1.0 if len(hyp_tokens) > closest_ref_len else math.exp(1 - closest_ref_len / len(hyp_tokens))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / len(precisions))


def accuracy_reward(predicted, expected):
    predicted = _canonical_number(predicted)
    expected = _canonical_number(expected)
    return 1.0 if predicted is not None and predicted == expected else 0.0


def format_reward(text, pattern):
    return 1.0 if re.search(pattern, text) else 0.0


def _canonical_number(text):
    if text is None:
        return None
    matches = re.findall(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def _normalize_number(raw):
    value = raw.replace(",", "")
    negative = value.startswith("-")
    if negative:
        value = value[1:]
    if "." in value:
        integer, fractional = value.split(".", 1)
        integer = integer.lstrip("0") or "0"
        fractional = fractional.rstrip("0")
        value = integer if not fractional else f"{integer}.{fractional}"
    else:
        value = value.lstrip("0") or "0"
    return f"-{value}" if negative and value != "0" else value
