import math
import re
from collections import Counter

import torch


@torch.no_grad()
def perplexity(model, dataloader):
    assert not model.training, "perplexity expects model.eval() at the call boundary"
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        _, loss = model(ids, labels)
        n = (labels != -100).sum().item()
        total_nll += loss.item() * n
        total_tokens += n
    return math.exp(total_nll / total_tokens)


def distinct_n(texts, n=2):
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def self_bleu(texts, n=4):
    scores = []
    for i, hyp in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        scores.append(_bleu_single(hyp, refs, n))
    return sum(scores) / len(scores) if scores else 0.0


def _bleu_single(hypothesis, references, max_n):
    hyp_tokens = hypothesis.split()
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
    return math.exp(sum(math.log(p) for p in precisions) / len(precisions))


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
