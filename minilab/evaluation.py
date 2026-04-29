import math
import re
from collections import Counter

import torch
import torch.nn.functional as F

from minilab.checks import require


@torch.no_grad()
def perplexity(model, dataloader):
    require(not model.training, "perplexity expects model.eval() at the call boundary")
    device = next(model.parameters()).device
    total_nll = 0.0
    total_tokens = 0
    for batch in dataloader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(ids)
        n = (labels != -100).sum().item()
        total_nll += F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        total_tokens += n
    require(total_tokens > 0, "perplexity received no valid target tokens")
    return math.exp(total_nll / total_tokens)


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
