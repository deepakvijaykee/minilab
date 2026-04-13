import math
import re
from collections import Counter

import torch


@torch.no_grad()
def perplexity(model, dataloader):
    device = next(model.parameters()).device
    model.eval()
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
    return 1.0 if predicted.strip() == expected.strip() else 0.0


def extract_number(text):
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else None


def format_reward(text, pattern):
    return 1.0 if re.search(pattern, text) else 0.0
