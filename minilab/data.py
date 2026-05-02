import hashlib
import random
import re
from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from minilab.checks import require
from minilab.tasks.gsm8k import parse_gold_answer, prompt_parts

_TEXT8_TRAIN_CHARS = 90_000_000
_TEXT8_VALIDATION_CHARS = 5_000_000
_TEXT8_TEST_CHARS = 5_000_000


def load_dataset(*args, **kwargs):
    from datasets import load_dataset as hf_load_dataset
    return hf_load_dataset(*args, **kwargs)


class TextDataset(Dataset):
    """Shifted targets: input_ids[i] predicts labels[i] = tokens[i+1]."""

    def __init__(self, tokens, seq_len):
        require(seq_len > 0, f"TextDataset requires seq_len > 0, got {seq_len}")
        require(tokens.dim() == 1, "TextDataset requires a 1D token tensor")
        require(len(tokens) > seq_len, f"Need > {seq_len} tokens, got {len(tokens)}")
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


class DiffusionDataset(Dataset):
    """No shift — input_ids are both the noising input and the denoising target."""

    def __init__(self, tokens, seq_len):
        require(seq_len > 0, f"DiffusionDataset requires seq_len > 0, got {seq_len}")
        require(tokens.dim() == 1, "DiffusionDataset requires a 1D token tensor")
        require(len(tokens) >= seq_len, f"Need >= {seq_len} tokens, got {len(tokens)}")
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return {"input_ids": self.tokens[start : start + self.seq_len]}


class SFTDataset(Dataset):

    def __init__(self, examples, tokenizer, seq_len):
        require(seq_len > 1, f"SFTDataset requires seq_len > 1, got {seq_len}")
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            r = tokenizer.encode(ex["response"])
            require(len(p) > 0, "SFT example has empty prompt")
            require(len(r) > 0, "SFT example has empty response")
            prompt_len = min(len(p), seq_len - 1)

            self.data.append({
                "input_ids": _pack(p, r, prompt_len, seq_len),
                "labels": _labels(p, r, prompt_len, seq_len),
            })
        require(self.data, "SFTDataset received no examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DiffusionSFTDataset(Dataset):
    """Instruction tuning for masked diffusion LMs.

    The full prompt+response sequence is the clean sequence. `loss_mask` marks
    response tokens only; trainers keep all other tokens fixed as conditioning
    context and only diffuse/supervise response positions.
    """

    def __init__(self, examples, tokenizer, seq_len):
        require(seq_len > 1, f"DiffusionSFTDataset requires seq_len > 1, got {seq_len}")
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            r = tokenizer.encode(ex["response"])
            require(len(p) > 0, "Diffusion SFT example has empty prompt")
            require(len(r) > 0, "Diffusion SFT example has empty response")
            prompt_len = min(len(p), seq_len - 1)
            input_ids, loss_mask, valid_mask = _diffusion_pack(p, r, prompt_len, seq_len)

            self.data.append({
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "valid_mask": valid_mask,
            })
        require(self.data, "DiffusionSFTDataset received no examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PreferenceDataset(Dataset):

    def __init__(self, examples, tokenizer, seq_len):
        require(seq_len > 1, f"PreferenceDataset requires seq_len > 1, got {seq_len}")
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            c = tokenizer.encode(ex["chosen"])
            r = tokenizer.encode(ex["rejected"])
            require(len(p) > 0, "Preference example has empty prompt")
            require(len(c) > 0, "Preference example has empty chosen response")
            require(len(r) > 0, "Preference example has empty rejected response")
            prompt_len = min(len(p), seq_len - 1)

            self.data.append({
                "chosen_ids": _pack(p, c, prompt_len, seq_len),
                "chosen_labels": _labels(p, c, prompt_len, seq_len),
                "rejected_ids": _pack(p, r, prompt_len, seq_len),
                "rejected_labels": _labels(p, r, prompt_len, seq_len),
            })
        require(self.data, "PreferenceDataset received no examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KTODataset(Dataset):
    """Unpaired desirable/undesirable completions for KTO.

    Preference pairs are split into separate binary-labeled examples. The
    trainer builds the KL batch by pairing each prompt with another completion
    in the minibatch, matching the KTO implementation recipe.
    """

    def __init__(self, examples, tokenizer, seq_len):
        require(seq_len > 1, f"KTODataset requires seq_len > 1, got {seq_len}")
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            c = tokenizer.encode(ex["chosen"])
            r = tokenizer.encode(ex["rejected"])
            require(len(p) > 0, "KTO example has empty prompt")
            require(len(c) > 0, "KTO example has empty chosen response")
            require(len(r) > 0, "KTO example has empty rejected response")
            prompt_len = min(len(p), seq_len - 1)
            self.data.append(_kto_row(p, c, prompt_len, seq_len, True))
            self.data.append(_kto_row(p, r, prompt_len, seq_len, False))
        require(self.data, "KTODataset received no examples")
        self.num_pairs = len(self.data) // 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class KTOBalancedBatchSampler(Sampler):
    """Yields KTO minibatches with equal desirable and undesirable rows."""

    def __init__(self, dataset, batch_size, generator=None, shuffle=True):
        require(isinstance(dataset, KTODataset), "KTOBalancedBatchSampler requires a KTODataset")
        require(batch_size > 1, "KTOBalancedBatchSampler requires batch_size > 1")
        require(batch_size % 2 == 0, "KTOBalancedBatchSampler requires an even batch_size")
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = generator
        self.shuffle = shuffle
        self.pairs_per_batch = max(1, batch_size // 2)

    def __iter__(self):
        num_pairs = self.dataset.num_pairs
        if self.shuffle:
            desirable = torch.randperm(num_pairs, generator=self.generator)
            undesirable = torch.randperm(num_pairs, generator=self.generator)
        else:
            desirable = torch.arange(num_pairs)
            undesirable = torch.arange(num_pairs)

        for start in range(0, num_pairs, self.pairs_per_batch):
            batch = []
            end = min(start + self.pairs_per_batch, num_pairs)
            for pos_pair, neg_pair in zip(desirable[start:end], undesirable[start:end], strict=True):
                batch.append(int(2 * pos_pair.item()))
                batch.append(int(2 * neg_pair.item() + 1))
            if self.shuffle and len(batch) > 2:
                order = torch.randperm(len(batch), generator=self.generator).tolist()
                batch = [batch[i] for i in order]
            yield batch

    def __len__(self):
        return (self.dataset.num_pairs + self.pairs_per_batch - 1) // self.pairs_per_batch


class DiffusionPreferenceDataset(Dataset):
    """Chosen/rejected instruction pairs for diffusion preference tuning."""

    def __init__(self, examples, tokenizer, seq_len):
        require(seq_len > 1, f"DiffusionPreferenceDataset requires seq_len > 1, got {seq_len}")
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            c = tokenizer.encode(ex["chosen"])
            r = tokenizer.encode(ex["rejected"])
            require(len(p) > 0, "Diffusion preference example has empty prompt")
            require(len(c) > 0, "Diffusion preference example has empty chosen response")
            require(len(r) > 0, "Diffusion preference example has empty rejected response")
            prompt_len = min(len(p), seq_len - 1)
            chosen_ids, chosen_mask, chosen_valid = _diffusion_pack(p, c, prompt_len, seq_len)
            rejected_ids, rejected_mask, rejected_valid = _diffusion_pack(p, r, prompt_len, seq_len)
            self.data.append({
                "chosen_ids": chosen_ids,
                "chosen_loss_mask": chosen_mask,
                "chosen_valid_mask": chosen_valid,
                "rejected_ids": rejected_ids,
                "rejected_loss_mask": rejected_mask,
                "rejected_valid_mask": rejected_valid,
            })
        require(self.data, "DiffusionPreferenceDataset received no examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PromptDataset(Dataset):
    """Takes pre-tokenized examples ({"ids": list[int]}) so the caller
    owns any task-specific truncation (e.g. preserving a format-instruction suffix)."""

    def __init__(self, examples, seq_len):
        require(seq_len > 0, f"PromptDataset requires seq_len > 0, got {seq_len}")
        self.data = []
        self.prompt_lens = []
        for ex in examples:
            require("ids" in ex, "PromptDataset example is missing ids")
            ids = ex["ids"]
            require(type(ids) is list and all(type(token_id) is int and token_id >= 0 for token_id in ids), (
                "PromptDataset ids must be a list of non-negative integer token ids"
            ))
            require(len(ids) > 0, "PromptDataset example has empty prompt")
            require(len(ids) <= seq_len, f"prompt len {len(ids)} exceeds seq_len {seq_len}")
            self.prompt_lens.append(len(ids))
            ids = ids + [0] * (seq_len - len(ids))
            self.data.append(torch.tensor(ids, dtype=torch.long))
        require(self.data, "PromptDataset received no examples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "prompt_ids": self.data[idx],
            "prompt_len": torch.tensor(self.prompt_lens[idx], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


class AnsweredPromptDataset(PromptDataset):
    """Prompt dataset variant for tasks whose reward/eval path needs gold answers."""

    def __init__(self, examples, seq_len):
        examples = list(examples)
        require(all("answer" in ex for ex in examples), "AnsweredPromptDataset requires answer for every example")
        self.answers = [ex["answer"] for ex in examples]
        super().__init__(examples, seq_len)


def load_tinystories(tokenizer, seq_len, split="train", max_examples=10000, mode="lm"):
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = ds["text"][:_example_limit(max_examples, len(ds))]
    combined = "\n".join(texts)
    return prepare_dataset(combined, tokenizer, seq_len, mode)


def load_alpaca(tokenizer, seq_len, max_examples=5000):
    """Maps instruction->prompt, output->response. Appends input field if present."""
    return SFTDataset(_alpaca_examples(max_examples), tokenizer, seq_len)


def load_alpaca_diffusion(tokenizer, seq_len, max_examples=5000):
    return DiffusionSFTDataset(_alpaca_examples(max_examples), tokenizer, seq_len)


def _alpaca_examples(max_examples):
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        prompt = row["instruction"]
        if row["input"]:
            prompt = prompt + "\n" + row["input"]
        examples.append({"prompt": prompt, "response": row["output"]})
    return examples


def load_hh_rlhf(tokenizer, seq_len, max_examples=5000):
    return PreferenceDataset(_hh_rlhf_examples(max_examples), tokenizer, seq_len)


def load_hh_rlhf_kto(tokenizer, seq_len, max_examples=5000):
    return KTODataset(_hh_rlhf_examples(max_examples), tokenizer, seq_len)


def load_hh_rlhf_diffusion(tokenizer, seq_len, max_examples=5000):
    return DiffusionPreferenceDataset(_hh_rlhf_examples(max_examples), tokenizer, seq_len)


def _hh_rlhf_examples(max_examples):
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        chosen, rejected = row["chosen"], row["rejected"]
        prompt, chosen_resp = _split_hh_assistant_turn(chosen, "chosen")
        rejected_prompt, rejected_resp = _split_hh_assistant_turn(rejected, "rejected")
        require(rejected_prompt == prompt, "HH-RLHF chosen/rejected prompts differ")
        examples.append({"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp})
    return examples


def load_text8(tokenizer, seq_len, split="train", mode="lm"):
    ds = load_dataset("afmck/text8", split="train")
    text = text8_standard_split(ds[0]["text"], split)
    return prepare_dataset(text, tokenizer, seq_len, mode)


def text8_standard_split(text, split):
    require(split in {"train", "validation", "test"}, "text8 split must be 'train', 'validation', or 'test'")
    train_end = _TEXT8_TRAIN_CHARS
    validation_end = train_end + _TEXT8_VALIDATION_CHARS
    test_end = validation_end + _TEXT8_TEST_CHARS
    require(len(text) >= test_end, (
        f"text8 corpus must contain at least {test_end} characters for the standard split, got {len(text)}"
    ))
    if split == "train":
        return text[:train_end]
    if split == "validation":
        return text[train_end:validation_end]
    return text[validation_end:test_end]


def load_wikitext(tokenizer, seq_len, split="train", max_examples=50000, mode="lm"):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    texts = [t for t in ds["text"][:_example_limit(max_examples, len(ds))] if len(t) > 50]
    combined = "\n".join(texts)
    return prepare_dataset(combined, tokenizer, seq_len, mode)


def load_gsm8k(tokenizer, seq_len, max_examples=2000, split="train"):
    ds = load_dataset("openai/gsm8k", "main", split=split)
    n = _example_limit(max_examples, len(ds))
    examples = []
    for row in ds.select(range(n)):
        answer = parse_gold_answer(row["answer"])
        body, suffix = prompt_parts(row["question"])
        body_ids = tokenizer.encode(body)
        suffix_ids = tokenizer.encode(suffix)
        require(len(suffix_ids) < seq_len, f"suffix alone ({len(suffix_ids)} tok) exceeds seq_len {seq_len}")
        # Truncate the question body so the format-instruction suffix is always kept —
        # the GSM8K reward strictly requires '####', so losing the suffix silently
        # breaks the task contract. Leave one context slot for online RL rollout
        # sampling, which refuses zero-token completions by design.
        body_ids = body_ids[: seq_len - len(suffix_ids) - 1]
        examples.append({"ids": body_ids + suffix_ids, "answer": answer})
    return AnsweredPromptDataset(examples, seq_len)


def load_gsm8k_diffusion(tokenizer, seq_len, max_response_tokens, max_examples=2000, split="train"):
    require(max_response_tokens > 0, "max_response_tokens must be > 0")
    require(max_response_tokens < seq_len, "diffusion GSM8K requires seq_len > max_response_tokens")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    n = _example_limit(max_examples, len(ds))
    prompt_budget = seq_len - max_response_tokens
    examples = []
    for row in ds.select(range(n)):
        answer = parse_gold_answer(row["answer"])
        body, suffix = prompt_parts(row["question"])
        body_ids = tokenizer.encode(body)
        suffix_ids = tokenizer.encode(suffix)
        require(len(suffix_ids) < prompt_budget, (
            f"suffix alone ({len(suffix_ids)} tok) exceeds diffusion prompt budget {prompt_budget}"
        ))
        body_ids = body_ids[: prompt_budget - len(suffix_ids)]
        examples.append({"ids": body_ids + suffix_ids, "answer": answer})
    return AnsweredPromptDataset(examples, seq_len)


def load_openwebtext(tokenizer, seq_len, max_examples=50000, mode="lm"):
    require(max_examples > 0, "streaming OpenWebText requires max_examples > 0")
    limit = _example_limit(max_examples)
    texts = []
    for row in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
        texts.append(row["text"])
        if limit is not None and len(texts) >= limit:
            break
    return prepare_dataset("\n".join(texts), tokenizer, seq_len, mode)


def load_dolly(tokenizer, seq_len, max_examples=5000):
    """Databricks Dolly-15k. Maps instruction->prompt, response->response."""
    return SFTDataset(_dolly_examples(max_examples), tokenizer, seq_len)


def load_dolly_diffusion(tokenizer, seq_len, max_examples=5000):
    return DiffusionSFTDataset(_dolly_examples(max_examples), tokenizer, seq_len)


def _dolly_examples(max_examples):
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        prompt = row["instruction"]
        if row["context"]:
            prompt = prompt + "\n" + row["context"]
        examples.append({"prompt": prompt, "response": row["response"]})
    return examples


def load_ultrafeedback(tokenizer, seq_len, max_examples=5000):
    """UltraFeedback binarized preference pairs. Used for DPO."""
    return PreferenceDataset(_ultrafeedback_examples(max_examples), tokenizer, seq_len)


def load_ultrafeedback_kto(tokenizer, seq_len, max_examples=5000):
    return KTODataset(_ultrafeedback_examples(max_examples), tokenizer, seq_len)


def load_ultrafeedback_diffusion(tokenizer, seq_len, max_examples=5000):
    return DiffusionPreferenceDataset(_ultrafeedback_examples(max_examples), tokenizer, seq_len)


def _ultrafeedback_examples(max_examples):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        examples.append({"prompt": row["prompt"], "chosen": chosen, "rejected": rejected})
    return examples


def _pack(prompt, response, prompt_len, seq_len):
    full = (prompt[:prompt_len] + response)[:seq_len]
    padded = full + [0] * (seq_len - len(full))
    return torch.tensor(padded, dtype=torch.long)


def _labels(prompt, response, prompt_len, seq_len):
    require(prompt_len > 0, f"prompt_len must be > 0, got {prompt_len}")
    full = (prompt[:prompt_len] + response)[:seq_len]
    labels = [-100] * seq_len
    for i in range(prompt_len - 1, len(full) - 1):
        labels[i] = full[i + 1]
    return torch.tensor(labels, dtype=torch.long)


def _kto_row(prompt, response, prompt_len, seq_len, desirable):
    full = (prompt[:prompt_len] + response)[:seq_len]
    response_len = len(full) - prompt_len
    require(response_len > 0, "KTO packed example has no response tokens after truncation")
    return {
        "input_ids": _pack(prompt, response, prompt_len, seq_len),
        "labels": _labels(prompt, response, prompt_len, seq_len),
        "preference_label": torch.tensor(desirable, dtype=torch.bool),
        "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
        "response_len": torch.tensor(response_len, dtype=torch.long),
    }


def _diffusion_pack(prompt, response, prompt_len, seq_len):
    full = (prompt[:prompt_len] + response)[:seq_len]
    require(len(full) > prompt_len, "diffusion packed example has no response tokens after truncation")
    padded = full + [0] * (seq_len - len(full))
    loss_mask = [False] * seq_len
    valid_mask = [False] * seq_len
    for i in range(prompt_len, len(full)):
        loss_mask[i] = True
    for i in range(len(full)):
        valid_mask[i] = True
    return (
        torch.tensor(padded, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.bool),
        torch.tensor(valid_mask, dtype=torch.bool),
    )


def _split_hh_assistant_turn(text, field_name):
    marker = "\n\nAssistant:"
    require(marker in text, f"HH-RLHF {field_name} response is missing the final Assistant turn marker")
    prompt, response = text.rsplit(marker, 1)
    require(response, f"HH-RLHF {field_name} response is empty after the final Assistant turn marker")
    return prompt + marker, response


def _example_limit(max_examples, total=None):
    require(max_examples >= 0, f"max_examples must be >= 0, got {max_examples}")
    if max_examples == 0:
        return total
    if total is None:
        return max_examples
    return min(max_examples, total)


def prepare_dataset(text, tokenizer, seq_len, mode="lm"):
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if mode == "lm":
        return TextDataset(tokens, seq_len)
    if mode == "diffusion":
        return DiffusionDataset(tokens, seq_len)
    raise ValueError(f"Unknown mode: {mode}")


def load_text(path):
    return Path(path).read_text()


def normalize_text_for_dedup(text):
    return " ".join(text.split()).lower()


def dedupe_texts(texts):
    seen = set()
    kept = []
    for text in texts:
        key = normalize_text_for_dedup(text)
        if key in seen:
            continue
        seen.add(key)
        kept.append(text)
    return kept


def curate_texts(texts, min_chars=1, max_chars=0, min_words=0, max_words=0, dedupe=True):
    require(min_chars >= 0, "min_chars must be >= 0")
    require(max_chars >= 0, "max_chars must be >= 0")
    require(min_words >= 0, "min_words must be >= 0")
    require(max_words >= 0, "max_words must be >= 0")
    require(max_chars == 0 or max_chars >= min_chars, "max_chars must be 0 or >= min_chars")
    require(max_words == 0 or max_words >= min_words, "max_words must be 0 or >= min_words")

    kept = []
    for text in texts:
        words = text.split()
        if len(text) < min_chars:
            continue
        if max_chars and len(text) > max_chars:
            continue
        if len(words) < min_words:
            continue
        if max_words and len(words) > max_words:
            continue
        kept.append(text)
    return dedupe_texts(kept) if dedupe else kept


def mix_text_sources(sources, weights, max_examples, seed=0):
    require(len(sources) == len(weights), "sources and weights must have the same length")
    require(len(sources) > 0, "mix_text_sources requires at least one source")
    require(max_examples > 0, "max_examples must be > 0")
    require(all(w > 0 for w in weights), "source weights must be > 0")
    rng = random.Random(seed)
    pools = [list(src) for src in sources]
    for pool in pools:
        rng.shuffle(pool)

    out = []
    while len(out) < max_examples and any(pools):
        active = [i for i, pool in enumerate(pools) if pool]
        active_weights = [weights[i] for i in active]
        chosen = rng.choices(active, weights=active_weights, k=1)[0]
        out.append(pools[chosen].pop())
    return out


def text_curation_report(texts):
    texts = list(texts)
    chars = [len(text) for text in texts]
    words = [len(text.split()) for text in texts]
    total_chars = sum(chars)
    total_words = sum(words)
    n = len(texts)
    return {
        "examples": n,
        "characters": total_chars,
        "words": total_words,
        "mean_characters": total_chars / n if n else 0.0,
        "mean_words": total_words / n if n else 0.0,
    }


def line_repetition_fraction(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    return (len(lines) - len(set(lines))) / len(lines)


def ngram_repetition_fraction(text, n=5):
    require(n > 0, "n must be > 0")
    words = re.findall(r"\w+", normalize_text_for_dedup(text))
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def text_quality_stats(text):
    non_space = sum(not ch.isspace() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    digits = sum(ch.isdigit() for ch in text)
    words = re.findall(r"\w+", text)
    return {
        "characters": len(text),
        "words": len(words),
        "alpha_fraction": letters / non_space if non_space else 0.0,
        "digit_fraction": digits / non_space if non_space else 0.0,
        "line_repetition_fraction": line_repetition_fraction(text),
        "ngram_repetition_fraction": ngram_repetition_fraction(text),
    }


def quality_filter_texts(
    texts,
    min_chars=200,
    min_words=40,
    max_repeated_line_fraction=0.3,
    max_repeated_ngram_fraction=1.0,
    min_alpha_fraction=0.55,
):
    """FineWeb/DCLM-style lightweight quality gate for raw web text."""
    require(min_chars >= 0, "min_chars must be >= 0")
    require(min_words >= 0, "min_words must be >= 0")
    require(0.0 <= max_repeated_line_fraction <= 1.0, (
        "max_repeated_line_fraction must be in [0, 1]"
    ))
    require(0.0 <= max_repeated_ngram_fraction <= 1.0, (
        "max_repeated_ngram_fraction must be in [0, 1]"
    ))
    require(0.0 <= min_alpha_fraction <= 1.0, "min_alpha_fraction must be in [0, 1]")
    kept = []
    for text in texts:
        stats = text_quality_stats(text)
        if stats["characters"] < min_chars:
            continue
        if stats["words"] < min_words:
            continue
        if stats["alpha_fraction"] < min_alpha_fraction:
            continue
        if stats["line_repetition_fraction"] > max_repeated_line_fraction:
            continue
        if stats["ngram_repetition_fraction"] > max_repeated_ngram_fraction:
            continue
        kept.append(text)
    return kept


def minhash_signature(text, num_hashes=64, shingle_size=5):
    require(num_hashes > 0, "num_hashes must be > 0")
    require(shingle_size > 0, "shingle_size must be > 0")
    words = re.findall(r"\w+", normalize_text_for_dedup(text))
    require(words, "minhash_signature requires at least one token")
    if len(words) < shingle_size:
        shingles = {" ".join(words)}
    else:
        shingles = {" ".join(words[i : i + shingle_size]) for i in range(len(words) - shingle_size + 1)}
    sig = []
    for seed in range(num_hashes):
        sig.append(min(_stable_hash(f"{seed}:{shingle}") for shingle in shingles))
    return tuple(sig)


def dedupe_texts_minhash(texts, threshold=0.8, num_hashes=64, shingle_size=5):
    """Near-deduplicate texts using MinHash-estimated Jaccard similarity."""
    require(0.0 <= threshold <= 1.0, "threshold must be in [0, 1]")
    kept = []
    signatures = []
    for text in texts:
        sig = minhash_signature(text, num_hashes=num_hashes, shingle_size=shingle_size)
        duplicate = False
        for prev_sig in signatures:
            sim = sum(a == b for a, b in zip(sig, prev_sig, strict=True)) / num_hashes
            if sim >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(text)
        signatures.append(sig)
    return kept


def minhash_lsh_buckets(signatures, bands=8):
    signatures = list(signatures)
    require(signatures, "minhash_lsh_buckets requires at least one signature")
    require(bands > 0, "bands must be > 0")
    width = len(signatures[0])
    require(width > 0, "signatures must be non-empty")
    require(width % bands == 0, "signature length must be divisible by bands")
    rows = width // bands
    buckets = {}
    for idx, sig in enumerate(signatures):
        require(len(sig) == width, "all signatures must have the same length")
        for band in range(bands):
            start = band * rows
            key = (band, tuple(sig[start : start + rows]))
            buckets.setdefault(key, []).append(idx)
    return buckets


def dedupe_texts_minhash_lsh(texts, threshold=0.8, num_hashes=64, shingle_size=5, bands=8):
    """Near-deduplicate texts with MinHash LSH candidate pruning."""
    require(0.0 <= threshold <= 1.0, "threshold must be in [0, 1]")
    require(bands > 0, "bands must be > 0")
    require(num_hashes % bands == 0, "num_hashes must be divisible by bands")
    kept = []
    signatures = []
    buckets = {}
    rows = num_hashes // bands
    for text in texts:
        sig = minhash_signature(text, num_hashes=num_hashes, shingle_size=shingle_size)
        candidates = set()
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows])
            if key in buckets:
                candidates.update(buckets[key])
        duplicate = False
        for prev_idx in candidates:
            prev_sig = signatures[prev_idx]
            sim = sum(a == b for a, b in zip(sig, prev_sig, strict=True)) / num_hashes
            if sim >= threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept_idx = len(kept)
        kept.append(text)
        signatures.append(sig)
        for band in range(bands):
            key = (band, sig[band * rows : (band + 1) * rows])
            buckets.setdefault(key, []).append(kept_idx)
    return kept


def contamination_report(train_texts, eval_texts, threshold=0.8, num_hashes=64, shingle_size=5):
    require(0.0 <= threshold <= 1.0, "threshold must be in [0, 1]")
    train_texts = list(train_texts)
    eval_texts = list(eval_texts)
    eval_sigs = [minhash_signature(text, num_hashes, shingle_size) for text in eval_texts]
    matches = []
    for train_idx, text in enumerate(train_texts):
        sig = minhash_signature(text, num_hashes, shingle_size)
        for eval_idx, eval_sig in enumerate(eval_sigs):
            sim = sum(a == b for a, b in zip(sig, eval_sig, strict=True)) / num_hashes
            if sim >= threshold:
                matches.append({"train_index": train_idx, "eval_index": eval_idx, "similarity": sim})
    return {
        "train_examples": len(train_texts),
        "eval_examples": len(eval_sigs),
        "matches": matches,
        "contaminated_train_examples": len({m["train_index"] for m in matches}),
    }


def source_mixture_report(sources, weights):
    require(len(sources) == len(weights), "sources and weights must have the same length")
    require(len(sources) > 0, "source_mixture_report requires at least one source")
    require(all(w > 0 for w in weights), "source weights must be > 0")
    total_weight = sum(weights)
    rows = []
    for i, (source, weight) in enumerate(zip(sources, weights, strict=True)):
        texts = list(source)
        rows.append({
            "source": i,
            "examples": len(texts),
            "weight": weight,
            "target_fraction": weight / total_weight,
            "words": sum(len(text.split()) for text in texts),
        })
    return rows


def allocate_source_counts(weights, max_examples):
    require(max_examples > 0, "max_examples must be > 0")
    require(len(weights) > 0, "weights must be non-empty")
    require(all(w > 0 for w in weights), "source weights must be > 0")
    total = sum(weights)
    raw = [max_examples * w / total for w in weights]
    counts = [int(x) for x in raw]
    remaining = max_examples - sum(counts)
    order = sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in order[:remaining]:
        counts[i] += 1
    return counts


def mix_text_sources_exact(sources, weights, max_examples, seed=0):
    """Deterministic source mixture with exact quota allocation when capacity allows."""
    require(len(sources) == len(weights), "sources and weights must have the same length")
    require(len(sources) > 0, "mix_text_sources_exact requires at least one source")
    require(max_examples > 0, "max_examples must be > 0")
    require(all(w > 0 for w in weights), "source weights must be > 0")
    rng = random.Random(seed)
    pools = [list(src) for src in sources]
    for pool in pools:
        rng.shuffle(pool)
    caps = [len(pool) for pool in pools]
    counts = _allocate_counts_with_caps(weights, caps, max_examples)
    out = []
    for pool, count in zip(pools, counts, strict=True):
        out.extend(pool[:count])
    rng.shuffle(out)
    return out


def _stable_hash(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "big")


def _allocate_counts_with_caps(weights, caps, max_examples):
    counts = [min(count, cap) for count, cap in zip(allocate_source_counts(weights, max_examples), caps, strict=True)]
    while sum(counts) < max_examples:
        active = [i for i, cap in enumerate(caps) if counts[i] < cap]
        if not active:
            break
        total_weight = sum(weights[i] for i in active)
        target = {i: max_examples * weights[i] / total_weight for i in active}
        chosen = max(active, key=lambda i: target[i] - counts[i])
        counts[chosen] += 1
    return counts
