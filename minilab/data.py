from pathlib import Path

import torch
from torch.utils.data import Dataset

from minilab.checks import require


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
            full = (p[:prompt_len] + r)[:seq_len]

            input_ids = full + [0] * (seq_len - len(full))
            labels = [-100] * seq_len
            for i in range(prompt_len - 1, len(full) - 1):
                labels[i] = full[i + 1]

            self.data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
        require(self.data, "SFTDataset received no examples")

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


class PromptDataset(Dataset):
    """Takes pre-tokenized examples ({"ids": list[int]}) so the caller
    owns any task-specific truncation (e.g. preserving a format-instruction suffix)."""

    def __init__(self, examples, seq_len):
        require(seq_len > 0, f"PromptDataset requires seq_len > 0, got {seq_len}")
        self.data = []
        self.prompt_lens = []
        for ex in examples:
            ids = ex["ids"]
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
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = ds["text"][:_example_limit(max_examples, len(ds))]
    combined = "\n".join(texts)
    return prepare_dataset(combined, tokenizer, seq_len, mode)


def load_alpaca(tokenizer, seq_len, max_examples=5000):
    """Maps instruction->prompt, output->response. Appends input field if present."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        prompt = row["instruction"]
        if row["input"]:
            prompt = prompt + "\n" + row["input"]
        examples.append({"prompt": prompt, "response": row["output"]})
    return SFTDataset(examples, tokenizer, seq_len)


def load_hh_rlhf(tokenizer, seq_len, max_examples=5000):
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        chosen, rejected = row["chosen"], row["rejected"]
        prompt, chosen_resp = _split_hh_assistant_turn(chosen, "chosen")
        rejected_prompt, rejected_resp = _split_hh_assistant_turn(rejected, "rejected")
        require(rejected_prompt == prompt, "HH-RLHF chosen/rejected prompts differ")
        examples.append({"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp})
    return PreferenceDataset(examples, tokenizer, seq_len)


def load_text8(tokenizer, seq_len, mode="lm"):
    from datasets import load_dataset
    ds = load_dataset("afmck/text8", split="train")
    text = ds[0]["text"]
    return prepare_dataset(text, tokenizer, seq_len, mode)


def load_wikitext(tokenizer, seq_len, max_examples=50000, mode="lm"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in ds["text"][:_example_limit(max_examples, len(ds))] if len(t) > 50]
    combined = "\n".join(texts)
    return prepare_dataset(combined, tokenizer, seq_len, mode)


def load_gsm8k(tokenizer, seq_len, max_examples=2000, split="train"):
    from datasets import load_dataset
    from minilab.tasks.gsm8k import parse_gold_answer, prompt_parts
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
        # breaks the task contract.
        body_ids = body_ids[: seq_len - len(suffix_ids)]
        examples.append({"ids": body_ids + suffix_ids, "answer": answer})
    return AnsweredPromptDataset(examples, seq_len)


def load_openwebtext(tokenizer, seq_len, max_examples=50000, mode="lm"):
    from datasets import load_dataset
    limit = _example_limit(max_examples)
    texts = []
    for row in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
        texts.append(row["text"])
        if limit is not None and len(texts) >= limit:
            break
    return prepare_dataset("\n".join(texts), tokenizer, seq_len, mode)


def load_dolly(tokenizer, seq_len, max_examples=5000):
    """Databricks Dolly-15k. Maps instruction->prompt, response->response."""
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        prompt = row["instruction"]
        if row["context"]:
            prompt = prompt + "\n" + row["context"]
        examples.append({"prompt": prompt, "response": row["response"]})
    return SFTDataset(examples, tokenizer, seq_len)


def load_ultrafeedback(tokenizer, seq_len, max_examples=5000):
    """UltraFeedback binarized preference pairs. Used for DPO."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    examples = []
    for row in ds.select(range(_example_limit(max_examples, len(ds)))):
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        examples.append({"prompt": row["prompt"], "chosen": chosen, "rejected": rejected})
    return PreferenceDataset(examples, tokenizer, seq_len)


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
