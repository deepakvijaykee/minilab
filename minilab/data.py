from pathlib import Path

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Shifted targets: input_ids[i] predicts labels[i] = tokens[i+1]."""

    def __init__(self, tokens, seq_len):
        assert tokens.dim() == 1
        assert len(tokens) > seq_len, f"Need > {seq_len} tokens, got {len(tokens)}"
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
        assert tokens.dim() == 1
        assert len(tokens) >= seq_len, f"Need >= {seq_len} tokens, got {len(tokens)}"
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        return {"input_ids": self.tokens[start : start + self.seq_len]}


class SFTDataset(Dataset):

    def __init__(self, examples, tokenizer, seq_len):
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            r = tokenizer.encode(ex["response"])
            assert len(p) > 0, "SFT example has empty prompt"
            assert len(r) > 0, "SFT example has empty response"
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PreferenceDataset(Dataset):

    def __init__(self, examples, tokenizer, seq_len):
        self.data = []
        for ex in examples:
            p = tokenizer.encode(ex["prompt"])
            c = tokenizer.encode(ex["chosen"])
            r = tokenizer.encode(ex["rejected"])
            assert len(p) > 0, "Preference example has empty prompt"
            prompt_len = min(len(p), seq_len - 1)

            self.data.append({
                "chosen_ids": _pack(p, c, prompt_len, seq_len),
                "chosen_labels": _labels(p, c, prompt_len, seq_len),
                "rejected_ids": _pack(p, r, prompt_len, seq_len),
                "rejected_labels": _labels(p, r, prompt_len, seq_len),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class PromptDataset(Dataset):
    """Takes pre-tokenized examples ({"ids": list[int], "answer": str}) so the caller
    owns any task-specific truncation (e.g. preserving a format-instruction suffix)."""

    def __init__(self, examples, seq_len):
        self.answers = [ex["answer"] for ex in examples]
        self.data = []
        self.prompt_lens = []
        for ex in examples:
            ids = ex["ids"]
            assert len(ids) <= seq_len, f"prompt len {len(ids)} exceeds seq_len {seq_len}"
            self.prompt_lens.append(len(ids))
            ids = ids + [0] * (seq_len - len(ids))
            self.data.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "prompt_ids": self.data[idx],
            "prompt_len": torch.tensor(self.prompt_lens[idx], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
        }


def load_tinystories(tokenizer, seq_len, split="train", max_examples=10000, mode="lm"):
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories", split=split)
    texts = ds["text"][:max_examples]
    combined = "\n".join(texts)
    tokens = torch.tensor(tokenizer.encode(combined), dtype=torch.long)
    if mode == "lm":
        return TextDataset(tokens, seq_len)
    return DiffusionDataset(tokens, seq_len)


def load_alpaca(tokenizer, seq_len, max_examples=5000):
    """Maps instruction->prompt, output->response. Appends input field if present."""
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    examples = []
    for row in ds.select(range(min(max_examples, len(ds)))):
        prompt = row["instruction"]
        if row["input"]:
            prompt = prompt + "\n" + row["input"]
        examples.append({"prompt": prompt, "response": row["output"]})
    return SFTDataset(examples, tokenizer, seq_len)


def load_hh_rlhf(tokenizer, seq_len, max_examples=5000):
    from datasets import load_dataset
    ds = load_dataset("Anthropic/hh-rlhf", split="train")
    examples = []
    for row in ds.select(range(min(max_examples, len(ds)))):
        # Format: "\n\nHuman: ...\n\nAssistant: ..."
        chosen, rejected = row["chosen"], row["rejected"]
        # Split on last Assistant turn
        if "\n\nAssistant:" in chosen:
            parts = chosen.rsplit("\n\nAssistant:", 1)
            prompt = parts[0] + "\n\nAssistant:"
            chosen_resp = parts[1]
        else:
            continue
        if "\n\nAssistant:" in rejected:
            rejected_resp = rejected.rsplit("\n\nAssistant:", 1)[1]
        else:
            continue
        examples.append({"prompt": prompt, "chosen": chosen_resp, "rejected": rejected_resp})
    return PreferenceDataset(examples, tokenizer, seq_len)


def load_text8(tokenizer, seq_len, mode="lm"):
    from datasets import load_dataset
    ds = load_dataset("afmck/text8", split="train")
    text = ds[0]["text"]
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if mode == "lm":
        return TextDataset(tokens, seq_len)
    return DiffusionDataset(tokens, seq_len)


def load_wikitext(tokenizer, seq_len, max_examples=50000, mode="lm"):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    texts = [t for t in ds["text"][:max_examples] if len(t) > 50]
    combined = "\n".join(texts)
    tokens = torch.tensor(tokenizer.encode(combined), dtype=torch.long)
    if mode == "lm":
        return TextDataset(tokens, seq_len)
    return DiffusionDataset(tokens, seq_len)


def load_gsm8k(tokenizer, seq_len, max_examples=2000, split="train"):
    from datasets import load_dataset
    from minilab.tasks.gsm8k import DELIMITER, prompt_parts
    ds = load_dataset("openai/gsm8k", "main", split=split)
    n = len(ds) if max_examples == 0 else min(max_examples, len(ds))
    examples = []
    for row in ds.select(range(n)):
        answer = row["answer"].split(DELIMITER)[-1].strip()
        body, suffix = prompt_parts(row["question"])
        body_ids = tokenizer.encode(body)
        suffix_ids = tokenizer.encode(suffix)
        assert len(suffix_ids) < seq_len, f"suffix alone ({len(suffix_ids)} tok) exceeds seq_len {seq_len}"
        # Truncate the question body so the format-instruction suffix is always kept —
        # the GSM8K reward strictly requires '####', so losing the suffix silently
        # breaks the task contract.
        body_ids = body_ids[: seq_len - len(suffix_ids)]
        examples.append({"ids": body_ids + suffix_ids, "answer": answer})
    return PromptDataset(examples, seq_len)


def load_openwebtext(tokenizer, seq_len, max_examples=50000, mode="lm"):
    from datasets import load_dataset
    texts = []
    for row in load_dataset("Skylion007/openwebtext", split="train", streaming=True):
        texts.append(row["text"])
        if len(texts) >= max_examples:
            break
    tokens = torch.tensor(tokenizer.encode("\n".join(texts)), dtype=torch.long)
    if mode == "lm":
        return TextDataset(tokens, seq_len)
    return DiffusionDataset(tokens, seq_len)


def load_dolly(tokenizer, seq_len, max_examples=5000):
    """Databricks Dolly-15k. Maps instruction->prompt, response->response."""
    from datasets import load_dataset
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    examples = []
    for row in ds.select(range(min(max_examples, len(ds)))):
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
    for row in ds.select(range(min(max_examples, len(ds)))):
        chosen = row["chosen"][-1]["content"]
        rejected = row["rejected"][-1]["content"]
        examples.append({"prompt": row["prompt"], "chosen": chosen, "rejected": rejected})
    return PreferenceDataset(examples, tokenizer, seq_len)


def _pack(prompt, response, prompt_len, seq_len):
    full = (prompt[:prompt_len] + response)[:seq_len]
    padded = full + [0] * (seq_len - len(full))
    return torch.tensor(padded, dtype=torch.long)


def _labels(prompt, response, prompt_len, seq_len):
    full = (prompt[:prompt_len] + response)[:seq_len]
    labels = [-100] * seq_len
    for i in range(prompt_len - 1, len(full) - 1):
        labels[i] = full[i + 1]
    return torch.tensor(labels, dtype=torch.long)


def prepare_dataset(text, tokenizer, seq_len, mode="lm"):
    tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if mode == "lm":
        return TextDataset(tokens, seq_len)
    if mode == "diffusion":
        return DiffusionDataset(tokens, seq_len)
    raise ValueError(f"Unknown mode: {mode}")


def load_text(path):
    return Path(path).read_text()
