from pathlib import Path

import torch
from torch.utils.data import Dataset, Sampler

from minilab.checks import require
from minilab.tasks.gsm8k import parse_gold_answer, prompt_parts


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
            full = (p[:prompt_len] + r)[:seq_len]
            response_start = prompt_len
            require(len(full) > response_start, "Diffusion SFT example has no response tokens after truncation")

            input_ids = full + [0] * (seq_len - len(full))
            loss_mask = [False] * seq_len
            valid_mask = [False] * seq_len
            for i in range(response_start, len(full)):
                loss_mask[i] = True
            for i in range(len(full)):
                valid_mask[i] = True

            self.data.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
                "valid_mask": torch.tensor(valid_mask, dtype=torch.bool),
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


def load_text8(tokenizer, seq_len, mode="lm"):
    ds = load_dataset("afmck/text8", split="train")
    text = ds[0]["text"]
    return prepare_dataset(text, tokenizer, seq_len, mode)


def load_wikitext(tokenizer, seq_len, max_examples=50000, mode="lm"):
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
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
