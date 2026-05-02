"""Rule and learned verifiers for outcome-supervised training."""

import json
import math
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.models.transformer_utils import (
    commit_transformer_block_updates,
    set_transformer_qk_clip_recording,
    transformer_auxiliary_loss,
    transformer_supports_qk_clip,
)
from minilab.models.gpt import (
    GPTConfig,
    TransformerBlock,
)
from minilab.registry import get_norm, get_position, register_model


class ExactMatchVerifier:
    def __call__(self, prediction, reference):
        return 1.0 if prediction.strip() == reference.strip() else 0.0


class RegexVerifier:
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)

    def __call__(self, prediction, reference=None):
        return 1.0 if self.pattern.search(prediction) else 0.0


class NumericVerifier:
    def __init__(self, tolerance=0.0):
        require(tolerance >= 0, "tolerance must be >= 0")
        self.tolerance = tolerance

    def __call__(self, prediction, reference):
        pred = _last_number(prediction)
        ref = _last_number(reference)
        if pred is None or ref is None:
            return 0.0
        return 1.0 if math.isclose(pred, ref, rel_tol=0.0, abs_tol=self.tolerance) else 0.0


class PythonUnitTestVerifier:
    """Run generated Python code against caller-provided tests in a subprocess."""

    def __init__(self, timeout_seconds=5.0):
        require(timeout_seconds > 0, "timeout_seconds must be > 0")
        self.timeout_seconds = timeout_seconds

    def __call__(self, solution_code, tests_code):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate_test.py"
            path.write_text(solution_code + "\n\n" + tests_code)
            try:
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    cwd=tmp,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return 0.0
        return 1.0 if proc.returncode == 0 else 0.0


class CompositeVerifier:
    def __init__(self, verifiers, reducer="all"):
        self.verifiers = list(verifiers)
        require(len(self.verifiers) > 0, "CompositeVerifier requires at least one verifier")
        require(reducer in {"all", "any", "mean"}, "reducer must be 'all', 'any', or 'mean'")
        self.reducer = reducer

    def __call__(self, prediction, reference):
        scores = [verifier(prediction, reference) for verifier in self.verifiers]
        if self.reducer == "all":
            return min(scores)
        if self.reducer == "any":
            return max(scores)
        return sum(scores) / len(scores)


class ToolCallVerifier:
    """Verify a JSON tool call against a caller-owned deterministic tool map."""

    def __init__(self, tools, result_verifier=None):
        require(len(tools) > 0, "ToolCallVerifier requires at least one tool")
        self.tools = dict(tools)
        self.result_verifier = result_verifier or ExactMatchVerifier()

    def __call__(self, prediction, reference):
        try:
            call = json.loads(_extract_json_object(prediction))
        except (json.JSONDecodeError, ValueError):
            return 0.0
        if not isinstance(call, dict):
            return 0.0
        if "tool" not in call or "arguments" not in call:
            return 0.0
        tool_name = call["tool"]
        arguments = call["arguments"]
        if tool_name not in self.tools or not isinstance(arguments, dict):
            return 0.0
        result = self.tools[tool_name](**arguments)
        return self.result_verifier(str(result), str(reference))


class VerifierDataset(Dataset):
    def __init__(self, examples, tokenizer, seq_len):
        require(seq_len > 1, "VerifierDataset requires seq_len > 1")
        self.rows = []
        for ex in examples:
            ids = tokenizer.encode(ex["text"])[:seq_len]
            require(ids, "VerifierDataset example has empty text")
            label = float(ex["label"])
            require(label in {0.0, 1.0}, "VerifierDataset labels must be 0 or 1")
            self.rows.append({
                "input_ids": torch.tensor(ids + [0] * (seq_len - len(ids)), dtype=torch.long),
                "attention_mask": torch.tensor([1] * len(ids) + [0] * (seq_len - len(ids)), dtype=torch.bool),
                "labels": torch.tensor(label, dtype=torch.float32),
            })
        require(self.rows, "VerifierDataset received no examples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


@dataclass
class VerifierConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 512
    dropout: float = 0.0
    ffn_mult: float = 4.0
    attention: str = "mha"
    ffn: str = "swiglu"
    norm: str = "rmsnorm"

    def __post_init__(self):
        require(self.vocab_size > 0, "vocab_size must be > 0")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.dim % self.num_heads == 0, "dim must be divisible by num_heads")
        require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.ffn_mult > 0, "ffn_mult must be > 0")


@register_model("verifier")
class OutcomeVerifier(BaseModel):
    config_class = VerifierConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        gpt_cfg = GPTConfig(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            ffn_mult=config.ffn_mult,
            attention=config.attention,
            ffn=config.ffn,
            norm=config.norm,
        )
        self.blocks = nn.ModuleList([TransformerBlock(gpt_cfg, i) for i in range(config.num_layers)])
        self.pos_enc = get_position("rope")(config.dim // config.num_heads, config.max_seq_len)
        self.ln_f = get_norm(config.norm)(config.dim)
        self.score_head = nn.Linear(config.dim, 1)
        self.apply(self._init_weights)

    def set_qk_clip_recording(self, enabled):
        set_transformer_qk_clip_recording(self.blocks, enabled)

    def supports_qk_clip(self):
        return transformer_supports_qk_clip(self.blocks)

    def auxiliary_loss(self):
        return transformer_auxiliary_loss(self.blocks, self.config.ffn, next(self.parameters()))

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_transformer_block_updates(
            self.blocks,
            self.config.ffn,
            qk_clip_threshold,
            qk_clip_balance,
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        require(input_ids.size(1) <= self.config.max_seq_len, (
            f"OutcomeVerifier supports at most {self.config.max_seq_len} tokens, got {input_ids.size(1)}"
        ))
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            require(attention_mask.shape == input_ids.shape, "OutcomeVerifier attention_mask must match input_ids")
            require(attention_mask.dtype == torch.bool, "OutcomeVerifier attention_mask must be bool")
            attention_mask = attention_mask.to(input_ids.device)
        require(attention_mask.any(dim=-1).all(), "OutcomeVerifier attention_mask must include at least one token")
        expected_mask = (
            torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
            < attention_mask.long().sum(dim=-1, keepdim=True)
        )
        require(torch.equal(attention_mask, expected_mask), (
            "OutcomeVerifier attention_mask must be a right-padded prefix mask"
        ))
        x = self._cast_hidden(self.tok_emb(input_ids))
        x = self.drop(x)
        freqs_cis = self.pos_enc(input_ids.size(1))
        for block in self.blocks:
            x = self._checkpointed_forward(block, x, freqs_cis=freqs_cis, is_causal=True)
        x = self.ln_f(x)
        lengths = attention_mask.long().sum(dim=-1).clamp(min=1) - 1
        pooled = x[torch.arange(x.size(0), device=x.device), lengths]
        logits = self.score_head(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            require(labels.shape == logits.shape, "OutcomeVerifier labels must have shape (batch,)")
            loss = F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))
            loss = loss + self.auxiliary_loss()
        return logits, loss

def verifier_accuracy(logits, labels):
    preds = logits.sigmoid() >= 0.5
    labels = labels.bool()
    require(labels.numel() > 0, "verifier_accuracy requires at least one label")
    return (preds == labels).float().mean().item()


def _last_number(text):
    matches = re.findall(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def _extract_json_object(text):
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("missing JSON object")
    return text[start : end + 1]
