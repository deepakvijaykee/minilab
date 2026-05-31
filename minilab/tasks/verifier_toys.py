"""Tiny verifiable tasks for local RLVR transfer experiments."""

import json
import re

import torch
from torch.utils.data import Dataset

from minilab.checks import require
from minilab.registry import register_task
from minilab.verifiers import ToolCallVerifier, extract_json_object, restricted_python_unit_test_result


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _masked_decode(tokenizer, tokens, mask):
    return tokenizer.decode(tokens[mask].tolist())


def format_answer_components(text, expected):
    match = ANSWER_PATTERN.search(text)
    format_score = 1.0 if match else 0.0
    answer = match.group(1).strip() if match else ""
    answer_score = 1.0 if answer == str(expected) else 0.0
    return {
        "format": format_score,
        "answer": answer_score,
        "reward": 1.0 if format_score and answer_score else 0.0,
    }


def mini_arithmetic_components(text, expected):
    match = re.search(r"(?<!\d)-?\d+", text)
    numeric = 1.0 if match else 0.0
    answer = match.group() if match else ""
    correct = 1.0 if answer == str(expected) else 0.0
    return {"numeric": numeric, "answer": correct, "reward": correct}


def tool_call_components(text, expected):
    def add(a, b):
        return int(a) + int(b)

    verifier = ToolCallVerifier({"add": add})
    valid_json = 0.0
    try:
        call = json.loads(extract_json_object(text))
        valid_json = 1.0 if isinstance(call, dict) else 0.0
    except (json.JSONDecodeError, ValueError):
        valid_json = 0.0
    reward = verifier(text, str(expected))
    return {"json": valid_json, "tool": reward, "reward": reward}


def tiny_code_repair_components(text, tests):
    result = restricted_python_unit_test_result(text, tests, timeout_seconds=1.0)
    result["reward"] = 1.0 if result["syntax"] and result["timeout_free"] and result["unit_tests"] else 0.0
    return result


def _component_batch(component_fn, tokenizer, answers, batch, completions, completion_mask):
    require(completions.dim() == 2, "toy verifier completions must have shape (batch, seq)")
    require(completions.size(0) > 0, "toy verifier reward batch requires at least one completion")
    require(completion_mask.shape == completions.shape, "toy verifier completion_mask must match completions")
    require(completion_mask.dtype == torch.bool, "toy verifier completion_mask must be bool")
    require("idx" in batch, "toy verifier reward batch is missing idx")
    require(batch["idx"].shape == (completions.size(0),), "toy verifier idx must have shape (batch,)")
    require(((0 <= batch["idx"]) & (batch["idx"] < len(answers))).all(), (
        "toy verifier idx values must reference answers"
    ))
    rows = [
        component_fn(
            _masked_decode(tokenizer, completions[b], completion_mask[b]),
            answers[batch["idx"][b].item()],
        )
        for b in range(completions.size(0))
    ]
    keys = rows[0].keys()
    return {
        key: torch.tensor([row[key] for row in rows], device=completions.device, dtype=torch.float32)
        for key in keys
    }


def format_answer_batch_reward(tokenizer, answers, batch, completions, completion_mask):
    return _component_batch(format_answer_components, tokenizer, answers, batch, completions, completion_mask)


def mini_arithmetic_batch_reward(tokenizer, answers, batch, completions, completion_mask):
    return _component_batch(mini_arithmetic_components, tokenizer, answers, batch, completions, completion_mask)


def tool_call_batch_reward(tokenizer, answers, batch, completions, completion_mask):
    return _component_batch(tool_call_components, tokenizer, answers, batch, completions, completion_mask)


def tiny_code_repair_batch_reward(tokenizer, answers, batch, completions, completion_mask):
    return _component_batch(tiny_code_repair_components, tokenizer, answers, batch, completions, completion_mask)


class PromptAnswerDataset(Dataset):
    def __init__(self, tokenizer, prompts, answers, seq_len):
        require(len(prompts) == len(answers), "prompts and answers must match")
        require(seq_len > 0, "PromptAnswerDataset requires seq_len > 0")
        self.answers = list(answers)
        self.rows = []
        for idx, prompt in enumerate(prompts):
            ids = tokenizer.encode(prompt)
            require(type(ids) is list and all(type(token_id) is int and token_id >= 0 for token_id in ids), (
                "PromptAnswerDataset tokenizer must return a list of non-negative integer token ids"
            ))
            require(ids, "toy verifier prompt encoded to an empty sequence")
            require(len(ids) <= seq_len, (
                f"toy verifier prompt len {len(ids)} exceeds seq_len {seq_len}; "
                "caller must choose a larger context or own task-specific truncation"
            ))
            self.rows.append({
                "prompt_ids": torch.tensor(ids + [0] * (seq_len - len(ids)), dtype=torch.long),
                "prompt_len": torch.tensor(len(ids), dtype=torch.long),
                "idx": torch.tensor(idx, dtype=torch.long),
            })
        require(self.rows, "PromptAnswerDataset received no examples")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def make_format_answer_dataset(tokenizer, seq_len, count=128):
    prompts = [f"Return exactly <answer>{i % 17}</answer>." for i in range(count)]
    answers = [str(i % 17) for i in range(count)]
    return PromptAnswerDataset(tokenizer, prompts, answers, seq_len)


def make_mini_arithmetic_dataset(tokenizer, seq_len, count=128):
    prompts = []
    answers = []
    for i in range(count):
        a = i % 13
        b = (i * 7) % 11
        prompts.append(f"What is {a} + {b}? Answer with only the number.")
        answers.append(str(a + b))
    return PromptAnswerDataset(tokenizer, prompts, answers, seq_len)


def make_tool_call_dataset(tokenizer, seq_len, count=128):
    prompts = []
    answers = []
    for i in range(count):
        a = i % 9
        b = (i * 5) % 8
        prompts.append(
            "Emit one JSON object with keys tool and arguments. "
            f"Call add with a={a}, b={b}."
        )
        answers.append(str(a + b))
    return PromptAnswerDataset(tokenizer, prompts, answers, seq_len)


def make_tiny_code_repair_dataset(tokenizer, seq_len, count=128):
    templates = [
        (
            "Replace the buggy function. Return only Python code.\n\n"
            "def add_one(x):\n    return x\n\n"
            "Visible tests:\nassert add_one(1) == 2\n",
            "assert add_one(0) == 1\nassert add_one(-3) == -2\n",
        ),
        (
            "Replace the buggy function. Return only Python code.\n\n"
            "def is_even(x):\n    return True\n\n"
            "Visible tests:\nassert is_even(2) is True\nassert is_even(3) is False\n",
            "assert is_even(0) is True\nassert is_even(11) is False\n",
        ),
        (
            "Replace the buggy function. Return only Python code.\n\n"
            "def first_char(text):\n    return text\n\n"
            "Visible tests:\nassert first_char('abc') == 'a'\n",
            "assert first_char('z') == 'z'\nassert first_char('hello') == 'h'\n",
        ),
        (
            "Replace the buggy function. Return only Python code.\n\n"
            "def max2(a, b):\n    return a\n\n"
            "Visible tests:\nassert max2(3, 2) == 3\nassert max2(2, 3) == 3\n",
            "assert max2(-1, -5) == -1\nassert max2(7, 7) == 7\n",
        ),
    ]
    prompts = []
    answers = []
    for i in range(count):
        prompt, tests = templates[i % len(templates)]
        prompts.append(prompt)
        answers.append(tests)
    return PromptAnswerDataset(tokenizer, prompts, answers, seq_len)


@register_task("format_answer")
class FormatAnswerTask:
    components = staticmethod(format_answer_components)
    batch_reward = staticmethod(format_answer_batch_reward)
    make_dataset = staticmethod(make_format_answer_dataset)


@register_task("mini_arithmetic")
class MiniArithmeticTask:
    components = staticmethod(mini_arithmetic_components)
    batch_reward = staticmethod(mini_arithmetic_batch_reward)
    make_dataset = staticmethod(make_mini_arithmetic_dataset)


@register_task("tool_call_json")
class ToolCallJSONTask:
    components = staticmethod(tool_call_components)
    batch_reward = staticmethod(tool_call_batch_reward)
    make_dataset = staticmethod(make_tool_call_dataset)


@register_task("tiny_code_repair")
class TinyCodeRepairTask:
    components = staticmethod(tiny_code_repair_components)
    batch_reward = staticmethod(tiny_code_repair_batch_reward)
    make_dataset = staticmethod(make_tiny_code_repair_dataset)
