"""GSM8K task: prompt template, answer extractor, and reward are colocated here
so the `####` format contract lives in one place and cannot drift between the
policy's training prompt and the reward function."""

import re

import torch

from minilab.checks import require
from minilab.evaluation import accuracy_reward
from minilab.registry import register_task

DELIMITER = "####"
_NUMBER = r"(?<!\d)-?\d[\d,]*(?:\.\d+)?"


def format_prompt(question):
    body, suffix = prompt_parts(question)
    return body + suffix


def prompt_parts(question):
    """Returns (body, suffix). The suffix carries the format instruction the reward
    depends on, so callers that need to truncate must preserve it verbatim."""
    body = f"Question: {question}"
    suffix = f"\nAnswer (end with '{DELIMITER} <number>'):"
    return body, suffix


def extract_answer(text):
    """Strictly require the '#### <number>' delimiter so multi-number chains of
    thought cannot earn credit by accident via the last-number fallback."""
    if DELIMITER not in text:
        return None
    after = text.split(DELIMITER)[-1]
    match = re.search(_NUMBER, after)
    return match.group() if match else None


def parse_gold_answer(answer_text):
    """Parse the dataset answer format used by GSM8K rows."""
    require(DELIMITER in answer_text, "GSM8K answer is missing the '####' delimiter")
    answer = extract_answer(answer_text)
    require(answer is not None, "GSM8K answer delimiter is not followed by a number")
    return answer


def reward(completion_text, expected):
    return reward_components(completion_text, expected)["reward"]


def reward_components(completion_text, expected):
    predicted = extract_answer(completion_text)
    format_score = 1.0 if predicted is not None else 0.0
    answer_score = accuracy_reward(predicted, expected) if predicted is not None else 0.0
    return {
        "format": format_score,
        "answer": answer_score,
        "reward": answer_score,
    }


def batch_reward(tokenizer, answers, batch, completions, completion_mask):
    require(completions.dim() == 2, "GSM8K completions must have shape (batch, seq)")
    require(completions.size(0) > 0, "GSM8K reward batch requires at least one completion")
    require(completion_mask.shape == completions.shape, "GSM8K completion_mask must match completions")
    require(completion_mask.dtype == torch.bool, "GSM8K completion_mask must be bool")
    require("idx" in batch, "GSM8K reward batch is missing idx")
    require(batch["idx"].shape == (completions.size(0),), "GSM8K idx must have shape (batch,)")
    require(((0 <= batch["idx"]) & (batch["idx"] < len(answers))).all(), (
        "GSM8K idx values must reference answers"
    ))
    rows = [
        reward_components(
            tokenizer.decode(completions[b][completion_mask[b]].tolist()),
            answers[batch["idx"][b].item()],
        )
        for b in range(completions.size(0))
    ]
    return {
        key: torch.tensor([row[key] for row in rows], device=completions.device, dtype=torch.float32)
        for key in rows[0]
    }


@register_task("gsm8k")
class GSM8KTask:
    delimiter = DELIMITER
    format_prompt = staticmethod(format_prompt)
    prompt_parts = staticmethod(prompt_parts)
    extract_answer = staticmethod(extract_answer)
    parse_gold_answer = staticmethod(parse_gold_answer)
    reward = staticmethod(reward)
    reward_components = staticmethod(reward_components)
    batch_reward = staticmethod(batch_reward)
