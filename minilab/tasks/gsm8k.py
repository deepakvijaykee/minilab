"""GSM8K task: prompt template, answer extractor, and reward are colocated here
so the `####` format contract lives in one place and cannot drift between the
policy's training prompt and the reward function."""

import re

from minilab.evaluation import accuracy_reward

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


def reward(completion_text, expected):
    predicted = extract_answer(completion_text)
    return accuracy_reward(predicted, expected) if predicted is not None else 0.0
