"""Optional Reasoning Gym adapter for deterministic procedural RLVR tasks."""

import importlib
import json
import math
import re
from dataclasses import dataclass

import torch

from minilab.checks import require
from minilab.registry import register_task
from minilab.tasks.verifier_toys import PromptAnswerDataset


ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
ANSWER_INSTRUCTION = "Return the final answer inside <answer>...</answer>."


def require_reasoning_gym():
    """Import the optional dependency without making it a core requirement."""
    try:
        module = importlib.import_module("reasoning_gym")
    except ModuleNotFoundError as exc:
        if exc.name != "reasoning_gym":
            raise
        raise ValueError(
            "Reasoning Gym tasks require the optional 'reasoning' extra: "
            "python -m pip install -e '.[reasoning]'"
        ) from exc
    return module


def parse_reasoning_gym_config(value, context):
    """Parse a task-specific JSON object while keeping split ownership explicit."""
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context} must be a JSON object") from exc
    require(isinstance(value, dict), f"{context} must be a JSON object")
    reserved = sorted(set(value) & {"seed", "size"})
    require(not reserved, f"{context} cannot override split-owned fields: {reserved}")
    return dict(value)


def extract_reasoning_gym_answer(text):
    match = ANSWER_PATTERN.search(text)
    return match.group(1).strip() if match else None


@dataclass(frozen=True)
class ReasoningGymReference:
    dataset: object
    entry: dict

    @property
    def expected(self):
        return str(self.entry["answer"])

    def __str__(self):
        return self.expected


def reasoning_gym_components(text, reference):
    require(isinstance(reference, ReasoningGymReference), (
        "Reasoning Gym reward requires a ReasoningGymReference"
    ))
    answer = extract_reasoning_gym_answer(text)
    format_score = 1.0 if answer is not None else 0.0
    if answer is None:
        answer_score = 0.0
    else:
        answer_score = float(reference.dataset.score_answer(answer=answer, entry=reference.entry))
        require(math.isfinite(answer_score), "Reasoning Gym score_answer must return a finite score")
    return {"format": format_score, "answer": answer_score, "reward": answer_score}


def reasoning_gym_batch_reward(tokenizer, answers, batch, completions, completion_mask):
    require(completions.dim() == 2, "Reasoning Gym completions must have shape (batch, seq)")
    require(completions.size(0) > 0, "Reasoning Gym reward batch requires at least one completion")
    require(completion_mask.shape == completions.shape, (
        "Reasoning Gym completion_mask must match completions"
    ))
    require(completion_mask.dtype == torch.bool, "Reasoning Gym completion_mask must be bool")
    require("idx" in batch, "Reasoning Gym reward batch is missing idx")
    require(batch["idx"].shape == (completions.size(0),), (
        "Reasoning Gym idx must have shape (batch,)"
    ))
    require(((0 <= batch["idx"]) & (batch["idx"] < len(answers))).all(), (
        "Reasoning Gym idx values must reference answers"
    ))
    rows = [
        reasoning_gym_components(
            tokenizer.decode(completions[b][completion_mask[b]].tolist()),
            answers[batch["idx"][b].item()],
        )
        for b in range(completions.size(0))
    ]
    return {
        key: torch.tensor([row[key] for row in rows], device=completions.device, dtype=torch.float32)
        for key in rows[0]
    }


class ReasoningGymPromptDataset(PromptAnswerDataset):
    def __init__(self, tokenizer, seq_len, task_name, count, seed, config=None):
        require(isinstance(task_name, str) and task_name.strip(), (
            "Reasoning Gym task name must be a non-empty string"
        ))
        require(type(count) is int and count > 0, "Reasoning Gym count must be a positive integer")
        require(type(seed) is int, "Reasoning Gym seed must be an integer")
        config = parse_reasoning_gym_config(config, "Reasoning Gym config")
        source = require_reasoning_gym().create_dataset(
            task_name.strip(), size=count, seed=seed, **config
        )
        entries = list(source)
        require(len(entries) == count, (
            f"Reasoning Gym requested {count} examples but created {len(entries)}"
        ))
        prompts = []
        references = []
        for entry in entries:
            require(isinstance(entry, dict), "Reasoning Gym entries must be mappings")
            question = entry.get("question")
            require(isinstance(question, str) and question.strip(), (
                "Reasoning Gym entries require a non-empty question"
            ))
            require("answer" in entry, "Reasoning Gym entries require an answer")
            prompts.append(f"{question.rstrip()}\n\n{ANSWER_INSTRUCTION}")
            references.append(ReasoningGymReference(source, entry))
        super().__init__(tokenizer, prompts, references, seq_len)
        self.task_name = task_name.strip()
        self.seed = seed
        self.task_config = config
        self.entries = entries


def make_reasoning_gym_dataset(tokenizer, seq_len, task_name, count, seed, config=None):
    return ReasoningGymPromptDataset(
        tokenizer,
        seq_len,
        task_name=task_name,
        count=count,
        seed=seed,
        config=config,
    )


@register_task("reasoning_gym")
class ReasoningGymTask:
    extract_answer = staticmethod(extract_reasoning_gym_answer)
    components = staticmethod(reasoning_gym_components)
    batch_reward = staticmethod(reasoning_gym_batch_reward)
    make_dataset = staticmethod(make_reasoning_gym_dataset)
