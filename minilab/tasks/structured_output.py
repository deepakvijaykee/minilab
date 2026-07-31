"""Deterministic exact-envelope curriculum for instruction-model alignment."""

from minilab.checks import require
from minilab.data import SFTDataset
from minilab.tasks.agentic_calculator import (
    AGENTIC_CALCULATOR_CHALLENGE_KINDS,
    agentic_calculator_challenge_cases,
    agentic_calculator_challenge_kind_cases,
    calculator_intermediate_observation,
    calculator_trajectory_cases,
    calculator_tool_call,
    calculator_tool_observation,
    calculator_user_prompt,
    execute_calculator_tool,
)
from minilab.tasks.verifier_toys import RAW_PYTHON_INSTRUCTION


STRUCTURED_OUTPUT_SPLITS = ("train", "validation")
STRUCTURED_OUTPUT_KINDS = ("raw_python", "tool_call", "final_answer")
STRUCTURED_OUTPUT_CURRICULA = (
    "basic",
    "hard",
    "chain_weighted",
    "two_tool",
    "replay_balanced",
    "canonical_replay",
)
STRUCTURED_OUTPUT_CURRICULUM_VERSIONS = {
    "basic": 1,
    "hard": 1,
    "chain_weighted": 1,
    "two_tool": 1,
    "replay_balanced": 1,
    "canonical_replay": 1,
}
HARD_STRUCTURED_OUTPUT_KINDS = (
    "raw_python",
    "tool_call",
    "final_answer",
    "challenge_tool_call",
    "challenge_final_answer",
)
HARD_CODE_TRAIN_FAMILIES = (
    "add_constant",
    "subtract_constant",
    "multiply_constant",
    "divisible_by",
    "prefix_chars",
    "suffix_chars",
    "prefix_items",
    "suffix_items",
    "repeat_text",
    "offset_pair_sum",
    "bounded_between",
    "difference_magnitude",
)
HARD_CODE_VARIANTS_PER_FAMILY = 4
HARD_SFT_CHALLENGE_STARTS = {"train": 10_000, "validation": 30_000}
CHAIN_WEIGHTED_STRUCTURED_OUTPUT_KINDS = (
    "raw_python",
    "tool_call",
    "final_answer",
    "chained_tool_call",
    "chained_final_answer",
)
CHAIN_WEIGHTED_SFT_CHALLENGE_STARTS = {
    "train": 200,
    "validation": 1_000,
}
TWO_TOOL_SFT_TRAJECTORY_STARTS = {"train": 2_000, "validation": 4_000}
TWO_TOOL_STRUCTURED_OUTPUT_SLOTS = (
    ("raw_python", 0),
    ("two_tool_first_call", 0),
    ("two_tool_second_call", 0),
    ("two_tool_final_answer", 0),
    ("raw_python", 1),
    ("two_tool_first_call", 1),
    ("two_tool_second_call", 1),
    ("two_tool_final_answer", 1),
    ("raw_python", 2),
)
REPLAY_BALANCED_STRUCTURED_OUTPUT_SLOTS = (
    ("raw_python", 0),
    ("tool_call", 0),
    ("final_answer", 0),
    ("two_tool_first_call", 0),
    ("two_tool_second_call", 0),
    ("two_tool_final_answer", 0),
    ("raw_python", 1),
    ("tool_call", 1),
    ("final_answer", 1),
    ("two_tool_first_call", 1),
    ("two_tool_second_call", 1),
    ("two_tool_final_answer", 1),
    ("raw_python", 2),
    ("tool_call", 2),
    ("final_answer", 2),
    ("raw_python", 3),
    ("raw_python", 4),
    ("raw_python", 5),
)
CANONICAL_REPLAY_STRUCTURED_OUTPUT_SLOTS = (
    ("raw_python", 0),
    ("tool_call", 0),
    ("final_answer", 0),
    ("two_tool_first_call", 0),
    ("two_tool_second_call", 0),
    ("two_tool_final_answer", 0),
    ("raw_python", 1),
    ("tool_call", 1),
    ("final_answer", 1),
    ("raw_python", 2),
    ("tool_call", 2),
    ("final_answer", 2),
    ("raw_python", 3),
    ("two_tool_first_call", 1),
    ("two_tool_second_call", 1),
    ("two_tool_final_answer", 1),
    ("raw_python", 4),
    ("tool_call", 3),
    ("final_answer", 3),
    ("raw_python", 5),
    ("tool_call", 4),
    ("final_answer", 4),
    ("raw_python", 6),
    ("raw_python", 7),
)

_RAW_PYTHON_EXAMPLES = (
    (
        "def double(x):\n    return x\n\nVisible tests:\nassert double(4) == 8\n",
        "def double(x):\n    return x * 2",
    ),
    (
        "def last_char(text):\n    return text\n\nVisible tests:\nassert last_char('abc') == 'c'\n",
        "def last_char(text):\n    return text[-1]",
    ),
    (
        "def absolute(x):\n    return x\n\nVisible tests:\nassert absolute(-3) == 3\n",
        "def absolute(x):\n    return -x if x < 0 else x",
    ),
    (
        "def min2(a, b):\n    return a\n\nVisible tests:\nassert min2(3, 2) == 2\n",
        "def min2(a, b):\n    return a if a < b else b",
    ),
    (
        "def square(x):\n    return x\n\nVisible tests:\nassert square(5) == 25\n",
        "def square(x):\n    return x * x",
    ),
    (
        "def negate(x):\n    return x\n\nVisible tests:\nassert negate(7) == -7\n",
        "def negate(x):\n    return -x",
    ),
)


def _calculator_values(sample, split):
    offset = 1_000 if split == "train" else 10_000
    return offset + sample, (sample * 11 + 3) % 97


def _raw_python_example(sample):
    prompt, response = _RAW_PYTHON_EXAMPLES[sample % len(_RAW_PYTHON_EXAMPLES)]
    return {
        "kind": "raw_python",
        "messages": [{"role": "user", "content": RAW_PYTHON_INSTRUCTION + prompt}],
        "response": response,
    }


def _hard_python_spec(family, constant):
    require(family in HARD_CODE_TRAIN_FAMILIES, (
        "hard Python example requires a declared training family"
    ))
    require(constant in (2, 3, 4, 5), (
        "hard Python example constant must be one of 2, 3, 4, or 5"
    ))
    if family == "add_constant":
        name, value = f"add_{constant}", 7 + constant
        body = f"def {name}(x):\n    return x"
        test = f"assert {name}(7) == {value}"
        response = f"def {name}(x):\n    return x + {constant}"
    elif family == "subtract_constant":
        name, value = f"subtract_{constant}", 11 - constant
        body = f"def {name}(x):\n    return x"
        test = f"assert {name}(11) == {value}"
        response = f"def {name}(x):\n    return x - {constant}"
    elif family == "multiply_constant":
        name, value = f"multiply_by_{constant}", 6 * constant
        body = f"def {name}(x):\n    return x"
        test = f"assert {name}(6) == {value}"
        response = f"def {name}(x):\n    return x * {constant}"
    elif family == "divisible_by":
        name = f"divisible_by_{constant}"
        body = f"def {name}(x):\n    return False"
        test = f"assert {name}({constant * 3}) is True\nassert {name}({constant * 3 + 1}) is False"
        response = f"def {name}(x):\n    return x % {constant} == 0"
    elif family == "prefix_chars":
        name, text = f"prefix_{constant}_chars", "alphabet"
        body = f"def {name}(text):\n    return text"
        test = f"assert {name}({text!r}) == {text[:constant]!r}"
        response = f"def {name}(text):\n    return text[:{constant}]"
    elif family == "suffix_chars":
        name, text = f"suffix_{constant}_chars", "notebook"
        body = f"def {name}(text):\n    return text"
        test = f"assert {name}({text!r}) == {text[-constant:]!r}"
        response = f"def {name}(text):\n    return text[-{constant}:]"
    elif family == "prefix_items":
        name, values = f"first_{constant}_items", list(range(7))
        body = f"def {name}(values):\n    return values"
        test = f"assert {name}({values!r}) == {values[:constant]!r}"
        response = f"def {name}(values):\n    return values[:{constant}]"
    elif family == "suffix_items":
        name, values = f"last_{constant}_items", list(range(7))
        body = f"def {name}(values):\n    return values"
        test = f"assert {name}({values!r}) == {values[-constant:]!r}"
        response = f"def {name}(values):\n    return values[-{constant}:]"
    elif family == "repeat_text":
        name = f"repeat_text_{constant}"
        body = f"def {name}(text):\n    return text"
        test = f"assert {name}('ab') == {'ab' * constant!r}"
        response = f"def {name}(text):\n    return text * {constant}"
    elif family == "offset_pair_sum":
        name = f"offset_pair_sum_{constant}"
        body = f"def {name}(a, b):\n    return a + b"
        test = f"assert {name}(3, -1) == {2 + constant}"
        response = f"def {name}(a, b):\n    return a + b + {constant}"
    elif family == "bounded_between":
        name = f"between_0_and_{constant}"
        body = f"def {name}(x):\n    return False"
        test = f"assert {name}({constant - 1}) is True\nassert {name}({constant + 1}) is False"
        response = f"def {name}(x):\n    return 0 <= x <= {constant}"
    else:
        name = f"distance_from_{constant}"
        body = f"def {name}(x):\n    return x"
        test = f"assert {name}(-2) == {constant + 2}"
        response = f"def {name}(x):\n    return abs(x - {constant})"
    prompt = f"{body}\n\nVisible tests:\n{test}\n"
    return prompt, response


def _hard_raw_python_example(sample):
    cycle = len(HARD_CODE_TRAIN_FAMILIES) * HARD_CODE_VARIANTS_PER_FAMILY
    offset = sample % cycle
    family = HARD_CODE_TRAIN_FAMILIES[offset // HARD_CODE_VARIANTS_PER_FAMILY]
    constant = 2 + offset % HARD_CODE_VARIANTS_PER_FAMILY
    prompt, response = _hard_python_spec(family, constant)
    return {
        "kind": "raw_python",
        "family": family,
        "messages": [{"role": "user", "content": RAW_PYTHON_INSTRUCTION + prompt}],
        "response": response,
    }


def _tool_call_example(sample, split):
    a, b = _calculator_values(sample, split)
    expected = str(a + b)
    return {
        "kind": "tool_call",
        "messages": [{"role": "user", "content": calculator_user_prompt(a, b)}],
        "response": calculator_tool_call(a, b),
        "expected": expected,
    }


def _final_answer_example(sample, split):
    a, b = _calculator_values(sample, split)
    tool_call = calculator_tool_call(a, b)
    execution = execute_calculator_tool(tool_call)
    expected = str(a + b)
    require(execution.result == expected, (
        "structured-output calculator target must execute to the expected result"
    ))
    return {
        "kind": "final_answer",
        "messages": [
            {"role": "user", "content": calculator_user_prompt(a, b)},
            {"role": "assistant", "content": tool_call},
            {"role": "user", "content": calculator_tool_observation(execution)},
        ],
        "response": f"<answer>{expected}</answer>",
        "expected": expected,
    }


def _challenge_example(sample, split, kind):
    challenge_index = HARD_SFT_CHALLENGE_STARTS[split] + sample
    case = agentic_calculator_challenge_cases(1, start=challenge_index)[0]
    a, b = case.target.arguments
    tool_call = calculator_tool_call(a, b)
    if kind == "challenge_tool_call":
        messages = [{"role": "user", "content": case.prompt}]
        response = tool_call
    else:
        execution = execute_calculator_tool(tool_call)
        require(execution.result == case.target.result, (
            "hard structured-output target must execute to its expected result"
        ))
        messages = [
            {"role": "user", "content": case.prompt},
            {"role": "assistant", "content": tool_call},
            {"role": "user", "content": calculator_tool_observation(execution)},
        ]
        response = f"<answer>{case.target.result}</answer>"
    return {
        "kind": kind,
        "challenge_kind": case.kind,
        "challenge_index": challenge_index,
        "messages": messages,
        "response": response,
        "expected": case.target.result,
    }


def _chained_example(sample, split, kind):
    challenge_sample = CHAIN_WEIGHTED_SFT_CHALLENGE_STARTS[split] + sample
    case = agentic_calculator_challenge_kind_cases(
        "chained", 1, start=challenge_sample
    )[0]
    stride = len(AGENTIC_CALCULATOR_CHALLENGE_KINDS)
    challenge_index = (
        challenge_sample * stride
        + AGENTIC_CALCULATOR_CHALLENGE_KINDS.index("chained")
    )
    a, b = case.target.arguments
    tool_call = calculator_tool_call(a, b)
    if kind == "chained_tool_call":
        messages = [{"role": "user", "content": case.prompt}]
        response = tool_call
    else:
        execution = execute_calculator_tool(tool_call)
        require(execution.result == case.target.result, (
            "chain-weighted target must execute to its expected result"
        ))
        messages = [
            {"role": "user", "content": case.prompt},
            {"role": "assistant", "content": tool_call},
            {"role": "user", "content": calculator_tool_observation(execution)},
        ]
        response = f"<answer>{case.target.result}</answer>"
    return {
        "kind": kind,
        "challenge_kind": case.kind,
        "challenge_index": challenge_index,
        "messages": messages,
        "response": response,
        "expected": case.target.result,
        "target_arguments": list(case.target.arguments),
    }


def _two_tool_example(sample, split, kind):
    trajectory_sample = TWO_TOOL_SFT_TRAJECTORY_STARTS[split] + sample
    case = calculator_trajectory_cases(1, start=trajectory_sample)[0]
    target = case.target
    first_call = calculator_tool_call(*target.first_arguments)
    first_execution = execute_calculator_tool(first_call)
    require(first_execution.result == target.intermediate, (
        "two-tool first target must execute to its intermediate"
    ))
    first_observation = calculator_intermediate_observation(
        first_execution, target.second_arguments[1]
    )
    if kind == "two_tool_first_call":
        messages = [{"role": "user", "content": case.prompt}]
        response = first_call
    else:
        second_call = calculator_tool_call(*target.second_arguments)
        second_execution = execute_calculator_tool(second_call)
        require(second_execution.result == target.result, (
            "two-tool second target must execute to its final result"
        ))
        messages = [
            {"role": "user", "content": case.prompt},
            {"role": "assistant", "content": first_call},
            {"role": "user", "content": first_observation},
        ]
        if kind == "two_tool_second_call":
            response = second_call
        else:
            messages.extend([
                {"role": "assistant", "content": second_call},
                {
                    "role": "user",
                    "content": calculator_tool_observation(second_execution),
                },
            ])
            response = f"<answer>{target.result}</answer>"
    return {
        "kind": kind,
        "trajectory_sample": trajectory_sample,
        "messages": messages,
        "response": response,
        "expected": target.result,
        "first_arguments": list(target.first_arguments),
        "second_arguments": list(target.second_arguments),
    }


def _two_tool_structured_output_example(index, split):
    group = index // len(TWO_TOOL_STRUCTURED_OUTPUT_SLOTS)
    kind, slot = TWO_TOOL_STRUCTURED_OUTPUT_SLOTS[
        index % len(TWO_TOOL_STRUCTURED_OUTPUT_SLOTS)
    ]
    if kind == "raw_python":
        return _raw_python_example(group * 3 + slot)
    return _two_tool_example(group * 2 + slot, split, kind)


def _replay_balanced_structured_output_example(index, split):
    group = index // len(REPLAY_BALANCED_STRUCTURED_OUTPUT_SLOTS)
    kind, slot = REPLAY_BALANCED_STRUCTURED_OUTPUT_SLOTS[
        index % len(REPLAY_BALANCED_STRUCTURED_OUTPUT_SLOTS)
    ]
    if kind == "raw_python":
        return _raw_python_example(group * 6 + slot)
    if kind == "tool_call":
        return _tool_call_example(group * 3 + slot, split)
    if kind == "final_answer":
        return _final_answer_example(group * 3 + slot, split)
    return _two_tool_example(group * 2 + slot, split, kind)


def _canonical_replay_structured_output_example(index, split):
    group = index // len(CANONICAL_REPLAY_STRUCTURED_OUTPUT_SLOTS)
    kind, slot = CANONICAL_REPLAY_STRUCTURED_OUTPUT_SLOTS[
        index % len(CANONICAL_REPLAY_STRUCTURED_OUTPUT_SLOTS)
    ]
    if kind == "raw_python":
        return _raw_python_example(group * 8 + slot)
    if kind == "tool_call":
        return _tool_call_example(group * 5 + slot, split)
    if kind == "final_answer":
        return _final_answer_example(group * 5 + slot, split)
    return _two_tool_example(group * 2 + slot, split, kind)


def structured_output_examples(count, split="train", curriculum="basic"):
    require(type(count) is int and count > 0, (
        "structured-output example count must be a positive integer"
    ))
    require(split in STRUCTURED_OUTPUT_SPLITS, (
        f"structured-output split must be one of {STRUCTURED_OUTPUT_SPLITS}"
    ))
    require(curriculum in STRUCTURED_OUTPUT_CURRICULA, (
        f"structured-output curriculum must be one of {STRUCTURED_OUTPUT_CURRICULA}"
    ))
    if curriculum in {"two_tool", "replay_balanced", "canonical_replay"}:
        builders = {
            "two_tool": _two_tool_structured_output_example,
            "replay_balanced": _replay_balanced_structured_output_example,
            "canonical_replay": _canonical_replay_structured_output_example,
        }
        builder = builders[curriculum]
        return [
            builder(index, split)
            for index in range(count)
        ]
    if curriculum == "basic":
        kinds = STRUCTURED_OUTPUT_KINDS
    elif curriculum == "hard":
        kinds = HARD_STRUCTURED_OUTPUT_KINDS
    else:
        kinds = CHAIN_WEIGHTED_STRUCTURED_OUTPUT_KINDS
    examples = []
    for index in range(count):
        kind = kinds[index % len(kinds)]
        sample = index // len(kinds)
        if kind == "raw_python":
            example = (
                _raw_python_example(sample)
                if curriculum != "hard"
                else _hard_raw_python_example(sample)
            )
            examples.append(example)
        elif kind == "tool_call":
            examples.append(_tool_call_example(sample, split))
        elif kind == "final_answer":
            examples.append(_final_answer_example(sample, split))
        elif kind.startswith("chained_"):
            examples.append(_chained_example(sample, split, kind))
        else:
            examples.append(_challenge_example(sample, split, kind))
    return examples


def make_structured_output_sft_dataset(
    tokenizer, seq_len, count=384, split="train", curriculum="basic"
):
    return SFTDataset(
        structured_output_examples(count, split=split, curriculum=curriculum),
        tokenizer,
        seq_len,
        allow_truncation=False,
    )
