"""Deterministic calculator environments for agentic probes."""

import json
import re
from dataclasses import dataclass

from minilab.checks import require
from minilab.registry import register_task
from minilab.tasks.verifier_toys import PromptAnswerDataset


INVALID_TOOL_OBSERVATION = "ERROR: invalid add tool call"
AGENTIC_CALCULATOR_CHALLENGE_KINDS = (
    "signed", "distractor", "chained", "recovery"
)
AGENTIC_CALCULATOR_MIXTURE_KINDS = (
    "canonical", "signed", "distractor", "chained", "recovery"
)
AGENTIC_CALCULATOR_MIXTURE_CONTRACT = "balanced_canonical_challenge_v1"


@dataclass(frozen=True)
class CalculatorTarget:
    result: str
    arguments: tuple[int, int]

    def __post_init__(self):
        require(type(self.result) is str and self.result, (
            "calculator target result must be a non-empty string"
        ))
        require(
            type(self.arguments) is tuple
            and len(self.arguments) == 2
            and all(type(value) is int for value in self.arguments),
            "calculator target arguments must be a pair of integers",
        )
        require(self.result == str(sum(self.arguments)), (
            "calculator target result must equal the sum of its arguments"
        ))


@dataclass(frozen=True)
class CalculatorCase:
    kind: str
    prompt: str
    target: CalculatorTarget

    def __post_init__(self):
        require(type(self.kind) is str and self.kind, (
            "calculator case kind must be a non-empty string"
        ))
        require(type(self.prompt) is str and self.prompt, (
            "calculator case prompt must be a non-empty string"
        ))
        require(isinstance(self.target, CalculatorTarget), (
            "calculator case requires a CalculatorTarget"
        ))


@dataclass(frozen=True)
class CalculatorTrajectoryTarget:
    result: str
    first_arguments: tuple[int, int]
    second_arguments: tuple[int, int]

    def __post_init__(self):
        require(type(self.result) is str and self.result, (
            "calculator trajectory result must be a non-empty string"
        ))
        for name, arguments in (
            ("first", self.first_arguments),
            ("second", self.second_arguments),
        ):
            require(
                type(arguments) is tuple
                and len(arguments) == 2
                and all(type(value) is int for value in arguments),
                f"calculator trajectory {name} arguments must be two integers",
            )
        require(sum(self.first_arguments) == self.second_arguments[0], (
            "calculator trajectory second call must consume the first result"
        ))
        require(self.result == str(sum(self.second_arguments)), (
            "calculator trajectory result must equal the second call result"
        ))

    @property
    def intermediate(self):
        return str(sum(self.first_arguments))


@dataclass(frozen=True)
class CalculatorTrajectoryCase:
    prompt: str
    target: CalculatorTrajectoryTarget

    def __post_init__(self):
        require(type(self.prompt) is str and self.prompt, (
            "calculator trajectory prompt must be a non-empty string"
        ))
        require(isinstance(self.target, CalculatorTrajectoryTarget), (
            "calculator trajectory case requires a CalculatorTrajectoryTarget"
        ))


def calculator_tool_call(a, b):
    require(type(a) is int and type(b) is int, (
        "calculator tool-call targets require integer arguments"
    ))
    return (
        f'<tool_call>{{"tool":"add","arguments":'
        f'{{"a":{a},"b":{b}}}}}</tool_call>'
    )


def calculator_user_prompt(a, b):
    require(type(a) is int and type(b) is int, (
        "calculator prompts require integer arguments"
    ))
    return (
        f"Use the calculator to add {a} and {b}. Your entire first response "
        "must be exactly one "
        '<tool_call>{"tool":"add","arguments":{"a":A,"b":B}}'
        "</tool_call> with A and B replaced by the given integers. Do not "
        "include Markdown, dialogue roles, or prose. Wait for the tool "
        "result before answering."
    )


def calculator_trajectory_prompt(x, y, b):
    require(all(type(value) is int for value in (x, y, b)), (
        "calculator trajectory prompts require integer arguments"
    ))
    return (
        f"Use the calculator twice to compute ({x} + ({y})) + {b}. First call "
        f"the add tool with exactly {x} as A and {y} as B. After its result, "
        f"call the add tool again with that result as A and {b} as B. Your "
        "entire first response and entire second response must each be exactly "
        'one <tool_call>{"tool":"add","arguments":{"a":A,"b":B}}'
        "</tool_call> with integer A and B. Do not include Markdown, dialogue "
        "roles, or prose. Wait for the second tool result before answering."
    )


@dataclass(frozen=True)
class CalculatorToolExecution:
    observation: str
    result: str | None
    arguments: tuple[int, int] | None
    json_valid: float
    tool_valid: float
    arguments_valid: float


def execute_calculator_tool(text):
    """Parse and execute the one safe tool exposed by the environment."""
    text = text.strip()
    prefix = "<tool_call>"
    suffix = "</tool_call>"
    if not text.startswith(prefix) or not text.endswith(suffix):
        return CalculatorToolExecution(
            INVALID_TOOL_OBSERVATION, None, None, 0.0, 0.0, 0.0
        )
    payload = text[len(prefix) : -len(suffix)].strip()
    try:
        call = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return CalculatorToolExecution(
            INVALID_TOOL_OBSERVATION, None, None, 0.0, 0.0, 0.0
        )
    if not isinstance(call, dict):
        return CalculatorToolExecution(
            INVALID_TOOL_OBSERVATION, None, None, 0.0, 0.0, 0.0
        )

    json_valid = 1.0
    if call.get("tool") != "add":
        return CalculatorToolExecution(
            INVALID_TOOL_OBSERVATION, None, None, json_valid, 0.0, 0.0
        )
    arguments = call.get("arguments")
    if not isinstance(arguments, dict) or set(arguments) != {"a", "b"}:
        return CalculatorToolExecution(
            INVALID_TOOL_OBSERVATION, None, None, json_valid, 1.0, 0.0
        )
    if type(arguments["a"]) is not int or type(arguments["b"]) is not int:
        return CalculatorToolExecution(
            INVALID_TOOL_OBSERVATION, None, None, json_valid, 1.0, 0.0
        )
    parsed_arguments = (arguments["a"], arguments["b"])
    result = str(sum(parsed_arguments))
    return CalculatorToolExecution(
        result, result, parsed_arguments, json_valid, 1.0, 1.0
    )


def calculator_tool_observation(execution):
    require(isinstance(execution, CalculatorToolExecution), (
        "calculator observation requires a CalculatorToolExecution"
    ))
    return (
        f"\n<tool_result>{execution.observation}</tool_result>\n"
        "Your entire next response must be exactly <answer>RESULT</answer>, "
        "with RESULT replaced by the tool result. Do not include Markdown, "
        "dialogue roles, or prose."
    )


def calculator_intermediate_observation(execution, second_argument):
    require(isinstance(execution, CalculatorToolExecution), (
        "calculator intermediate observation requires a tool execution"
    ))
    require(type(second_argument) is int, (
        "calculator intermediate observation requires an integer second argument"
    ))
    return (
        f"\n<tool_result>{execution.observation}</tool_result>\n"
        "Your entire next response must be exactly one "
        '<tool_call>{"tool":"add","arguments":{"a":A,"b":B}}'
        f"</tool_call> with A replaced by the tool result and B replaced by "
        f"{second_argument}. Do not include Markdown, dialogue roles, or prose. "
        "Wait for the next tool result before answering."
    )


def calculator_agent_components(execution, answer_text, target):
    require(isinstance(execution, CalculatorToolExecution), (
        "calculator agent reward requires a CalculatorToolExecution"
    ))
    require(isinstance(target, CalculatorTarget), (
        "calculator agent reward requires a CalculatorTarget"
    ))
    match = re.fullmatch(
        r"\s*<answer>\s*([^<]*?)\s*</answer>\s*", answer_text, re.DOTALL
    )
    answer_format = 1.0 if match else 0.0
    answer = match.group(1).strip() if match else ""
    arguments_match = 1.0 if execution.arguments == target.arguments else 0.0
    tool_result = 1.0 if execution.result == target.result else 0.0
    answer_correct = 1.0 if answer == target.result else 0.0
    grounded = 1.0 if execution.result is not None and answer == execution.result else 0.0
    reward = min(arguments_match, tool_result, answer_correct, grounded)
    return {
        "json": execution.json_valid,
        "tool": execution.tool_valid,
        "arguments": execution.arguments_valid,
        "arguments_match": arguments_match,
        "tool_result": tool_result,
        "answer_format": answer_format,
        "answer": answer_correct,
        "grounded": grounded,
        "reward": reward,
    }


def calculator_trajectory_components(
    first_execution,
    second_execution,
    answer_text,
    target,
):
    require(isinstance(first_execution, CalculatorToolExecution), (
        "calculator trajectory reward requires a first tool execution"
    ))
    require(isinstance(second_execution, CalculatorToolExecution), (
        "calculator trajectory reward requires a second tool execution"
    ))
    require(isinstance(target, CalculatorTrajectoryTarget), (
        "calculator trajectory reward requires a CalculatorTrajectoryTarget"
    ))
    match = re.fullmatch(
        r"\s*<answer>\s*([^<]*?)\s*</answer>\s*", answer_text, re.DOTALL
    )
    answer_format = 1.0 if match else 0.0
    answer = match.group(1).strip() if match else ""
    first_arguments_match = float(
        first_execution.arguments == target.first_arguments
    )
    first_tool_result = float(first_execution.result == target.intermediate)
    second_arguments_match = float(
        second_execution.arguments == target.second_arguments
    )
    second_tool_result = float(second_execution.result == target.result)
    answer_correct = float(answer == target.result)
    grounded = float(
        second_execution.result is not None
        and answer == second_execution.result
    )
    reward = min(
        first_arguments_match,
        first_tool_result,
        second_arguments_match,
        second_tool_result,
        answer_correct,
        grounded,
    )
    return {
        "first_json": first_execution.json_valid,
        "first_tool": first_execution.tool_valid,
        "first_arguments": first_execution.arguments_valid,
        "first_arguments_match": first_arguments_match,
        "first_tool_result": first_tool_result,
        "second_json": second_execution.json_valid,
        "second_tool": second_execution.tool_valid,
        "second_arguments": second_execution.arguments_valid,
        "second_arguments_match": second_arguments_match,
        "second_tool_result": second_tool_result,
        "answer_format": answer_format,
        "answer": answer_correct,
        "grounded": grounded,
        "reward": reward,
    }


class AgenticCalculatorDataset(PromptAnswerDataset):
    def __init__(self, tokenizer, cases, seq_len):
        cases = list(cases)
        require(cases and all(isinstance(case, CalculatorCase) for case in cases), (
            "agentic calculator dataset requires CalculatorCase examples"
        ))
        self.targets = [case.target for case in cases]
        self.kinds = [case.kind for case in cases]
        super().__init__(
            tokenizer,
            [case.prompt for case in cases],
            [case.target.result for case in cases],
            seq_len,
            include_raw_prompt=True,
        )


class CalculatorTrajectoryDataset(PromptAnswerDataset):
    def __init__(self, tokenizer, cases, seq_len):
        cases = list(cases)
        require(
            cases
            and all(isinstance(case, CalculatorTrajectoryCase) for case in cases),
            "calculator trajectory dataset requires trajectory cases",
        )
        self.targets = [case.target for case in cases]
        self.kinds = ["two_tool"] * len(cases)
        super().__init__(
            tokenizer,
            [case.prompt for case in cases],
            [case.target.result for case in cases],
            seq_len,
            include_raw_prompt=True,
        )


def _require_case_range(count, start):
    require(type(count) is int and count > 0, (
        "agentic calculator count must be a positive integer"
    ))
    require(type(start) is int and start >= 0, (
        "agentic calculator start must be a non-negative integer"
    ))


def _canonical_case(index):
    a = index
    b = (index * 7) % 13
    return CalculatorCase(
        "canonical",
        calculator_user_prompt(a, b),
        CalculatorTarget(str(a + b), (a, b)),
    )


def agentic_calculator_cases(count=128, start=0):
    _require_case_range(count, start)
    return [_canonical_case(i) for i in range(start, start + count)]


def _challenge_case(index):
    kind = AGENTIC_CALCULATOR_CHALLENGE_KINDS[
        index % len(AGENTIC_CALCULATOR_CHALLENGE_KINDS)
    ]
    sample = index // len(AGENTIC_CALCULATOR_CHALLENGE_KINDS)
    contract = (
        "Your entire first response must be exactly one "
        '<tool_call>{"tool":"add","arguments":{"a":A,"b":B}}'
        "</tool_call> with A and B replaced by the requested integers. Do not "
        "include Markdown, dialogue roles, or prose. Wait for the tool result "
        "before answering."
    )
    if kind == "signed":
        a = -(101 + sample * 3)
        b = 37 + sample * 5
        prompt = f"Use the calculator to add signed integers {a} and {b}. {contract}"
    elif kind == "distractor":
        a = 211 + sample * 7
        b = -(19 + sample * 2)
        reference = 9_000 + sample
        prompt = (
            f"Reference ticket {reference} is not an operand. Use the calculator "
            f"to add only {a} and {b}. {contract}"
        )
    elif kind == "chained":
        x, y, b = _chained_values(sample)
        a = x + y
        prompt = (
            f"First compute {x} + ({y}), then use that intermediate value as A "
            f"and {b} as B in the calculator add call. {contract}"
        )
    else:
        a = 503 + sample * 11
        b = 29 + sample * 3
        wrong = calculator_tool_call(a, b + 1)
        prompt = (
            f"A previous attempt was rejected because it used the wrong second "
            f"operand: {wrong}. Recover by calling the calculator with exactly "
            f"{a} and {b}. {contract}"
        )
    return CalculatorCase(
        kind,
        prompt,
        CalculatorTarget(str(a + b), (a, b)),
    )


def _chained_values(sample):
    require(type(sample) is int and sample >= 0, (
        "chained calculator sample must be a non-negative integer"
    ))
    return 31 + sample * 2, -(7 + sample), 13 + sample * 3


def calculator_trajectory_cases(count=16, start=0):
    """Return deterministic two-call cases in a per-trajectory namespace."""
    _require_case_range(count, start)
    cases = []
    for sample in range(start, start + count):
        x, y, b = _chained_values(sample)
        intermediate = x + y
        cases.append(CalculatorTrajectoryCase(
            calculator_trajectory_prompt(x, y, b),
            CalculatorTrajectoryTarget(
                str(intermediate + b),
                (x, y),
                (intermediate, b),
            ),
        ))
    return cases


def agentic_calculator_challenge_cases(count=16, start=0):
    _require_case_range(count, start)
    return [_challenge_case(i) for i in range(start, start + count)]


def agentic_calculator_challenge_kind_cases(kind, count=16, start=0):
    """Return one challenge family from a deterministic per-kind namespace."""
    require(kind in AGENTIC_CALCULATOR_CHALLENGE_KINDS, (
        f"calculator challenge kind must be one of "
        f"{AGENTIC_CALCULATOR_CHALLENGE_KINDS}"
    ))
    _require_case_range(count, start)
    slot = AGENTIC_CALCULATOR_CHALLENGE_KINDS.index(kind)
    stride = len(AGENTIC_CALCULATOR_CHALLENGE_KINDS)
    return [_challenge_case((start + i) * stride + slot) for i in range(count)]


def _mixture_case(index):
    slot = index % len(AGENTIC_CALCULATOR_MIXTURE_KINDS)
    if slot == 0:
        return _canonical_case(index)
    # Give the mixed stream its own deterministic challenge-index namespace.
    # Multiplication by four keeps every mixed index disjoint from its neighbors
    # while the slot selects one of the four challenge families.
    return _challenge_case(index * 4 + slot - 1)


def agentic_calculator_mixture_cases(count=20, start=0):
    """Return a deterministic five-way canonical/challenge curriculum."""
    _require_case_range(count, start)
    return [_mixture_case(i) for i in range(start, start + count)]


def make_agentic_calculator_dataset(tokenizer, seq_len, count=128, start=0):
    return AgenticCalculatorDataset(
        tokenizer, agentic_calculator_cases(count, start), seq_len
    )


def make_agentic_calculator_challenge_dataset(
    tokenizer, seq_len, count=16, start=0
):
    return AgenticCalculatorDataset(
        tokenizer, agentic_calculator_challenge_cases(count, start), seq_len
    )


def make_agentic_calculator_mixture_dataset(
    tokenizer, seq_len, count=20, start=0
):
    return AgenticCalculatorDataset(
        tokenizer, agentic_calculator_mixture_cases(count, start), seq_len
    )


def make_calculator_trajectory_dataset(
    tokenizer, seq_len, count=16, start=0
):
    return CalculatorTrajectoryDataset(
        tokenizer, calculator_trajectory_cases(count, start), seq_len
    )


@register_task("agentic_calculator")
class AgenticCalculatorTask:
    execute = staticmethod(execute_calculator_tool)
    components = staticmethod(calculator_agent_components)
    make_dataset = staticmethod(make_agentic_calculator_dataset)
