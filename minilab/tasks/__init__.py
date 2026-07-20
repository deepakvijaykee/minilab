from minilab.registry import get_task, list_available
from minilab.tasks.agentic_calculator import AgenticCalculatorTask
from minilab.tasks.gsm8k import GSM8KTask
from minilab.tasks.reasoning_gym import ReasoningGymTask
from minilab.tasks.verifier_toys import (
    FormatAnswerTask,
    MiniArithmeticTask,
    TinyCodeRepairTask,
    ToolCallJSONTask,
    VerifierHackingCodeRepairTask,
)


def list_tasks():
    return list_available("task")


__all__ = [
    "AgenticCalculatorTask",
    "FormatAnswerTask",
    "GSM8KTask",
    "MiniArithmeticTask",
    "ReasoningGymTask",
    "TinyCodeRepairTask",
    "ToolCallJSONTask",
    "VerifierHackingCodeRepairTask",
    "get_task",
    "list_tasks",
]
