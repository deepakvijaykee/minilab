from minilab.registry import get_task, list_available
from minilab.tasks.gsm8k import GSM8KTask
from minilab.tasks.verifier_toys import (
    FormatAnswerTask,
    MiniArithmeticTask,
    TinyCodeRepairTask,
    ToolCallJSONTask,
)


def list_tasks():
    return list_available("task")


__all__ = [
    "FormatAnswerTask",
    "GSM8KTask",
    "MiniArithmeticTask",
    "TinyCodeRepairTask",
    "ToolCallJSONTask",
    "get_task",
    "list_tasks",
]
