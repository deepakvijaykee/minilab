from minilab.registry import get_task, list_available
from minilab.tasks.gsm8k import GSM8KTask


def list_tasks():
    return list_available("task")


__all__ = ["GSM8KTask", "get_task", "list_tasks"]
