"""Character-level tokenizer: one token per unique character."""

from minilab.base import BaseTokenizer
from minilab.registry import register_tokenizer


@register_tokenizer("character")
class CharacterTokenizer(BaseTokenizer):
    """Simplest tokenizer: maps each character to an ID. Good for text8, debugging."""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def train(self, text: str, vocab_size: int = 0) -> None:
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def _get_state(self):
        return {"type": "character", "char_to_id": self.char_to_id}

    def _set_state(self, state):
        self.char_to_id = state["char_to_id"]
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
