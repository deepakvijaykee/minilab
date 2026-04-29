"""Character-level tokenizer: one token per unique character."""

from minilab.base import BaseTokenizer
from minilab.checks import require
from minilab.registry import register_tokenizer


@register_tokenizer("character")
class CharacterTokenizer(BaseTokenizer):
    """Simplest tokenizer: maps each character to an ID. Good for text8, debugging."""

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def train(self, text: str, vocab_size: int = 0) -> None:
        require(isinstance(text, str) and len(text) > 0, (
            "Character tokenizer training text must be a non-empty string"
        ))
        chars = sorted(set(text))
        require(vocab_size == 0 or vocab_size >= len(chars), (
            f"Character vocab_size ({vocab_size}) cannot cover {len(chars)} unique characters"
        ))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}

    def encode(self, text: str) -> list[int]:
        require(self.char_to_id, "Character tokenizer must be trained or loaded before encoding")
        missing = sorted(set(text) - set(self.char_to_id))
        require(not missing, f"Character encode received unknown characters: {missing[:5]}")
        return [self.char_to_id[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        missing = [i for i in ids if i not in self.id_to_char]
        require(not missing, f"Character decode received unknown token ids: {missing[:5]}")
        return "".join(self.id_to_char[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_id)

    def _get_state(self):
        return {"type": "character", "char_to_id": self.char_to_id}

    def _set_state(self, state):
        self.char_to_id = state["char_to_id"]
        self.id_to_char = {i: c for c, i in self.char_to_id.items()}
