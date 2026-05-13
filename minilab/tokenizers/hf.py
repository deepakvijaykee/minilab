from pathlib import Path

from minilab.base import BaseTokenizer
from minilab.checks import require
from minilab.registry import register_tokenizer


@register_tokenizer("hf")
class HFTokenizer(BaseTokenizer):
    """Thin wrapper around a local Hugging Face tokenizer directory.

    This lets imported HF checkpoints use Minilab's native datasets and trainers
    without teaching every training script about Transformers tokenizers.
    """

    def __init__(self, path="", vocab_size=0):
        self.path = str(path)
        self._vocab_size = int(vocab_size) if vocab_size else 0
        self._tokenizer = None
        self._state_base_dir = None

    @classmethod
    def from_pretrained(cls, path):
        tok = cls(str(Path(path).expanduser().resolve()))
        tok._ensure_loaded()
        tok._vocab_size = len(tok._tokenizer)
        return tok

    def _set_state_base_dir(self, path):
        self._state_base_dir = Path(path)

    def _resolved_path(self):
        path = Path(self.path).expanduser()
        if path.is_absolute():
            return path
        if self._state_base_dir is not None:
            return self._state_base_dir / path
        return path

    def _ensure_loaded(self):
        if self._tokenizer is not None:
            return
        require(self.path, "HF tokenizer path is empty")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HF tokenizer support requires transformers. "
                "Install with: python -m pip install -e \".[hf]\""
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(str(self._resolved_path()))
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._vocab_size = len(self._tokenizer)

    def encode(self, text):
        self._ensure_loaded()
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        self._ensure_loaded()
        return self._tokenizer.decode(ids, skip_special_tokens=True)

    @property
    def vocab_size(self):
        if self._vocab_size == 0:
            self._ensure_loaded()
        return self._vocab_size

    def _get_state(self):
        vocab_size = self._vocab_size
        if vocab_size == 0 and self.path:
            vocab_size = self.vocab_size
        return {
            "type": "hf",
            "path": self.path,
            "vocab_size": vocab_size,
        }

    def _set_state(self, state):
        require(type(state) is dict, "HF tokenizer state must be a JSON object")
        require(set(state) == {"type", "path", "vocab_size"}, (
            "HF tokenizer state fields must be exactly: path, type, vocab_size"
        ))
        require(state["type"] == "hf", "HF tokenizer state has wrong type")
        require(type(state["path"]) is str and state["path"], "HF tokenizer path must be a non-empty string")
        require(type(state["vocab_size"]) is int and state["vocab_size"] > 0, (
            "HF tokenizer vocab_size must be a positive integer"
        ))
        self.path = state["path"]
        self._vocab_size = state["vocab_size"]
        self._tokenizer = None
