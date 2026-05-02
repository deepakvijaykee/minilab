import json
from pathlib import Path

from minilab.tokenizers import byte as _byte
from minilab.tokenizers import bpe as _bpe
from minilab.tokenizers import character as _character
from minilab.tokenizers import unigram as _unigram
from minilab.tokenizers import wordpiece as _wordpiece
from minilab.checks import require
from minilab.registry import get_tokenizer, list_available

_TOKENIZER_MODULES = (_byte, _bpe, _character, _unigram, _wordpiece)
__all__ = ["available_tokenizers", "build_tokenizer", "load_tokenizer"]


def available_tokenizers():
    return list_available("tokenizer")


def build_tokenizer(name):
    return get_tokenizer(name)()


def load_tokenizer(path):
    """Load any saved tokenizer — reads the 'type' field to pick the right class."""
    state = json.loads(Path(path).read_text())
    require(isinstance(state, dict), "Tokenizer state must be a JSON object")
    require("type" in state, "Tokenizer state is missing required field: type")
    require(isinstance(state["type"], str) and state["type"], (
        "Tokenizer state field 'type' must be a non-empty string"
    ))
    cls = get_tokenizer(state["type"])
    tok = cls()
    tok._set_state(state)
    return tok
