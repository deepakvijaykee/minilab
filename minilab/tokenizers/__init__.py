import json
from pathlib import Path

from minilab.tokenizers import bpe
from minilab.tokenizers import character
from minilab.tokenizers import unigram
from minilab.tokenizers import wordpiece

_CLASSES = {
    "bpe": bpe.BPETokenizer,
    "character": character.CharacterTokenizer,
    "wordpiece": wordpiece.WordPieceTokenizer,
    "unigram": unigram.UnigramTokenizer,
}


def load_tokenizer(path):
    """Load any saved tokenizer — reads the 'type' field to pick the right class."""
    state = json.loads(Path(path).read_text())
    cls = _CLASSES[state["type"]]
    tok = cls()
    tok._set_state(state)
    return tok
