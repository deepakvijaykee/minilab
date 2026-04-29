"""Byte-Pair Encoding tokenizer from scratch.

Algorithm (training):
  1. Start with byte-level vocabulary (256 tokens)
  2. Count all adjacent pairs in corpus
  3. Merge most frequent pair into new token
  4. Repeat until desired vocab size

Algorithm (encoding):
  1. Convert text to bytes
  2. Iteratively apply learned merges (earliest learned first)

Used by: GPT-2/3/4, Llama, Mistral, RoBERTa
"""

from minilab.base import BaseTokenizer
from minilab.checks import require
from minilab.registry import register_tokenizer
from minilab.tokenizers._state import require_tokenizer_state


@register_tokenizer("bpe")
class BPETokenizer(BaseTokenizer):

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> merged_id
        self.vocab: dict[int, bytes] = {}  # id -> bytes

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        require(isinstance(text, str) and len(text) > 0, "BPE training text must be a non-empty string")
        require(vocab_size >= 256, "BPE vocab must be >= 256 (byte-level base)")
        num_merges = vocab_size - 256

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        ids = list(text.encode("utf-8"))

        for i in range(num_merges):
            counts: dict[tuple[int, int], int] = {}
            for j in range(len(ids) - 1):
                pair = (ids[j], ids[j + 1])
                counts[pair] = counts.get(pair, 0) + 1

            if not counts:
                break

            best = max(counts, key=counts.get)
            new_id = 256 + i
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            ids = _apply_merge(ids, best, new_id)

            if verbose and (i + 1) % 500 == 0:
                token = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1}/{num_merges}: {best} -> '{token}' (freq={counts[best]})")

    def encode(self, text: str) -> list[int]:
        require(self.vocab, "BPE tokenizer must be trained or loaded before encoding")
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            # Find mergeable pair with lowest new_id (earliest learned)
            best_pair = None
            best_id = float("inf")
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                if pair in self.merges and self.merges[pair] < best_id:
                    best_pair = pair
                    best_id = self.merges[pair]
            if best_pair is None:
                break
            ids = _apply_merge(ids, best_pair, best_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        missing = [i for i in ids if i not in self.vocab]
        require(not missing, f"BPE decode received unknown token ids: {missing[:5]}")
        raw = b"".join(self.vocab[i] for i in ids)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("BPE decode received a token sequence that is not valid UTF-8") from exc

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _get_state(self):
        return {
            "type": "bpe",
            "merges": {f"{a},{b}": v for (a, b), v in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }

    def _set_state(self, state):
        require_tokenizer_state(state, "BPE", "bpe", ("merges", "vocab"))

        merges: dict[tuple[int, int], int] = {}
        for key, value in state["merges"].items():
            parts = key.split(",") if type(key) is str else []
            require(len(parts) == 2 and all(part.isdecimal() for part in parts), (
                "BPE tokenizer merge keys must be 'left,right' token ids"
            ))
            require(type(value) is int and value >= 0, "BPE tokenizer merge values must be token ids")
            merges[(int(parts[0]), int(parts[1]))] = value

        vocab: dict[int, bytes] = {}
        for key, value in state["vocab"].items():
            require(type(key) is str and key.isdecimal(), "BPE tokenizer vocab keys must be token ids")
            require(type(value) is list and all(type(byte) is int and 0 <= byte <= 255 for byte in value), (
                "BPE tokenizer vocab values must be byte lists"
            ))
            vocab[int(key)] = bytes(value)
        require(vocab, "BPE tokenizer state vocab must be non-empty")
        self.merges = merges
        self.vocab = vocab


def _apply_merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace all occurrences of pair with new_id in the id list."""
    result = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result
