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
from minilab.registry import register_tokenizer


@register_tokenizer("bpe")
class BPETokenizer(BaseTokenizer):

    def __init__(self):
        self.merges: dict[tuple[int, int], int] = {}  # (a, b) -> merged_id
        self.vocab: dict[int, bytes] = {}  # id -> bytes

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        assert vocab_size >= 256, "BPE vocab must be >= 256 (byte-level base)"
        num_merges = vocab_size - 256

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
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

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
        self.merges = {tuple(map(int, k.split(","))): v for k, v in state["merges"].items()}
        self.vocab = {int(k): bytes(v) for k, v in state["vocab"].items()}


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
