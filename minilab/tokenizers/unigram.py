"""Unigram Language Model tokenizer from scratch.

Key difference from BPE/WordPiece:
  - BPE/WordPiece: bottom-up (start small, add tokens)
  - Unigram: top-down (start large, prune tokens)

Training: start with all substrings, iteratively prune least useful tokens.
Encoding: Viterbi algorithm finds the tokenization maximizing P(x) = prod(P(token_i)).

Used by: T5, XLNet, ALBERT, mBART (via SentencePiece)
"""

import math

from minilab.base import BaseTokenizer
from minilab.registry import register_tokenizer


@register_tokenizer("unigram")
class UnigramTokenizer(BaseTokenizer):

    MAX_PIECE_LEN = 16

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.scores: dict[str, float] = {}  # token -> log probability

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        words = text.split()

        substr_freqs: dict[str, int] = {}
        for word in words:
            for length in range(1, min(len(word), self.MAX_PIECE_LEN) + 1):
                for start in range(len(word) - length + 1):
                    s = word[start : start + length]
                    substr_freqs[s] = substr_freqs.get(s, 0) + 1

        pool = dict(sorted(substr_freqs.items(), key=lambda x: -x[1])[: vocab_size * 10])

        all_chars = set(text)
        for c in all_chars:
            if c not in pool:
                pool[c] = 1

        while len(pool) > vocab_size:
            target = max(vocab_size, int(len(pool) * 0.75))
            scored = sorted(pool.items(), key=lambda x: -(x[1] * len(x[0])))
            pool = {}
            for token, freq in scored[:target]:
                pool[token] = freq
            for c in all_chars:
                if c not in pool:
                    pool[c] = 1

            if verbose:
                print(f"pruned to {len(pool)} tokens")

        total = sum(pool.values())
        self.scores = {t: math.log(f / total) for t, f in pool.items()}
        tokens = sorted(pool.keys())
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def encode(self, text: str) -> list[int]:
        ids = []
        space_id = self.token_to_id.get(" ")
        for i, word in enumerate(text.split(" ")):
            if i > 0 and space_id is not None:
                ids.append(space_id)
            if word:
                tokens = self._viterbi(word)
                ids.extend(self.token_to_id[t] for t in tokens)
        return ids

    def _viterbi(self, word: str) -> list[str]:
        """Find optimal tokenization via dynamic programming."""
        n = len(word)
        NEG_INF = float("-inf")
        # best[i] = (best_score_to_position_i, backpointer)
        best = [(NEG_INF, -1)] * (n + 1)
        best[0] = (0.0, -1)

        for end in range(1, n + 1):
            for start in range(max(0, end - self.MAX_PIECE_LEN), end):
                piece = word[start:end]
                if piece in self.scores:
                    score = best[start][0] + self.scores[piece]
                    if score > best[end][0]:
                        best[end] = (score, start)

        # Fallback to characters if no path found
        if best[n][0] == NEG_INF:
            return list(word)

        # Backtrack
        tokens = []
        pos = n
        while pos > 0:
            prev = best[pos][1]
            tokens.append(word[prev:pos])
            pos = prev
        return tokens[::-1]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_token[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _get_state(self):
        return {"type": "unigram", "token_to_id": self.token_to_id, "scores": self.scores}

    def _set_state(self, state):
        self.token_to_id = state["token_to_id"]
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.scores = state["scores"]
