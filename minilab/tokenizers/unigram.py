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
from minilab.checks import require
from minilab.registry import register_tokenizer


@register_tokenizer("unigram")
class UnigramTokenizer(BaseTokenizer):

    MAX_PIECE_LEN = 16
    SEED_MULTIPLIER = 10
    EM_ITERS_PER_PRUNE = 2
    FINAL_EM_ITERS = 4
    PRUNE_FRACTION = 0.25
    SMOOTHING = 1e-8
    UNK = "[UNK]"

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.scores: dict[str, float] = {}  # token -> log probability

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        require(isinstance(text, str) and len(text) > 0, "Unigram training text must be a non-empty string")
        all_chars = set(text)
        require(vocab_size >= len(all_chars) + 1, (
            f"Unigram vocab_size ({vocab_size}) must cover all {len(all_chars)} unique characters plus {self.UNK}"
        ))
        word_freqs = self._word_frequencies(text)
        required = set(all_chars) | {self.UNK}
        seed_counts = self._seed_piece_counts(word_freqs, required, vocab_size)
        self.scores = self._scores_from_counts(seed_counts)

        counts = seed_counts
        while len(self.scores) > vocab_size:
            for _ in range(self.EM_ITERS_PER_PRUNE):
                counts, nll = self._expected_counts(word_freqs, self.scores)
                self.scores = self._scores_from_counts(counts, required)

            optional = [piece for piece in self.scores if piece not in required]
            remove_count = min(
                len(self.scores) - vocab_size,
                max(1, int(len(optional) * self.PRUNE_FRACTION)),
            )
            for piece in sorted(optional, key=lambda p: (counts.get(p, 0.0), len(p), p))[:remove_count]:
                del self.scores[piece]

            if verbose:
                print(f"pruned to {len(self.scores)} tokens (nll={nll:.2f})")

        for _ in range(self.FINAL_EM_ITERS):
            counts, _ = self._expected_counts(word_freqs, self.scores)
            self.scores = self._scores_from_counts(counts, required)

        tokens = sorted(self.scores.keys())
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def _word_frequencies(self, text: str) -> dict[str, int]:
        words = text.split()
        require(words, "Unigram training text must contain at least one non-whitespace word")
        freqs: dict[str, int] = {}
        for word in words:
            freqs[word] = freqs.get(word, 0) + 1
        return freqs

    def _seed_piece_counts(self, word_freqs: dict[str, int], required: set[str], vocab_size: int) -> dict[str, float]:
        counts: dict[str, float] = {}
        for word, freq in word_freqs.items():
            for length in range(1, min(len(word), self.MAX_PIECE_LEN) + 1):
                for start in range(len(word) - length + 1):
                    piece = word[start : start + length]
                    counts[piece] = counts.get(piece, 0.0) + freq

        optional_budget = max(0, vocab_size * self.SEED_MULTIPLIER - len(required))
        optional = sorted(
            ((p, c) for p, c in counts.items() if p not in required),
            key=lambda item: (-item[1], -len(item[0]), item[0]),
        )[:optional_budget]
        seed_counts = {p: c for p, c in optional}
        for piece in required:
            seed_counts[piece] = max(seed_counts.get(piece, 0.0), self.SMOOTHING)
        return seed_counts

    def _expected_counts(self, word_freqs: dict[str, int], scores: dict[str, float]):
        counts = {piece: 0.0 for piece in scores}
        nll = 0.0
        for word, freq in word_freqs.items():
            edges = self._lattice_edges(word, scores)
            forward = self._forward_scores(edges, len(word), scores)
            backward = self._backward_scores(edges, len(word), scores)
            log_z = forward[-1]
            require(log_z > float("-inf"), f"Unigram lattice has no valid path for word: {word!r}")
            nll -= freq * log_z
            for start, outgoing in enumerate(edges):
                if forward[start] == float("-inf"):
                    continue
                for end, piece in outgoing:
                    posterior = math.exp(forward[start] + scores[piece] + backward[end] - log_z)
                    counts[piece] += freq * posterior
        return counts, nll

    def _lattice_edges(self, word: str, scores: dict[str, float]):
        edges = [[] for _ in range(len(word) + 1)]
        for start in range(len(word)):
            for end in range(start + 1, min(len(word), start + self.MAX_PIECE_LEN) + 1):
                piece = word[start:end]
                if piece in scores:
                    edges[start].append((end, piece))
        return edges

    def _forward_scores(self, edges, n: int, scores: dict[str, float]):
        forward = [float("-inf")] * (n + 1)
        forward[0] = 0.0
        for start, outgoing in enumerate(edges):
            if forward[start] == float("-inf"):
                continue
            for end, piece in outgoing:
                forward[end] = _logaddexp(forward[end], forward[start] + scores[piece])
        return forward

    def _backward_scores(self, edges, n: int, scores: dict[str, float]):
        backward = [float("-inf")] * (n + 1)
        backward[n] = 0.0
        for start in range(n - 1, -1, -1):
            total = float("-inf")
            for end, piece in edges[start]:
                total = _logaddexp(total, scores[piece] + backward[end])
            backward[start] = total
        return backward

    def _scores_from_counts(self, counts: dict[str, float], required: set[str] | None = None):
        adjusted = {piece: count + self.SMOOTHING for piece, count in counts.items()}
        for piece in required or ():
            adjusted[piece] = adjusted.get(piece, 0.0) + self.SMOOTHING
        total = sum(adjusted.values())
        return {piece: math.log(count / total) for piece, count in adjusted.items()}

    def encode(self, text: str) -> list[int]:
        require(self.UNK in self.token_to_id, "Unigram tokenizer must be trained or loaded before encoding")
        ids = []
        unk_id = self.token_to_id[self.UNK]
        space_id = self.token_to_id.get(" ")
        for i, word in enumerate(text.split(" ")):
            if i > 0:
                ids.append(space_id if space_id is not None else unk_id)
            if word:
                tokens = self._viterbi(word)
                ids.extend(self.token_to_id.get(t, unk_id) for t in tokens)
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
        missing = [i for i in ids if i not in self.id_to_token]
        require(not missing, f"Unigram decode received unknown token ids: {missing[:5]}")
        return "".join(self.id_to_token[i] for i in ids)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _get_state(self):
        return {"type": "unigram", "token_to_id": self.token_to_id, "scores": self.scores}

    def _set_state(self, state):
        self.token_to_id = state["token_to_id"]
        require(self.UNK in self.token_to_id, f"Unigram tokenizer state is missing {self.UNK}")
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.scores = state["scores"]
        require(self.UNK in self.scores, f"Unigram tokenizer state scores are missing {self.UNK}")


def _logaddexp(a: float, b: float) -> float:
    if a == float("-inf"):
        return b
    if b == float("-inf"):
        return a
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))
