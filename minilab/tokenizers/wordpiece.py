"""WordPiece tokenizer from scratch.

Key difference from BPE:
  - BPE merges most FREQUENT pair
  - WordPiece merges pair that maximizes LIKELIHOOD: score(a,b) = freq(ab) / (freq(a) * freq(b))
  - Encoding uses greedy longest-match (no merge rules needed)
  - ## prefix marks continuation of a word

Used by: BERT, DistilBERT, Electra
"""

from minilab.base import BaseTokenizer
from minilab.registry import register_tokenizer


@register_tokenizer("wordpiece")
class WordPieceTokenizer(BaseTokenizer):

    PREFIX = "##"
    UNK = "[UNK]"

    def __init__(self):
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        words = text.split()
        word_freqs: dict[str, int] = {}
        for w in words:
            word_freqs[w] = word_freqs.get(w, 0) + 1

        # Initialize with characters (## prefix for non-initial)
        vocab: set[str] = {self.UNK}
        for word in word_freqs:
            vocab.add(word[0])
            for c in word[1:]:
                vocab.add(self.PREFIX + c)

        # Iteratively merge pair with highest likelihood: score(a,b) = freq(ab) / (freq(a) * freq(b))
        while len(vocab) < vocab_size:
            symbol_freqs: dict[str, int] = {}
            pair_freqs: dict[tuple[str, str], int] = {}
            for word, freq in word_freqs.items():
                symbols = self._split_word(word, vocab)
                for s in symbols:
                    symbol_freqs[s] = symbol_freqs.get(s, 0) + freq
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + freq

            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p] / (symbol_freqs[p[0]] * symbol_freqs[p[1]]))
            merged = best_pair[0] + best_pair[1].removeprefix(self.PREFIX)
            vocab.add(merged)

            if verbose and len(vocab) % 1000 == 0:
                print(f"vocab size: {len(vocab)}, added: '{merged}'")

        tokens = sorted(vocab)
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def _split_word(self, word: str, vocab: set[str]) -> list[str]:
        """Split word into known vocab tokens using greedy longest-match."""
        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            while start < end:
                piece = word[start:end]
                if start > 0:
                    piece = self.PREFIX + piece
                if piece in vocab:
                    tokens.append(piece)
                    found = True
                    break
                end -= 1
            if not found:
                return [self.UNK]
            start = end
        return tokens

    def encode(self, text: str) -> list[int]:
        ids = []
        unk_id = self.token_to_id[self.UNK]
        for word in text.split():
            tokens = self._split_word(word, set(self.token_to_id.keys()))
            ids.extend(self.token_to_id.get(t, unk_id) for t in tokens)
        return ids

    def decode(self, ids: list[int]) -> str:
        parts: list[str] = []
        for i in ids:
            token = self.id_to_token[i]
            if token == self.UNK:
                parts.append(token)
            elif token.startswith(self.PREFIX):
                parts.append(token[len(self.PREFIX) :])
            else:
                if parts:
                    parts.append(" ")
                parts.append(token)
        return "".join(parts)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def _get_state(self):
        return {"type": "wordpiece", "token_to_id": self.token_to_id}

    def _set_state(self, state):
        self.token_to_id = state["token_to_id"]
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
