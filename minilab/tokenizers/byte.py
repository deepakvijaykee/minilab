"""UTF-8 byte tokenizer used by byte-latent experiments."""

from minilab.base import BaseTokenizer
from minilab.checks import require
from minilab.registry import register_tokenizer


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
BOE_ID = 3
BPE_ID = 4
BYTE_OFFSET = 5
BYTE_VOCAB_SIZE = 256 + BYTE_OFFSET


@register_tokenizer("byte")
class ByteTokenizer(BaseTokenizer):
    """BLT-style tokenizer: bytes are represented as `byte + BYTE_OFFSET`."""

    def __init__(self):
        self.add_bos = False
        self.add_eos = False

    def train(self, text: str = "", vocab_size: int = 0) -> None:
        require(vocab_size in {0, BYTE_VOCAB_SIZE}, (
            f"ByteTokenizer has fixed vocab size {BYTE_VOCAB_SIZE}"
        ))

    def encode(self, text: str) -> list[int]:
        ids = [byte + BYTE_OFFSET for byte in text.encode("utf-8")]
        if self.add_bos:
            ids = [BOS_ID] + ids
        if self.add_eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        byte_values = []
        for token_id in ids:
            if token_id in {PAD_ID, BOS_ID, EOS_ID, BOE_ID, BPE_ID}:
                continue
            require(BYTE_OFFSET <= token_id < BYTE_VOCAB_SIZE, (
                f"ByteTokenizer decode received non-byte id {token_id}"
            ))
            byte_values.append(token_id - BYTE_OFFSET)
        return bytes(byte_values).decode("utf-8", errors="strict")

    @property
    def vocab_size(self) -> int:
        return BYTE_VOCAB_SIZE

    def _get_state(self):
        return {"type": "byte", "add_bos": self.add_bos, "add_eos": self.add_eos}

    def _set_state(self, state):
        require(type(state) is dict, "Byte tokenizer state must be a JSON object")
        require(set(state) == {"type", "add_bos", "add_eos"}, (
            "Byte tokenizer state fields must be exactly: add_bos, add_eos, type"
        ))
        require(state["type"] == "byte", "Byte tokenizer state has wrong type")
        require(type(state["add_bos"]) is bool, "Byte tokenizer add_bos must be bool")
        require(type(state["add_eos"]) is bool, "Byte tokenizer add_eos must be bool")
        self.add_bos = state["add_bos"]
        self.add_eos = state["add_eos"]
