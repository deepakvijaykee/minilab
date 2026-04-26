from dataclasses import dataclass

import torch

from minilab.checks import require
from minilab.config import BaseConfig


@dataclass
class DiffusionModelConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    mask_token_id: int = 50256
    time_sampling: str = "continuous"

    def __post_init__(self):
        require(self.vocab_size > 1, "diffusion vocab_size must include at least one clean token plus [MASK]")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(self.dim % self.num_heads == 0, "dim must be divisible by num_heads")
        require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.ffn_mult > 0, "ffn_mult must be > 0")
        require(self.mask_token_id == self.vocab_size - 1, (
            "diffusion models reserve mask_token_id as the final vocab id; "
            "set vocab_size=data_vocab_size+1 and mask_token_id=data_vocab_size"
        ))
        require(self.time_sampling in {"continuous", "discrete"}, (
            "time_sampling must be 'continuous' or 'discrete'"
        ))


def validate_clean_tokens(x_0, config, context):
    require(x_0.dtype == torch.long, f"{context} requires integer token ids")
    require((x_0 >= 0).all(), f"{context} clean tokens must be non-negative")
    require((x_0 < config.mask_token_id).all(), f"{context} clean tokens must exclude the reserved [MASK] token")


def validate_infill_tokens(tokens, mask_positions, config, context):
    require(tokens.dtype == torch.long, f"{context} requires integer token ids")
    require((tokens >= 0).all(), f"{context} tokens must be non-negative")
    require((tokens <= config.mask_token_id).all(), (
        f"{context} tokens must be clean token ids or the reserved [MASK] placeholder"
    ))
    context_tokens = tokens[~mask_positions]
    require((context_tokens < config.mask_token_id).all(), (
        f"{context} context tokens must exclude the reserved [MASK] token"
    ))
