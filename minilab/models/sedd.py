"""SEDD: Score Entropy Discrete Diffusion.
Learns probability ratios (scores) instead of probabilities.
Score entropy loss avoids computing the normalizing constant.
Lou et al., ICML 2024 Best Paper."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.config import BaseConfig
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import register_model


@dataclass
class SEDDConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    mask_token_id: int = 0


@register_model("sedd")
class SEDD(BaseModel):
    config_class = SEDDConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_emb = SinusoidalTimeEmbedding(config.dim)
        self.blocks = nn.ModuleList([
            DiffusionBlock(config.dim, config.num_heads, int(config.dim * config.ffn_mult), config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.score_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.score_head.weight
        self.mask_token_id = config.mask_token_id
        self.apply(self._init_weights)

    def forward(self, z_t, t):
        x = self.tok_emb(z_t)
        t_emb = self.time_emb(t)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, use_reentrant=False)
            else:
                x = block(x, t_emb)
        x = self.ln_f(x)
        return self.score_head(x)

    def compute_loss(self, scores, x_0, mask, t, fwd):
        """Score entropy: L = sum_y exp(s_y) - s_{x_0} per masked position."""
        s = scores[mask]
        targets = x_0[mask]
        return (s.exp().sum(dim=-1) - s.gather(-1, targets.unsqueeze(-1)).squeeze(-1)).mean()
