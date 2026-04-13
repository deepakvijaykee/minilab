"""MDLM: Masked Diffusion Language Model.
SUBS parameterization: model never predicts [MASK].
Sahoo et al., NeurIPS 2024."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.config import BaseConfig
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import register_model


@dataclass
class MDLMConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    mask_token_id: int = 0


@register_model("mdlm")
class MDLM(BaseModel):
    config_class = MDLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_emb = SinusoidalTimeEmbedding(config.dim)
        self.blocks = nn.ModuleList([
            DiffusionBlock(config.dim, config.num_heads, int(config.dim * config.ffn_mult), config.dropout)
            for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
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
        logits = self.lm_head(x)
        logits[:, :, self.mask_token_id] = float("-inf")
        return logits

    def compute_loss(self, logits, x_0, mask, t, fwd):
        return F.cross_entropy(logits[mask], x_0[mask])
