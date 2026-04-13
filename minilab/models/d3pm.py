"""D3PM: Discrete Denoising Diffusion Probabilistic Models.
Transition-matrix approach. VLB loss + auxiliary CE.
Austin et al., NeurIPS 2021."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.config import BaseConfig
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import register_model


@dataclass
class D3PMConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    mask_token_id: int = 0
    transition: str = "absorbing"
    hybrid_coef: float = 0.001


@register_model("d3pm")
class D3PM(BaseModel):
    config_class = D3PMConfig

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
        self.hybrid_coef = config.hybrid_coef
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
        return self.lm_head(x)

    def compute_loss(self, logits, x_0, mask, t, fwd):
        """VLB (weighted CE where weight=1/(1-alpha_t)) + auxiliary CE."""
        ce_loss = F.cross_entropy(logits[mask], x_0[mask])
        alpha_t = fwd.get_alpha(t).to(logits.device).view(-1, 1)
        weight = 1.0 / (1.0 - alpha_t + 1e-8)
        log_probs = F.log_softmax(logits, dim=-1)
        target_logp = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        vlb_loss = -(target_logp * mask.float() * weight).sum() / mask.float().sum()
        return vlb_loss + self.hybrid_coef * ce_loss
