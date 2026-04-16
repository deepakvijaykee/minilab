"""SEDD: Score Entropy Discrete Diffusion (absorbing noise).
Learns probability ratios (scores) instead of probabilities.
Score entropy loss avoids computing the normalizing constant.
Lou et al., ICML 2024 Best Paper."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.config import BaseConfig
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import get_position, register_model


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
        assert config.dim % config.num_heads == 0, "dim must be divisible by num_heads"
        head_dim = config.dim // config.num_heads
        assert head_dim % 2 == 0, "RoPE requires even head dimension"
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_emb = SinusoidalTimeEmbedding(config.dim)
        self.pos_enc = get_position("rope")(head_dim, config.max_seq_len)
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
        x = self._cast_hidden(self.tok_emb(z_t))
        t_emb = self.time_emb(t)
        freqs_cis = self.pos_enc(z_t.size(1))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, freqs_cis, use_reentrant=False)
            else:
                x = block(x, t_emb, freqs_cis=freqs_cis)
        x = self.ln_f(x)
        return self.score_head(x)

    def compute_loss(self, scores, x_0, mask, t, fwd):
        """Score entropy on masked positions.

        The partition sum is over non-[MASK] tokens only: [MASK] is the absorbing
        reference state and its score is implicitly fixed at 0, so including it in
        sum exp(s_y) both misstates the objective and adds a numerically unstable
        extra exponential term. This matches the sampler, which also drops [MASK].

        The partition is reduced in float64 because exp() of moderately large
        scores overflows bf16/fp32 (exp(90) ≈ 1e39 > fp32 max); bf16/fp32 during
        autocast is fine for the network, but the objective itself must be
        evaluated at a precision that can hold the unbounded score range."""
        s = scores[mask].double()
        targets = x_0[mask]
        target_scores = s.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        vocab_ids = torch.arange(s.size(-1), device=s.device)
        s_no_mask = s.masked_fill(vocab_ids == self.mask_token_id, float("-inf"))
        loss = (s_no_mask.exp().sum(dim=-1) - target_scores).mean()
        assert torch.isfinite(loss), f"SEDD score-entropy loss is non-finite (max score={scores[mask].max().item():.2f})"
        return loss
