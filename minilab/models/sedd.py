"""SEDD: Score Entropy Discrete Diffusion (absorbing noise).
Learns probability ratios (scores) instead of probabilities.
Score entropy loss avoids computing the normalizing constant.
Lou et al., ICML 2024 Best Paper."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.checks import require
from minilab.models.diffusion_base import DiffusionModelConfig, validate_clean_tokens
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import get_position, register_model


@dataclass
class SEDDConfig(DiffusionModelConfig):
    """Configuration for score-entropy discrete diffusion."""


@register_model("sedd")
class SEDD(BaseModel):
    config_class = SEDDConfig
    forward_process_type = "absorbing"
    reverse_parameterization = "sedd_log_scores"

    def __init__(self, config):
        super().__init__(config)
        head_dim = config.dim // config.num_heads
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

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.score_head)

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
        """Absorbing-graph score entropy from Lou et al. 2024.

        The network emits log score ratios. For masked positions the absorbing
        graph reduces the loss to the non-mask partition term, the clean-token
        score weighted by 1/expm1(sigma), and the matching constant term.
        """
        validate_clean_tokens(x_0, self.config, "SEDD loss")
        require(fwd.process_type == self.forward_process_type, "SEDD loss requires the absorbing forward process")
        if not mask.any():
            return scores.sum() * 0.0

        log_score = scores.double()
        sigma = fwd.get_sigma(t.to(scores.device)).double().view(-1, 1)
        dsigma = fwd.get_sigma_derivative(t.to(scores.device)).double().view(-1, 1)

        entropy = torch.zeros_like(x_0, dtype=log_score.dtype)
        masked_scores = log_score[mask]
        targets = x_0[mask]
        target_scores = masked_scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        vocab_ids = torch.arange(log_score.size(-1), device=scores.device)
        non_mask_scores = masked_scores.masked_fill(vocab_ids == self.mask_token_id, float("-inf"))
        pos_term = non_mask_scores.exp().sum(dim=-1)

        ratio = 1.0 / torch.expm1(sigma.expand_as(x_0)[mask]).clamp(min=1e-12)
        neg_term = ratio * target_scores
        const = ratio * (ratio.log() - 1.0)
        entropy[mask] = pos_term - neg_term + const

        loss = (dsigma * entropy).sum(dim=-1).mean()
        require(
            torch.isfinite(loss),
            f"SEDD score-entropy loss is non-finite (max score={scores[mask].max().item():.2f})",
            FloatingPointError,
        )
        return loss
