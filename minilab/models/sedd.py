"""SEDD: Score Entropy Discrete Diffusion (absorbing noise).
Learns probability ratios (scores) instead of probabilities.
Score entropy loss avoids computing the normalizing constant.
Lou et al., ICML 2024 Best Paper."""

from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.checks import require
from minilab.models.diffusion_base import (
    DiffusionModelConfig,
    loss_normalizer,
    validate_clean_tokens,
    validate_loss_mask,
)
from minilab.nn.diffusion import (
    DiffusionBlock,
    SinusoidalTimeEmbedding,
    commit_diffusion_block_updates,
    diffusion_blocks_auxiliary_loss,
)
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
            DiffusionBlock(
                config.dim,
                config.num_heads,
                int(config.dim * config.ffn_mult),
                config.dropout,
                attention=config.attention,
                num_kv_heads=config.num_kv_heads,
                ffn=config.ffn,
                num_experts=config.num_experts,
                top_k_experts=config.top_k_experts,
                block_id=i,
            )
            for i in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.dim)
        self.score_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.score_head.weight
        self.mask_token_id = config.mask_token_id
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.score_head)

    def auxiliary_loss(self):
        return diffusion_blocks_auxiliary_loss(self.blocks, self.tok_emb.weight)

    def set_qk_clip_recording(self, enabled):
        for block in self.blocks:
            block.set_qk_clip_recording(enabled)

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_diffusion_block_updates(self.blocks, qk_clip_threshold, qk_clip_balance)

    def forward(self, z_t, t):
        require(z_t.size(1) <= self.config.max_seq_len, (
            f"SEDD supports at most {self.config.max_seq_len} tokens, got {z_t.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(z_t))
        t_emb = self.time_emb(t)
        freqs_cis = self.pos_enc(z_t.size(1))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, freqs_cis, use_reentrant=False)
            else:
                x = block(x, t_emb, freqs_cis=freqs_cis)
        x = self.ln_f(x)
        scores = self.score_head(x)
        return scores.scatter(-1, z_t.unsqueeze(-1), torch.zeros_like(scores[..., :1]))

    def compute_loss_per_example(self, scores, x_0, mask, t, fwd, loss_mask=None, normalization="sequence"):
        """Absorbing-graph score entropy from Lou et al. 2024.

        The network emits log score ratios. For masked positions the absorbing
        graph reduces the loss to the non-mask partition term, the clean-token
        score weighted by 1/expm1(sigma), and the matching constant term.
        """
        validate_clean_tokens(x_0, self.config, "SEDD loss")
        loss_mask = validate_loss_mask(loss_mask, x_0, "SEDD loss")
        require(fwd.process_type == self.forward_process_type, "SEDD loss requires the absorbing forward process")
        target_mask = mask if loss_mask is None else (mask & loss_mask)

        log_score = scores.double()
        sigma = fwd.get_sigma(t.to(scores.device)).double().view(-1, 1)
        dsigma = fwd.get_sigma_derivative(t.to(scores.device)).double().view(-1, 1)

        entropy = torch.zeros_like(x_0, dtype=log_score.dtype)
        masked_scores = log_score[target_mask]
        targets = x_0[target_mask]
        target_scores = masked_scores.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        vocab_ids = torch.arange(log_score.size(-1), device=scores.device)
        non_mask_scores = masked_scores.masked_fill(vocab_ids == self.mask_token_id, float("-inf"))
        pos_term = non_mask_scores.exp().sum(dim=-1)

        ratio = 1.0 / torch.expm1(sigma.expand_as(x_0)[target_mask]).clamp(min=1e-12)
        neg_term = ratio * target_scores
        const = ratio * (ratio.log() - 1.0)
        entropy[target_mask] = pos_term - neg_term + const

        per_example = (dsigma * entropy).sum(dim=-1)
        per_example = per_example / loss_normalizer(x_0, loss_mask, normalization).to(per_example.device, dtype=per_example.dtype)
        loss = per_example.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"SEDD score-entropy loss is non-finite (max score={scores[target_mask].max().item():.2f})"
            )
        return per_example

    def compute_loss(self, scores, x_0, mask, t, fwd):
        return self.compute_loss_per_example(scores, x_0, mask, t, fwd).mean()
