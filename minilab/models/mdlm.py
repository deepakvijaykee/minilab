"""MDLM: Masked Diffusion Language Model.
Full SUBS parameterization: (1) never predict [MASK], (2) carry over observed
tokens at unmasked positions. Loss computed only on masked positions.
Sahoo et al., NeurIPS 2024."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

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
class MDLMConfig(DiffusionModelConfig):
    """Configuration for masked diffusion language modeling."""


@register_model("mdlm")
class MDLM(BaseModel):
    config_class = MDLMConfig
    forward_process_type = "absorbing"
    reverse_parameterization = "clean_logits"
    requires_terminal_mask_prior = True

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
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.mask_token_id = config.mask_token_id
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.lm_head)

    def auxiliary_loss(self):
        return diffusion_blocks_auxiliary_loss(self.blocks, self.tok_emb.weight)

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_diffusion_block_updates(self.blocks, qk_clip_threshold, qk_clip_balance)

    def forward(self, z_t, t):
        require(z_t.size(1) <= self.config.max_seq_len, (
            f"MDLM supports at most {self.config.max_seq_len} tokens, got {z_t.size(1)}"
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
        logits = self.lm_head(x)
        # SUBS: never predict [MASK]
        logits[:, :, self.mask_token_id] = float("-inf")
        # SUBS carry-over: unmasked positions predict the observed token
        unmasked = z_t != self.mask_token_id
        logits[unmasked] = float("-inf")
        logits[unmasked, z_t[unmasked]] = 0.0
        return logits

    def compute_loss_per_example(self, logits, x_0, mask, t, fwd, loss_mask=None, normalization="sequence"):
        """MDLM continuous-time NELBO (Sahoo et al., 2024, Eq. 11):
            L = E_{t~U, z_t~q} [ -α'(t)/(1-α(t)) · Σ_{i: masked} log p_θ(x_0^i | z_t) ]
        The earlier implementation was unweighted masked CE, which ignores the
        schedule-derived weighting and is no longer the ELBO objective. We normalize
        by sequence length so the loss magnitude is comparable across seq_len."""
        validate_clean_tokens(x_0, self.config, "MDLM loss")
        loss_mask = validate_loss_mask(loss_mask, x_0, "MDLM loss")
        require(fwd.process_type == self.forward_process_type, "MDLM loss requires the absorbing forward process")
        target_mask = mask if loss_mask is None else (mask & loss_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_x0 = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        per_ex_loglik = (log_p_x0 * target_mask.float()).sum(dim=-1)
        w = fwd.get_weight(t.to(logits.device))
        denom = loss_normalizer(x_0, loss_mask, normalization).to(logits.device, dtype=logits.dtype)
        return -(w * per_ex_loglik) / denom

    def compute_loss(self, logits, x_0, mask, t, fwd):
        return self.compute_loss_per_example(logits, x_0, mask, t, fwd).mean()
