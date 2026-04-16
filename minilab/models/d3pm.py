"""D3PM: Discrete Denoising Diffusion Probabilistic Models (absorbing noise).
VLB loss: KL between true posterior q(z_{t-1}|z_t,x_0) and model p_θ(z_{t-1}|z_t).
For absorbing noise, the posterior at masked positions is a two-point distribution:
unmask to x_0 with prob (alpha_{t-1} - alpha_t)/(1 - alpha_t), or stay masked.
Austin et al., NeurIPS 2021."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.config import BaseConfig
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import get_position, register_model


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
    hybrid_coef: float = 0.001


@register_model("d3pm")
class D3PM(BaseModel):
    config_class = D3PMConfig

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
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.mask_token_id = config.mask_token_id
        self.hybrid_coef = config.hybrid_coef
        self.apply(self._init_weights)

    def forward(self, z_t, t):
        """Predicts p_θ(z_{t-1} | z_t) — distribution over all tokens including mask."""
        x = self._cast_hidden(self.tok_emb(z_t))
        t_emb = self.time_emb(t)
        freqs_cis = self.pos_enc(z_t.size(1))
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, t_emb, freqs_cis, use_reentrant=False)
            else:
                x = block(x, t_emb, freqs_cis=freqs_cis)
        x = self.ln_f(x)
        return self.lm_head(x)

    def compute_loss(self, logits, x_0, mask, t, fwd):
        """VLB: KL(q(z_{t-1}|z_t,x_0) || p_θ(z_{t-1}|z_t)) at masked positions + auxiliary CE."""
        eps = 1e-8
        idx = (t * fwd.num_timesteps).long().clamp(max=fwd.num_timesteps)
        alpha_t = fwd.alpha.to(logits.device)[idx].view(-1, 1)
        alpha_prev = fwd.alpha.to(logits.device)[(idx - 1).clamp(min=0)].view(-1, 1)

        # True posterior for absorbing noise at masked positions
        denom = (1 - alpha_t).clamp(min=eps)
        q_unmask = ((alpha_prev - alpha_t) / denom).clamp(min=eps)
        q_stay = ((1 - alpha_prev) / denom).clamp(min=eps)

        # Model's predicted distribution
        log_p = F.log_softmax(logits, dim=-1)
        log_p_x0 = log_p.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        log_p_mask = log_p[:, :, self.mask_token_id]

        # KL at masked positions
        kl = q_unmask * (q_unmask.log() - log_p_x0) + q_stay * (q_stay.log() - log_p_mask)
        vlb = (kl * mask.float()).sum() / mask.float().sum()

        # Auxiliary CE (Austin et al. Section 3.3)
        ce = F.cross_entropy(logits[mask], x_0[mask])
        return vlb + self.hybrid_coef * ce
