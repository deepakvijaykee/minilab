"""MDLM: Masked Diffusion Language Model.
Full SUBS parameterization: (1) never predict [MASK], (2) carry over observed
tokens at unmasked positions. Loss computed only on masked positions.
Sahoo et al., NeurIPS 2024."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.config import BaseConfig
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import get_position, register_model


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
        logits = self.lm_head(x)
        # SUBS: never predict [MASK]
        logits[:, :, self.mask_token_id] = float("-inf")
        # SUBS carry-over: unmasked positions predict the observed token
        unmasked = z_t != self.mask_token_id
        logits[unmasked] = float("-inf")
        logits[unmasked, z_t[unmasked]] = 0.0
        return logits

    def compute_loss(self, logits, x_0, mask, t, fwd):
        """MDLM continuous-time NELBO (Sahoo et al., 2024, Eq. 11):
            L = E_{t~U, z_t~q} [ -α'(t)/(1-α(t)) · Σ_{i: masked} log p_θ(x_0^i | z_t) ]
        The earlier implementation was unweighted masked CE, which ignores the
        schedule-derived weighting and is no longer the ELBO objective. We normalize
        by sequence length so the loss magnitude is comparable across seq_len."""
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_x0 = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        per_ex_loglik = (log_p_x0 * mask.float()).sum(dim=-1)
        w = fwd.get_weight(t.to(logits.device))
        return -(w * per_ex_loglik).mean() / x_0.size(1)
