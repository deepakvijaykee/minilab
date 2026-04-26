"""D3PM: Discrete Denoising Diffusion Probabilistic Models with absorbing noise.

The model uses the x0-parameterization from Austin et al., NeurIPS 2021:
predict clean-token logits, then combine them with the absorbing forward posterior
to obtain p_theta(z_s | z_t)."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.models.diffusion_base import DiffusionModelConfig, validate_clean_tokens
from minilab.nn.diffusion import DiffusionBlock, SinusoidalTimeEmbedding
from minilab.registry import get_position, register_model


@dataclass
class D3PMConfig(DiffusionModelConfig):
    transition: str = "absorbing"
    hybrid_coef: float = 0.001
    time_sampling: str = "discrete"

    def __post_init__(self):
        super().__post_init__()
        require(self.transition == "absorbing", "D3PM currently implements only the absorbing transition family")
        require(self.hybrid_coef >= 0, "hybrid_coef must be >= 0")
        require(self.time_sampling == "discrete", "D3PM is a discrete-time diffusion objective")


@register_model("d3pm")
class D3PM(BaseModel):
    config_class = D3PMConfig
    forward_process_type = "absorbing"
    reverse_parameterization = "d3pm_x0_logits"
    requires_terminal_mask_prior = True

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
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.mask_token_id = config.mask_token_id
        self.transition = config.transition
        self.hybrid_coef = config.hybrid_coef
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.lm_head)

    def forward(self, z_t, t):
        """Predicts clean-token logits p_tilde_theta(x_0 | z_t)."""
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
        """VLB KL over the absorbing posterior plus auxiliary clean-token CE."""
        validate_clean_tokens(x_0, self.config, "D3PM loss")
        require(fwd.process_type == self.forward_process_type == self.transition, (
            "D3PM loss requires the absorbing forward process"
        ))
        idx = fwd.time_index(t, min_index=1, max_index=fwd.num_timesteps)
        t_now = idx.to(device=logits.device, dtype=torch.float32) / fwd.num_timesteps
        t_prev = (idx - 1).to(device=logits.device, dtype=torch.float32) / fwd.num_timesteps
        q_unmask, q_stay = _absorbing_reverse_probs(fwd, t_now, t_prev)

        z_t = torch.where(mask, torch.full_like(x_0, self.mask_token_id), x_0)
        log_p = absorbing_posterior_log_probs(logits, z_t, t_now, t_prev, fwd, self.mask_token_id)
        log_p_x0 = log_p.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        log_p_mask = log_p[:, :, self.mask_token_id]
        if mask.any():
            kl = log_p_x0.new_zeros(x_0.shape)
            q_unmask = q_unmask.expand_as(x_0)[mask]
            q_stay = q_stay.expand_as(x_0)[mask]
            kl[mask] = _kl_two_atom(q_unmask, q_stay, log_p_x0[mask], log_p_mask[mask])
            vlb = kl.sum(dim=-1).mean() / x_0.size(1)
        else:
            vlb = logits.sum() * 0.0

        clean_logits = _clean_x0_logits(logits, self.mask_token_id)
        ce = F.cross_entropy(clean_logits.reshape(-1, clean_logits.size(-1)), x_0.reshape(-1))
        return vlb + self.hybrid_coef * ce


def absorbing_posterior_log_probs(x0_logits, z_t, t_now, t_prev, fwd, mask_token_id):
    """Build log p_theta(z_s | z_t) for an absorbing D3PM step s < t.

    `x0_logits` are clean-token logits. Observed non-mask tokens are fixed by the
    absorbing process; masked tokens mix the predicted x0 distribution with the
    probability of staying at [MASK].
    """
    require(fwd.process_type == "absorbing", "D3PM posterior requires the absorbing forward process")
    q_unmask, q_stay = _absorbing_reverse_probs(fwd, t_now, t_prev)
    clean_log_probs = F.log_softmax(_clean_x0_logits(x0_logits, mask_token_id), dim=-1)
    log_probs = clean_log_probs + _prob_log(q_unmask).unsqueeze(-1)
    log_probs[:, :, mask_token_id] = _prob_log(q_stay).expand_as(log_probs[:, :, mask_token_id])

    observed = z_t != mask_token_id
    if observed.any():
        fixed = torch.full_like(log_probs, float("-inf"))
        fixed.scatter_(-1, z_t.unsqueeze(-1), 0.0)
        log_probs = torch.where(observed.unsqueeze(-1), fixed, log_probs)
    return log_probs


def _absorbing_reverse_probs(fwd, t_now, t_prev):
    alpha_now = fwd.get_alpha(t_now).to(t_now.device).view(-1, 1)
    alpha_prev = fwd.get_alpha(t_prev).to(t_now.device).view(-1, 1)
    denom = (1 - alpha_now).clamp(min=1e-12)
    q_unmask = ((alpha_prev - alpha_now) / denom).clamp(min=0.0)
    q_stay = ((1 - alpha_prev) / denom).clamp(min=0.0)
    return q_unmask, q_stay


def _clean_x0_logits(logits, mask_token_id):
    clean_logits = logits.clone()
    clean_logits[:, :, mask_token_id] = float("-inf")
    return clean_logits


def _prob_log(prob):
    return torch.where(prob > 0, prob.clamp(min=1e-12).log(), torch.full_like(prob, float("-inf")))


def _kl_two_atom(q_unmask, q_stay, log_p_x0, log_p_mask):
    return _kl_term(q_unmask, log_p_x0) + _kl_term(q_stay, log_p_mask)


def _kl_term(q, log_p):
    term = torch.zeros_like(log_p)
    positive = q > 0
    term[positive] = q[positive] * (q[positive].clamp(min=1e-12).log() - log_p[positive])
    return term
