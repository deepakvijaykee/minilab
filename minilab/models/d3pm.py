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
from minilab.models.diffusion_base import (
    DiffusionBackboneMixin,
    DiffusionModelConfig,
    loss_normalizer,
    supervised_diffusion_mask,
    validate_clean_tokens,
    validate_loss_mask,
)
from minilab.registry import register_model


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
class D3PM(DiffusionBackboneMixin, BaseModel):
    config_class = D3PMConfig
    forward_process_type = "absorbing"
    reverse_parameterization = "d3pm_x0_logits"
    requires_terminal_mask_prior = True

    def __init__(self, config):
        super().__init__(config)
        self._init_diffusion_backbone(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.transition = config.transition
        self.hybrid_coef = config.hybrid_coef
        self.apply(self._init_weights)

    def forward(self, z_t, t):
        """Predicts clean-token logits p_tilde_theta(x_0 | z_t)."""
        return self.lm_head(self._diffusion_backbone_forward(z_t, t, "D3PM"))

    def compute_loss_per_example(self, logits, x_0, mask, t, fwd, loss_mask=None, normalization="sequence"):
        """VLB KL over the absorbing posterior plus auxiliary clean-token CE."""
        validate_clean_tokens(x_0, self.config, "D3PM loss")
        loss_mask = validate_loss_mask(loss_mask, x_0, "D3PM loss")
        require(fwd.process_type == self.forward_process_type == self.transition, (
            "D3PM loss requires the absorbing forward process"
        ))
        target_mask = supervised_diffusion_mask(mask, loss_mask)
        idx = fwd.time_index(t, min_index=1, max_index=fwd.num_timesteps)
        t_now = idx.to(device=logits.device, dtype=torch.float32) / fwd.num_timesteps
        t_prev = (idx - 1).to(device=logits.device, dtype=torch.float32) / fwd.num_timesteps
        q_unmask, q_stay = _absorbing_reverse_probs(fwd, t_now, t_prev)

        z_t = torch.where(mask, torch.full_like(x_0, self.mask_token_id), x_0)
        log_p = absorbing_posterior_log_probs(logits, z_t, t_now, t_prev, fwd, self.mask_token_id)
        log_p_x0 = log_p.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        log_p_mask = log_p[:, :, self.mask_token_id]
        kl = log_p_x0.new_zeros(x_0.shape)
        q_unmask = q_unmask.expand_as(x_0)[target_mask]
        q_stay = q_stay.expand_as(x_0)[target_mask]
        kl[target_mask] = _kl_two_atom(q_unmask, q_stay, log_p_x0[target_mask], log_p_mask[target_mask])
        vlb = kl.sum(dim=-1) / loss_normalizer(x_0, loss_mask, normalization).to(logits.device, dtype=logits.dtype)

        clean_logits = _clean_x0_logits(logits, self.mask_token_id)
        token_ce = F.cross_entropy(
            clean_logits.reshape(-1, clean_logits.size(-1)),
            x_0.reshape(-1),
            reduction="none",
        ).view_as(x_0)
        ce_mask = torch.ones_like(x_0, dtype=torch.bool) if loss_mask is None else loss_mask
        ce = (token_ce * ce_mask.float()).sum(dim=-1)
        ce = ce / loss_normalizer(x_0, loss_mask, normalization).to(logits.device, dtype=logits.dtype)
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
