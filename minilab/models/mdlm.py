"""MDLM: Masked Diffusion Language Model.
Full SUBS parameterization: (1) never predict [MASK], (2) carry over observed
tokens at unmasked positions. Loss computed only on masked positions.
Sahoo et al., NeurIPS 2024."""

from dataclasses import dataclass

import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.models.diffusion_base import (
    apply_subs_clean_logits,
    DiffusionBackboneMixin,
    DiffusionModelConfig,
    loss_normalizer,
    validate_clean_tokens,
    validate_loss_mask,
)
from minilab.registry import register_model


@dataclass
class MDLMConfig(DiffusionModelConfig):
    """Configuration for masked diffusion language modeling."""


@register_model("mdlm")
class MDLM(DiffusionBackboneMixin, BaseModel):
    config_class = MDLMConfig
    forward_process_type = "absorbing"
    reverse_parameterization = "clean_logits"
    requires_terminal_mask_prior = True

    def __init__(self, config):
        super().__init__(config)
        self._init_diffusion_backbone(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def forward(self, z_t, t):
        x = self._diffusion_backbone_forward(z_t, t, "MDLM")
        return apply_subs_clean_logits(self.lm_head(x), z_t, self.mask_token_id)

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
