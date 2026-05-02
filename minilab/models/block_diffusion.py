"""Block diffusion language model utilities.

This module implements the BD3-style block attention contract used by block
diffusion LMs: noisy-token queries attend within their own noisy block and to
clean-context blocks to their left; clean-context tokens attend causally by
block. The compact model below is a scoped reference path for minilab-scale
experiments, not checkpoint-compatible with the original DiT implementation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel, apply_conditional_diffusion_mask
from minilab.checks import require
from minilab.models.diffusion_base import (
    apply_subs_clean_logits,
    DiffusionBackboneMixin,
    DiffusionModelConfig,
    loss_normalizer,
    supervised_diffusion_mask,
    validate_clean_tokens,
    validate_loss_mask,
)
from minilab.registry import register_model


@dataclass
class BlockDiffusionConfig(DiffusionModelConfig):
    block_size: int = 32
    cross_attention: bool = True
    antithetic_time_sampling: bool = False

    def __post_init__(self):
        super().__post_init__()
        require(self.block_size > 0, "block_size must be > 0")


def block_diffusion_attention_mask(seq_len, block_size, cross_attention=True, device=None):
    """Boolean allow-mask matching the BD3 block-diffusion attention pattern.

    With `cross_attention=True`, the sequence is interpreted as `[x_t, x_0]`
    with `seq_len` tokens in each half. The returned mask has shape
    `(2 * seq_len, 2 * seq_len)`.
    """
    require(seq_len > 0, "seq_len must be > 0")
    require(block_size > 0, "block_size must be > 0")
    if not cross_attention:
        idx = torch.arange(seq_len, device=device)
        return (idx[:, None] // block_size) == (idx[None, :] // block_size)

    idx = torch.arange(2 * seq_len, device=device)
    q = idx[:, None]
    kv = idx[None, :]
    q_is_x0 = q >= seq_len
    kv_is_x0 = kv >= seq_len
    q_block = (q % seq_len) // block_size
    kv_block = (kv % seq_len) // block_size

    block_diagonal = (q_block == kv_block) & (q_is_x0 == kv_is_x0)
    noisy_to_left_clean = (q_block > kv_block) & kv_is_x0 & (~q_is_x0)
    clean_block_causal = (q_block >= kv_block) & kv_is_x0 & q_is_x0
    return block_diagonal | noisy_to_left_clean | clean_block_causal


def block_mask_to_attn_bias(allow_mask, dtype):
    bias = torch.zeros(allow_mask.shape, device=allow_mask.device, dtype=dtype)
    return bias.masked_fill(~allow_mask, float("-inf"))


def sample_block_times(batch_size, seq_len, block_size, device, antithetic=False):
    require(batch_size > 0, "batch_size must be > 0")
    require(seq_len > 0, "seq_len must be > 0")
    require(block_size > 0, "block_size must be > 0")
    num_blocks = (seq_len + block_size - 1) // block_size
    if antithetic:
        half = (batch_size + 1) // 2
        base = torch.rand(half, num_blocks, device=device)
        paired = torch.cat([base, 1.0 - base], dim=0)[:batch_size]
    else:
        paired = torch.rand(batch_size, num_blocks, device=device)
    return paired.repeat_interleave(block_size, dim=1)[:, :seq_len]


def block_mask_probabilities(forward_process, block_t):
    require(forward_process.process_type == "absorbing", (
        "block diffusion mask probabilities require the absorbing forward process"
    ))
    require((block_t >= 0).all() and (block_t <= 1).all(), "block_t values must be in [0, 1]")
    alpha = forward_process.alpha_at(block_t).to(block_t.device)
    return (1.0 - alpha).clamp(min=0.0, max=1.0)


def block_absorbing_q_sample(x_0, mask_prob, mask_token_id):
    require(x_0.shape == mask_prob.shape, "mask_prob must have the same shape as x_0")
    require((mask_prob >= 0).all() and (mask_prob <= 1).all(), "mask_prob values must be in [0, 1]")
    mask = torch.rand(x_0.shape, device=x_0.device) < mask_prob
    return torch.where(mask, torch.full_like(x_0, mask_token_id), x_0), mask


@register_model("block_diffusion")
class BlockDiffusionLM(DiffusionBackboneMixin, BaseModel):
    config_class = BlockDiffusionConfig
    forward_process_type = "absorbing"
    reverse_parameterization = "clean_logits"
    requires_terminal_mask_prior = True

    def __init__(self, config):
        super().__init__(config)
        pos_len = config.max_seq_len * (2 if config.cross_attention else 1)
        self._init_diffusion_backbone(config, pos_len=pos_len)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def supports_unconditional_diffusion_sampling(self):
        return not self.config.cross_attention

    def diffusion_forward_kwargs(self, x_0):
        if self.config.cross_attention:
            return {"x_0": x_0}
        return {}

    def diffusion_training_state(self, forward_process, x_0, device):
        if not self.config.cross_attention:
            return super().diffusion_training_state(forward_process, x_0, device)
        require(forward_process.process_type == self.forward_process_type, (
            "BlockDiffusionLM training requires the absorbing forward process"
        ))
        z_t, mask, block_t = self.sample_training_state(forward_process, x_0)
        return z_t, mask, block_t, self.diffusion_forward_kwargs(x_0)

    def diffusion_conditional_training_state(self, forward_process, x_0, loss_mask, valid_mask, device, t=None):
        require(forward_process.process_type == self.forward_process_type, (
            "BlockDiffusionLM conditional training requires the absorbing forward process"
        ))
        validate_clean_tokens(x_0, self.config, "BlockDiffusionLM conditional training")
        loss_mask = validate_loss_mask(loss_mask, x_0, "BlockDiffusionLM conditional training")
        if t is None:
            block_t = sample_block_times(
                x_0.size(0),
                x_0.size(1),
                self.config.block_size,
                device,
                antithetic=self.config.antithetic_time_sampling,
            )
        else:
            t = t.to(device)
            require(t.shape == (x_0.size(0),) or t.shape == x_0.shape, (
                "BlockDiffusionLM conditional time must have shape (batch,) or (batch, seq)"
            ))
            block_t = t.view(-1, 1).expand_as(x_0) if t.dim() == 1 else t
        mask_prob = block_mask_probabilities(forward_process, block_t)
        z_t, mask = block_absorbing_q_sample(x_0, mask_prob, self.mask_token_id)
        z_t, mask = apply_conditional_diffusion_mask(
            z_t,
            mask,
            x_0,
            loss_mask,
            valid_mask,
            self.mask_token_id,
        )
        return z_t, mask, block_t, self.diffusion_forward_kwargs(x_0)

    def sample_training_state(self, forward_process, x_0):
        validate_clean_tokens(x_0, self.config, "BlockDiffusionLM sample_training_state")
        require(forward_process.process_type == self.forward_process_type, (
            "BlockDiffusionLM sample_training_state requires the absorbing forward process"
        ))
        block_t = sample_block_times(
            x_0.size(0),
            x_0.size(1),
            self.config.block_size,
            x_0.device,
            antithetic=self.config.antithetic_time_sampling,
        )
        mask_prob = block_mask_probabilities(forward_process, block_t)
        z_t, mask = block_absorbing_q_sample(x_0, mask_prob, self.mask_token_id)
        return z_t, mask, block_t

    def forward(self, z_t, t, x_0=None):
        require(z_t.size(1) <= self.config.max_seq_len, (
            f"BlockDiffusionLM supports at most {self.config.max_seq_len} clean tokens, got {z_t.size(1)}"
        ))
        require(t.shape == (z_t.size(0),) or t.shape == z_t.shape, (
            "t must have shape (batch,) or (batch, seq)"
        ))

        if self.config.cross_attention:
            require(x_0 is not None, "cross-attention block diffusion forward requires clean x_0 context")
            validate_clean_tokens(x_0, self.config, "BlockDiffusionLM forward")
            require(x_0.shape == z_t.shape, "x_0 must have the same shape as z_t")
            tokens = torch.cat([z_t, x_0], dim=1)
            allow = block_diffusion_attention_mask(z_t.size(1), self.config.block_size, True, z_t.device)
        else:
            tokens = z_t
            allow = block_diffusion_attention_mask(z_t.size(1), self.config.block_size, False, z_t.device)

        x = self._cast_hidden(self.tok_emb(tokens))
        attn_bias = block_mask_to_attn_bias(allow, x.dtype)
        t_emb = self.time_emb(t.to(z_t.device))
        if t_emb.dim() == 3 and self.config.cross_attention:
            t_emb = torch.cat([t_emb, t_emb], dim=1)
        freqs_cis = self.pos_enc(tokens.size(1))
        for block in self.blocks:
            x = self._checkpointed_forward(block, x, t_emb, freqs_cis=freqs_cis, attn_bias=attn_bias)
        x = self.ln_f(x[:, : z_t.size(1)])
        return apply_subs_clean_logits(self.lm_head(x), z_t, self.mask_token_id)

    def compute_loss_per_example(self, logits, x_0, mask, t, fwd, loss_mask=None, normalization="sequence"):
        validate_clean_tokens(x_0, self.config, "BlockDiffusionLM loss")
        loss_mask = validate_loss_mask(loss_mask, x_0, "BlockDiffusionLM loss")
        require(fwd.process_type == self.forward_process_type, (
            "BlockDiffusionLM loss requires the absorbing forward process"
        ))
        target_mask = supervised_diffusion_mask(mask, loss_mask)
        log_probs = F.log_softmax(logits, dim=-1)
        log_p_x0 = log_probs.gather(-1, x_0.unsqueeze(-1)).squeeze(-1)
        target_mask_f = target_mask.float()
        if t.dim() == 2:
            require(t.shape == x_0.shape, "BlockDiffusionLM token-wise loss times must match x_0")
            weights = fwd.get_weight(t.to(logits.device)).to(logits.dtype)
            per_ex_loglik = (weights * log_p_x0 * target_mask_f).sum(dim=-1)
        else:
            weights = fwd.get_weight(t.to(logits.device)).to(logits.dtype).view(-1, 1)
            per_ex_loglik = (weights * log_p_x0 * target_mask_f).sum(dim=-1)
        denom = loss_normalizer(x_0, loss_mask, normalization).to(logits.device, dtype=logits.dtype)
        return -per_ex_loglik / denom
