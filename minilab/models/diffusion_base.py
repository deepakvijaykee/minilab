from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.checks import require
from minilab.config import BaseConfig
from minilab.nn.architecture import (
    GQA_ATTENTIONS,
    MOE_FFNS,
    QK_CLIP_ATTENTIONS,
    TOP_ONE_MOE_FFNS,
    resolve_deepseek_v4_attention,
)
from minilab.nn.diffusion import (
    commit_diffusion_block_updates,
    DiffusionBlock,
    diffusion_blocks_auxiliary_loss,
    SinusoidalTimeEmbedding,
)
from minilab.registry import get_position

_UNSUPPORTED_DIFFUSION_ATTENTIONS = {
    "cosformer",
    "lightning",
    "gated_deltanet",
    "qwen3_next",
    "gemma3",
    "gemma4",
}
_DEFAULT_NUM_EXPERTS = 8
_DEFAULT_TOP_K_EXPERTS = 2


@dataclass
class DiffusionModelConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.0
    ffn_mult: float = 4.0
    attention: str = "mha"
    num_kv_heads: int | None = None
    ffn: str = "swiglu"
    num_experts: int = _DEFAULT_NUM_EXPERTS
    top_k_experts: int = _DEFAULT_TOP_K_EXPERTS
    mask_token_id: int = 50256
    time_sampling: str = "continuous"

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        require(self.vocab_size > 1, "diffusion vocab_size must include at least one clean token plus [MASK]")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.num_kv_heads > 0, "num_kv_heads must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(self.dim % self.num_heads == 0, "dim must be divisible by num_heads")
        require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.ffn_mult > 0, "ffn_mult must be > 0")
        require(self.attention not in _UNSUPPORTED_DIFFUSION_ATTENTIONS, (
            f"diffusion backbones use bidirectional RoPE attention; unsupported attention={self.attention!r}"
        ))
        if resolve_deepseek_v4_attention(self.attention, 0) in GQA_ATTENTIONS:
            require(self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads")
        else:
            require(self.num_kv_heads == self.num_heads, "num_kv_heads only applies to GQA attention variants")
        if self.ffn in MOE_FFNS:
            require(self.num_experts > 0, "num_experts must be > 0")
            require(1 <= self.top_k_experts <= self.num_experts, "top_k_experts must be in [1, num_experts]")
            if self.ffn in TOP_ONE_MOE_FFNS:
                require(self.top_k_experts == 1, f"{self.ffn} requires top_k_experts=1")
        else:
            require(
                self.num_experts == _DEFAULT_NUM_EXPERTS and self.top_k_experts == _DEFAULT_TOP_K_EXPERTS,
                "num_experts and top_k_experts only apply to MoE FFNs",
            )
        require(self.mask_token_id == self.vocab_size - 1, (
            "diffusion models reserve mask_token_id as the final vocab id; "
            "set vocab_size=data_vocab_size+1 and mask_token_id=data_vocab_size"
        ))
        require(self.time_sampling in {"continuous", "discrete"}, (
            "time_sampling must be 'continuous' or 'discrete'"
        ))


def validate_clean_tokens(x_0, config, context):
    require(x_0.dtype == torch.long, f"{context} requires integer token ids")
    require((x_0 >= 0).all(), f"{context} clean tokens must be non-negative")
    require((x_0 < config.mask_token_id).all(), f"{context} clean tokens must exclude the reserved [MASK] token")


def validate_infill_tokens(tokens, mask_positions, config, context):
    require(tokens.shape == mask_positions.shape, f"{context} tokens and mask_positions must have the same shape")
    require(mask_positions.dtype == torch.bool, f"{context} mask_positions must be a bool tensor")
    require(tokens.dtype == torch.long, f"{context} requires integer token ids")
    require((tokens >= 0).all(), f"{context} tokens must be non-negative")
    require((tokens <= config.mask_token_id).all(), (
        f"{context} tokens must be clean token ids or the reserved [MASK] placeholder"
    ))
    context_tokens = tokens[~mask_positions]
    require((context_tokens < config.mask_token_id).all(), (
        f"{context} context tokens must exclude the reserved [MASK] token"
    ))


def validate_loss_mask(loss_mask, x_0, context):
    if loss_mask is None:
        return None
    require(loss_mask.shape == x_0.shape, f"{context} loss_mask must have the same shape as x_0")
    require(loss_mask.dtype == torch.bool, f"{context} loss_mask must be a bool tensor")
    require(loss_mask.any(dim=-1).all(), f"{context} requires at least one supervised token per example")
    return loss_mask.to(x_0.device)


def supervised_diffusion_mask(noised_mask, loss_mask):
    return noised_mask if loss_mask is None else (noised_mask & loss_mask)


def loss_normalizer(x_0, loss_mask=None, normalization="sequence"):
    require(normalization in {"sequence", "target", "none"}, "normalization must be 'sequence', 'target', or 'none'")
    if normalization == "none":
        return torch.ones((x_0.size(0),), device=x_0.device, dtype=torch.float32)
    if normalization == "sequence" or loss_mask is None:
        return torch.full((x_0.size(0),), x_0.size(1), device=x_0.device, dtype=torch.float32)
    return loss_mask.float().sum(dim=-1).clamp(min=1.0)


def apply_subs_clean_logits(logits, z_t, mask_token_id):
    """SUBS parameterization: never predict [MASK], carry observed tokens through."""
    logits = logits.clone()
    logits[:, :, mask_token_id] = float("-inf")
    unmasked = z_t != mask_token_id
    if unmasked.any():
        carry = torch.full_like(logits, float("-inf"))
        carry.scatter_(-1, z_t.unsqueeze(-1), 0.0)
        logits = torch.where(unmasked.unsqueeze(-1), carry, logits)
    return logits


class DiffusionBackboneMixin:
    def _init_diffusion_backbone(self, config, pos_len=None):
        head_dim = config.dim // config.num_heads
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.time_emb = SinusoidalTimeEmbedding(config.dim)
        self.pos_enc = get_position("rope")(head_dim, pos_len or config.max_seq_len)
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
        self.mask_token_id = config.mask_token_id

    def _diffusion_backbone_forward(self, z_t, t, context):
        require(z_t.size(1) <= self.config.max_seq_len, (
            f"{context} supports at most {self.config.max_seq_len} tokens, got {z_t.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(z_t))
        t_emb = self.time_emb(t)
        freqs_cis = self.pos_enc(z_t.size(1))
        for block in self.blocks:
            x = self._checkpointed_forward(block, x, t_emb, freqs_cis=freqs_cis)
        return self.ln_f(x)

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.lm_head)

    def auxiliary_loss(self):
        return diffusion_blocks_auxiliary_loss(self.blocks, self.tok_emb.weight)

    def set_qk_clip_recording(self, enabled):
        for block in self.blocks:
            block.set_qk_clip_recording(enabled)

    def supports_qk_clip(self):
        return any(block.attention_name in QK_CLIP_ATTENTIONS for block in self.blocks)

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        commit_diffusion_block_updates(self.blocks, qk_clip_threshold, qk_clip_balance)

    def compute_loss(self, output, x_0, mask, t, fwd):
        return self.compute_loss_per_example(output, x_0, mask, t, fwd).mean()
