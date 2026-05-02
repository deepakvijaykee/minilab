"""Byte latent language model components.

The patching utilities follow the BLT interface: byte sequences are segmented
into monotonic patches, locally encoded, pooled to patch latents, processed by a
global transformer, and broadcast back to byte positions for next-byte logits.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.losses import causal_lm_cross_entropy
from minilab.models.gpt import GPTConfig, TransformerBlock
from minilab.nn.architecture import MOE_FFNS, QK_CLIP_ATTENTIONS
from minilab.registry import get_norm, get_position, register_model
from minilab.tokenizers.byte import BYTE_VOCAB_SIZE


@dataclass
class ByteLatentConfig(BaseConfig):
    vocab_size: int = BYTE_VOCAB_SIZE
    dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    max_seq_len: int = 1024
    patch_size: int = 8
    dropout: float = 0.0
    ffn_mult: float = 4.0
    attention: str = "mha"
    ffn: str = "swiglu"
    norm: str = "rmsnorm"

    def __post_init__(self):
        require(self.vocab_size == BYTE_VOCAB_SIZE, (
            f"ByteLatentLM requires byte vocab size {BYTE_VOCAB_SIZE}"
        ))
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.num_heads > 0, "num_heads must be > 0")
        require(self.dim % self.num_heads == 0, "dim must be divisible by num_heads")
        require((self.dim // self.num_heads) % 2 == 0, "RoPE requires even head dimension")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(self.patch_size > 0, "patch_size must be > 0")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.ffn_mult > 0, "ffn_mult must be > 0")


def entropy_from_logits(logits):
    probs = logits.softmax(dim=-1)
    return -(probs * probs.clamp_min(torch.finfo(probs.dtype).tiny).log()).sum(dim=-1)


def patch_start_mask_from_entropy(entropy, threshold, min_patch_size=1):
    require(threshold >= 0, "threshold must be >= 0")
    require(min_patch_size > 0, "min_patch_size must be > 0")
    require(entropy.dim() == 2, "entropy must have shape (batch, seq)")
    B, T = entropy.shape
    starts = torch.zeros(B, T, device=entropy.device, dtype=torch.bool)
    starts[:, 0] = True
    last = torch.zeros(B, device=entropy.device, dtype=torch.long)
    for pos in range(1, T):
        active = (entropy[:, pos] >= threshold) & (pos - last >= min_patch_size)
        starts[:, pos] = active
        last = torch.where(active, torch.full_like(last, pos), last)
    return starts


def static_patch_start_mask(batch_size, seq_len, patch_size, device=None):
    require(batch_size > 0, "batch_size must be > 0")
    require(seq_len > 0, "seq_len must be > 0")
    require(patch_size > 0, "patch_size must be > 0")
    starts = torch.zeros(batch_size, seq_len, device=device, dtype=torch.bool)
    starts[:, ::patch_size] = True
    return starts


def patch_ids_from_start_mask(start_mask):
    require(start_mask.dim() == 2, "start_mask must have shape (batch, seq)")
    require(start_mask.dtype == torch.bool, "start_mask must be a bool tensor")
    require(start_mask.size(1) > 0, "start_mask must contain at least one position")
    require(start_mask[:, 0].all(), "every sequence must start with a patch")
    return start_mask.long().cumsum(dim=-1) - 1


def patch_lengths_from_start_mask(start_mask):
    patch_ids = patch_ids_from_start_mask(start_mask)
    lengths = []
    for row in patch_ids:
        lengths.append(torch.bincount(row, minlength=int(row[-1].item()) + 1))
    max_patches = max(length.numel() for length in lengths)
    out = torch.zeros(len(lengths), max_patches, device=start_mask.device, dtype=torch.long)
    for i, length in enumerate(lengths):
        out[i, : length.numel()] = length
    return out


@register_model("byte_latent")
class ByteLatentLM(BaseModel):
    config_class = ByteLatentConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.byte_emb = nn.Embedding(config.vocab_size, config.dim)
        self.local_encoder = nn.Sequential(
            _CausalConv1d(config.dim, config.dim, kernel_size=3),
            nn.GELU(),
            _CausalConv1d(config.dim, config.dim, kernel_size=3),
        )
        self.max_patches = (config.max_seq_len + config.patch_size - 1) // config.patch_size
        gpt_cfg = GPTConfig(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            max_seq_len=self.max_patches,
            dropout=config.dropout,
            ffn_mult=config.ffn_mult,
            attention=config.attention,
            ffn=config.ffn,
            norm=config.norm,
        )
        self.blocks = nn.ModuleList([TransformerBlock(gpt_cfg, i) for i in range(config.num_layers)])
        self.pos_enc = get_position("rope")(config.dim // config.num_heads, gpt_cfg.max_seq_len)
        self.ln_f = get_norm(config.norm)(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.byte_emb.weight = self.lm_head.weight
        self.drop = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def muon_auxiliary_modules(self):
        return (self.byte_emb, self.lm_head)

    def set_qk_clip_recording(self, enabled):
        for block in self.blocks:
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.set_qk_clip_recording(enabled)

    def supports_qk_clip(self):
        return any(block.attention_name in QK_CLIP_ATTENTIONS for block in self.blocks)

    def forward(self, input_ids, targets=None, patch_start_mask=None):
        logits, _ = self.forward_hidden(input_ids, patch_start_mask=patch_start_mask)
        loss = None
        if targets is not None:
            loss = causal_lm_cross_entropy(logits, targets) + self.auxiliary_loss()
        return logits, loss

    def forward_hidden(self, input_ids, patch_start_mask=None):
        require(input_ids.size(1) <= self.config.max_seq_len, (
            f"ByteLatentLM supports at most {self.config.max_seq_len} bytes, got {input_ids.size(1)}"
        ))
        B, T = input_ids.shape
        if patch_start_mask is None:
            patch_start_mask = static_patch_start_mask(B, T, self.config.patch_size, input_ids.device)
        require(patch_start_mask.shape == input_ids.shape, "patch_start_mask must match input_ids")
        require(patch_start_mask.dtype == torch.bool, "patch_start_mask must be a bool tensor")
        patch_start_mask = patch_start_mask.to(input_ids.device)
        patch_ids = patch_ids_from_start_mask(patch_start_mask)
        num_patches = int(patch_ids.max().item()) + 1
        require(num_patches <= self.max_patches, "number of patches exceeds global transformer context")

        byte_hidden = self._cast_hidden(self.byte_emb(input_ids))
        local = self.local_encoder(byte_hidden.transpose(1, 2)).transpose(1, 2)
        local = self.drop(local + byte_hidden)

        pooled = torch.zeros(B, num_patches, self.config.dim, device=input_ids.device, dtype=local.dtype)
        pooled.scatter_add_(1, patch_ids.unsqueeze(-1).expand(-1, -1, self.config.dim), local)
        counts = torch.zeros(B, num_patches, 1, device=input_ids.device, dtype=local.dtype)
        counts.scatter_add_(1, patch_ids.unsqueeze(-1), torch.ones(B, T, 1, device=input_ids.device, dtype=local.dtype))
        pooled = pooled / counts.clamp_min(1.0)

        freqs_cis = self.pos_enc(num_patches)
        global_hidden = self.drop(pooled)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                def run_block(h, block=block):
                    return block(h, freqs_cis=freqs_cis, is_causal=True)
                global_hidden = torch.utils.checkpoint.checkpoint(run_block, global_hidden, use_reentrant=False)
            else:
                global_hidden = block(global_hidden, freqs_cis=freqs_cis, is_causal=True)
        prev_patch_ids = (patch_ids - 1).clamp(min=0)
        expanded = global_hidden.gather(1, prev_patch_ids.unsqueeze(-1).expand(-1, -1, self.config.dim))
        expanded = torch.where((patch_ids > 0).unsqueeze(-1), expanded, torch.zeros_like(expanded))
        hidden = self.ln_f(local + expanded)
        return self.lm_head(hidden), hidden

    def auxiliary_loss(self):
        loss = next(self.parameters()).sum() * 0.0
        if self.config.ffn not in MOE_FFNS:
            return loss
        for block in self.blocks:
            loss = loss + block.ffn.aux_loss
        return loss

    def post_optimizer_step(self, qk_clip_threshold, qk_clip_balance):
        if self.config.ffn == "aux_free_moe":
            for block in self.blocks:
                block.ffn.commit_routing_bias_update()
        if qk_clip_threshold <= 0:
            return
        for block in self.blocks:
            if block.attention_name in QK_CLIP_ATTENTIONS:
                block.attn.commit_qk_clip_update(qk_clip_threshold, qk_clip_balance)


class _CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        require(kernel_size > 0, "causal convolution kernel_size must be > 0")
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=0)

    def forward(self, x):
        return self.conv(F.pad(x, (self.left_padding, 0)))
