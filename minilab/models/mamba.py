from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.nn.norm import RMSNorm
from minilab.nn.ssm import MambaMixer
from minilab.registry import register_model


@dataclass
class MambaConfig(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    max_seq_len: int = 1024
    dropout: float = 0.0
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dt_rank: int | None = None
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4

    def __post_init__(self):
        if self.dt_rank is None:
            self.dt_rank = (self.dim + 15) // 16
        require(self.vocab_size > 0, "vocab_size must be > 0")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.d_state > 0, "d_state must be > 0")
        require(self.d_conv > 0, "d_conv must be > 0")
        require(self.expand > 0, "expand must be > 0")
        require(self.dt_rank > 0, "dt_rank must be > 0")
        require(0 < self.dt_min < self.dt_max, "dt_min and dt_max must satisfy 0 < dt_min < dt_max")
        require(self.dt_init_floor > 0, "dt_init_floor must be > 0")


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.dim)
        self.mixer = MambaMixer(
            config.dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dt_rank=config.dt_rank,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))


@register_model("mamba")
class MambaLM(BaseModel):
    config_class = MambaConfig
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([MambaBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.tok_emb.weight = self.lm_head.weight
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)

    def muon_auxiliary_modules(self):
        return (self.tok_emb, self.lm_head)

    def no_weight_decay_parameter_names(self):
        return tuple(
            name
            for name, _ in self.named_parameters()
            if name.endswith(".A_log")
        )

    def forward(self, idx, targets=None):
        logits, _ = self.forward_hidden(idx)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100)
        return logits, loss

    def forward_hidden(self, idx):
        require(idx.size(1) <= self.config.max_seq_len, (
            f"MambaLM supports at most {self.config.max_seq_len} tokens, got {idx.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        x = self.drop(x)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, x
