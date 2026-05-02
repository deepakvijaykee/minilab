from dataclasses import dataclass

import torch
import torch.nn as nn

from minilab.base import BaseModel
from minilab.checks import require
from minilab.config import BaseConfig
from minilab.nn.norm import RMSNorm
from minilab.nn.ssm import Mamba2Mixer
from minilab.registry import register_model


@dataclass
class Mamba2Config(BaseConfig):
    vocab_size: int = 50257
    dim: int = 512
    num_layers: int = 6
    max_seq_len: int = 1024
    dropout: float = 0.0
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    A_init_min: float = 1.0
    A_init_max: float = 16.0

    def __post_init__(self):
        require(self.vocab_size > 0, "vocab_size must be > 0")
        require(self.dim > 0, "dim must be > 0")
        require(self.num_layers > 0, "num_layers must be > 0")
        require(self.max_seq_len > 0, "max_seq_len must be > 0")
        require(0.0 <= self.dropout < 1.0, "dropout must be in [0, 1)")
        require(self.d_state > 0, "d_state must be > 0")
        require(self.d_conv > 0, "d_conv must be > 0")
        require(self.expand > 0, "expand must be > 0")
        require(self.headdim > 0, "headdim must be > 0")
        require((self.expand * self.dim) % self.headdim == 0, "expand * dim must be divisible by headdim")
        require(self.ngroups > 0, "ngroups must be > 0")
        require((self.expand * self.dim // self.headdim) % self.ngroups == 0, (
            "Mamba2 nheads must be divisible by ngroups"
        ))
        require(0 < self.dt_min < self.dt_max, "dt_min and dt_max must satisfy 0 < dt_min < dt_max")
        require(self.dt_init_floor > 0, "dt_init_floor must be > 0")
        require(0 < self.A_init_min <= self.A_init_max, "A init range must satisfy 0 < min <= max")


class Mamba2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.dim)
        self.mixer = Mamba2Mixer(
            config.dim,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            headdim=config.headdim,
            ngroups=config.ngroups,
            dt_min=config.dt_min,
            dt_max=config.dt_max,
            dt_init_floor=config.dt_init_floor,
            A_init_min=config.A_init_min,
            A_init_max=config.A_init_max,
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x):
        return x + self.drop(self.mixer(self.norm(x)))


@register_model("mamba2")
class Mamba2LM(BaseModel):
    config_class = Mamba2Config
    provides_hidden_states = True

    def __init__(self, config):
        super().__init__(config)
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Mamba2Block(config) for _ in range(config.num_layers)])
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
            if name.endswith(".A_log") or name.endswith(".dt_bias")
        )

    def forward(self, idx, targets=None):
        return self._causal_lm_forward(idx, targets)

    def forward_hidden(self, idx):
        require(idx.size(1) <= self.config.max_seq_len, (
            f"Mamba2LM supports at most {self.config.max_seq_len} tokens, got {idx.size(1)}"
        ))
        x = self._cast_hidden(self.tok_emb(idx))
        x = self.drop(x)
        for block in self.blocks:
            if self._gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x), x
