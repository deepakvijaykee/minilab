import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require


class LoRALinear(nn.Module):
    """Low-rank update around a frozen linear layer."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        require(isinstance(base, nn.Linear), "LoRALinear base must be nn.Linear")
        require(type(rank) is int and rank > 0, "LoRA rank must be a positive integer")
        require(isinstance(alpha, (int, float)) and math.isfinite(alpha) and alpha > 0, (
            "LoRA alpha must be a finite positive number"
        ))
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        update = F.linear(F.linear(x, self.lora_a), self.lora_b)
        return self.base(x) + update * self.scaling
