"""Sparse Mixture of Experts FFN: top-k routing over N expert FFNs.
Auxiliary load-balancing loss encourages uniform expert utilization."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.registry import register_ffn


@register_ffn("moe")
class MoEFFN(nn.Module):

    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2, aux_loss_coef=0.01):
        super().__init__()
        require(dim > 0, "MoEFFN dim must be > 0")
        require(hidden_dim > 0, "MoEFFN hidden_dim must be > 0")
        require(num_experts > 0, "num_experts must be > 0")
        require(1 <= top_k <= num_experts, "top_k must be in [1, num_experts]")
        require(aux_loss_coef >= 0, "aux_loss_coef must be >= 0")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = indices[:, k]
            w = weights[:, k].unsqueeze(-1)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x_flat[mask])

        # Load balancing: num_experts * sum(fraction_routed_i * mean_prob_i)
        probs = F.softmax(logits, dim=-1)
        frac = F.one_hot(indices, self.num_experts).float().mean(dim=(0, 1))
        aux_loss = self.aux_loss_coef * self.num_experts * (frac * probs.mean(0)).sum()

        self.aux_loss = aux_loss
        return out.reshape(B, T, C)


class _Expert(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        require(dim > 0, "expert dim must be > 0")
        require(hidden_dim > 0, "expert hidden_dim must be > 0")
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
