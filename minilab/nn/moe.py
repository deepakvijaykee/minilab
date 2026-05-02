"""Sparse Mixture of Experts FFNs.

The registry includes compact routing families used by modern sparse LMs:
token-choice top-k MoE, Switch top-1 routing, Mixtral top-2 routing,
expert-choice routing, DeepSeek-style shared+routed experts, Qwen3-Next-style
gated shared experts, auxiliary-loss-free biased routing, and BASE-style
balanced assignment.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.registry import register_ffn


@register_ffn("moe")
class MoEFFN(nn.Module):

    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2, aux_loss_coef=0.01):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, "MoEFFN")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        out = _combine_token_choice(x_flat, self.experts, weights, indices)

        # Load balancing: num_experts * sum(fraction_routed_i * mean_prob_i)
        self.aux_loss = _load_balancing_loss(logits, indices, self.num_experts, self.aux_loss_coef)
        return out.reshape(B, T, C)


@register_ffn("mixtral_moe")
class MixtralMoEFFN(MoEFFN):
    """Mixtral-style sparse FFN: token-choice top-2 routing by default."""

    def __init__(self, dim, hidden_dim, num_experts=8, top_k=2, aux_loss_coef=0.01):
        super().__init__(dim, hidden_dim, num_experts=num_experts, top_k=top_k, aux_loss_coef=aux_loss_coef)


@register_ffn("switch_moe")
class SwitchMoEFFN(nn.Module):
    """Switch Transformer top-1 expert routing with per-expert capacity."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=8,
        top_k=1,
        aux_loss_coef=0.01,
        capacity_factor=1.25,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, 1, aux_loss_coef, "SwitchMoEFFN")
        require(top_k == 1, "SwitchMoEFFN implements Switch top-1 routing; set top_k=1")
        require(capacity_factor > 0, "capacity_factor must be > 0")
        self.num_experts = num_experts
        self.top_k = 1
        self.aux_loss_coef = aux_loss_coef
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)
        self.register_buffer("dropped_fraction", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.max(dim=-1)
        capacity = _expert_capacity(x_flat.size(0), self.num_experts, self.capacity_factor)

        out = torch.zeros_like(x_flat)
        kept = torch.zeros(x_flat.size(0), device=x.device, dtype=torch.bool)
        for e, expert in enumerate(self.experts):
            token_idx = (indices == e).nonzero(as_tuple=False).flatten()
            if token_idx.numel() > capacity:
                token_idx = token_idx[:capacity]
            if token_idx.numel() > 0:
                kept[token_idx] = True
                out[token_idx] = weights[token_idx].unsqueeze(-1) * expert(x_flat[token_idx])

        self.dropped_fraction = 1.0 - kept.float().mean()
        self.aux_loss = _load_balancing_loss(logits, indices.unsqueeze(-1), self.num_experts, self.aux_loss_coef)
        return out.reshape(B, T, C)


@register_ffn("expert_choice_moe")
class ExpertChoiceMoEFFN(nn.Module):
    """Expert Choice MoE: each expert selects its highest-scoring tokens."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=8,
        top_k=2,
        aux_loss_coef=0.0,
        capacity_factor=1.0,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, "ExpertChoiceMoEFFN")
        require(capacity_factor > 0, "capacity_factor must be > 0")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        capacity = self._expert_capacity_for(x_flat.size(0))

        out = torch.zeros_like(x_flat)
        denom = torch.zeros(x_flat.size(0), 1, device=x.device, dtype=x.dtype)
        for e, expert in enumerate(self.experts):
            count = min(capacity, x_flat.size(0))
            values, token_idx = probs[:, e].topk(count, dim=0)
            expert_out = expert(x_flat[token_idx])
            weights = values.to(dtype=x.dtype).unsqueeze(-1)
            out[token_idx] += weights * expert_out
            denom[token_idx] += weights

        selected = denom.squeeze(-1) > 0
        out[selected] = out[selected] / denom[selected].clamp(min=torch.finfo(out.dtype).tiny)
        self.aux_loss = logits.sum() * 0.0
        return out.reshape(B, T, C)

    def _expert_capacity_for(self, num_tokens):
        return _expert_capacity(num_tokens, self.num_experts, self.capacity_factor * self.top_k)


@register_ffn("deepseek_moe")
class DeepSeekMoEFFN(nn.Module):
    """DeepSeekMoE-style FFN with always-on shared experts plus sparse routed experts."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=8,
        top_k=2,
        aux_loss_coef=0.01,
        num_shared_experts=1,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, "DeepSeekMoEFFN")
        require(num_shared_experts > 0, "num_shared_experts must be > 0")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.routed_experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.shared_experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_shared_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        weights, indices = logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)

        routed = _combine_token_choice(x_flat, self.routed_experts, weights, indices)
        shared = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared = shared + expert(x_flat)

        self.aux_loss = _load_balancing_loss(logits, indices, self.num_experts, self.aux_loss_coef)
        return (routed + shared).reshape(B, T, C)


@register_ffn("qwen3_next_moe")
class Qwen3NextMoEFFN(nn.Module):
    """Qwen3-Next-style sparse MoE with a gated shared expert.

    The released configs route each token to a small top-k subset of many
    experts, renormalize the selected probabilities, and add a separate shared
    expert path with a learned scalar gate.
    """

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=8,
        top_k=2,
        aux_loss_coef=0.001,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, "Qwen3NextMoEFFN")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.routed_experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.shared_expert = _Expert(dim, hidden_dim)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(weights.dtype).tiny)

        routed = _combine_token_choice(x_flat, self.routed_experts, weights, indices)
        shared_gate = torch.sigmoid(self.shared_expert_gate(x_flat))
        shared = shared_gate * self.shared_expert(x_flat)

        self.aux_loss = _load_balancing_loss(logits, indices, self.num_experts, self.aux_loss_coef)
        return (routed + shared).reshape(B, T, C)


@register_ffn("aux_free_moe")
class AuxFreeMoEFFN(nn.Module):
    """Auxiliary-loss-free biased top-k routing from DeepSeek-V3."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=8,
        top_k=2,
        aux_loss_coef=0.0,
        bias_update_rate=1e-3,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, "AuxFreeMoEFFN")
        require(bias_update_rate >= 0, "bias_update_rate must be >= 0")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.bias_update_rate = bias_update_rate
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.register_buffer("routing_bias", torch.zeros(num_experts))
        self.register_buffer("_routing_load_sum", torch.zeros(num_experts))
        self.register_buffer("_routing_load_count", torch.zeros(()))
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        scores = torch.sigmoid(logits)
        _, indices = (scores + self.routing_bias).topk(self.top_k, dim=-1)
        weights = scores.gather(-1, indices)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(weights.dtype).tiny)

        out = _combine_token_choice(x_flat, self.experts, weights, indices)
        self._record_routing_load(indices)
        self.aux_loss = logits.sum() * 0.0
        return out.reshape(B, T, C)

    def _record_routing_load(self, indices):
        if not torch.is_grad_enabled() or self.bias_update_rate == 0:
            return
        with torch.no_grad():
            load = F.one_hot(indices, self.num_experts).float().mean(dim=(0, 1))
            self._routing_load_sum.add_(load.to(self._routing_load_sum.dtype))
            self._routing_load_count.add_(1)

    @torch.no_grad()
    def commit_routing_bias_update(self):
        if self._routing_load_count.item() == 0 or self.bias_update_rate == 0:
            return
        load = self._routing_load_sum / self._routing_load_count
        target = torch.full_like(load, 1.0 / self.num_experts)
        self.routing_bias.add_(self.bias_update_rate * torch.sign(target - load).to(self.routing_bias.dtype))
        self._routing_load_sum.zero_()
        self._routing_load_count.zero_()


@register_ffn("base_moe")
class BaseMoEFFN(nn.Module):
    """BASE-layer style balanced assignment: one token per routed expert slot."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=8,
        top_k=1,
        aux_loss_coef=0.0,
        capacity_factor=1.0,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, 1, aux_loss_coef, "BaseMoEFFN")
        require(top_k == 1, "BaseMoEFFN implements one expert per token; set top_k=1")
        require(capacity_factor >= 1.0, "BASE balanced assignment requires capacity_factor >= 1")
        self.num_experts = num_experts
        self.top_k = 1
        self.aux_loss_coef = aux_loss_coef
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(dim, hidden_dim) for _ in range(num_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        logits = self.gate(x_flat)
        probs = F.softmax(logits, dim=-1)
        capacity = _expert_capacity(x_flat.size(0), self.num_experts, self.capacity_factor)
        indices = _balanced_assignment(logits.detach(), capacity).to(logits.device)
        weights = probs.gather(-1, indices.unsqueeze(-1))
        out = _combine_token_choice(x_flat, self.experts, weights, indices.unsqueeze(-1))
        self.aux_loss = logits.sum() * 0.0
        return out.reshape(B, T, C)


@register_ffn("gemma4_moe")
class Gemma4MoEFFN(nn.Module):
    """Gemma-style sparse FFN: RMS-normalized router, top-k weights, GELU experts."""

    def __init__(
        self,
        dim,
        hidden_dim,
        num_experts=128,
        top_k=8,
        aux_loss_coef=0.0,
    ):
        super().__init__()
        _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, "Gemma4MoEFFN")
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_coef = aux_loss_coef
        self.router_scale = nn.Parameter(torch.ones(dim))
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([_GELUTanhExpert(dim, hidden_dim) for _ in range(num_experts)])
        self.register_buffer("aux_loss", torch.tensor(0.0), persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.reshape(-1, C)
        routed_input = _rms_without_scale(x_flat) * self.router_scale.to(x_flat.dtype) * (C ** -0.5)
        logits = self.gate(routed_input)
        probs = F.softmax(logits, dim=-1)
        weights, indices = probs.topk(self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(weights.dtype).tiny)
        weights = weights * self.per_expert_scale[indices].to(weights.dtype)
        out = _combine_token_choice(x_flat, self.experts, weights, indices)
        self.aux_loss = _load_balancing_loss(logits, indices, self.num_experts, self.aux_loss_coef)
        return out.reshape(B, T, C)


class _ExpertBase(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1, self.w2, self.w3 = _expert_projections(dim, hidden_dim)


class _Expert(_ExpertBase):

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class _GELUTanhExpert(_ExpertBase):
    def forward(self, x):
        return self.w3(F.gelu(self.w1(x), approximate="tanh") * self.w2(x))


def _expert_projections(dim, hidden_dim):
    require(dim > 0, "expert dim must be > 0")
    require(hidden_dim > 0, "expert hidden_dim must be > 0")
    return (
        nn.Linear(dim, hidden_dim, bias=False),
        nn.Linear(dim, hidden_dim, bias=False),
        nn.Linear(hidden_dim, dim, bias=False),
    )


def _validate_moe_config(dim, hidden_dim, num_experts, top_k, aux_loss_coef, name):
    require(dim > 0, f"{name} dim must be > 0")
    require(hidden_dim > 0, f"{name} hidden_dim must be > 0")
    require(num_experts > 0, "num_experts must be > 0")
    require(1 <= top_k <= num_experts, "top_k must be in [1, num_experts]")
    require(aux_loss_coef >= 0, "aux_loss_coef must be >= 0")


def _rms_without_scale(x, eps=1e-6):
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


def _expert_capacity(num_tokens, num_experts, capacity_factor):
    return max(1, math.ceil(capacity_factor * num_tokens / num_experts))


def _combine_token_choice(x_flat, experts, weights, indices):
    out = torch.zeros_like(x_flat)
    num_experts = len(experts)
    for k in range(indices.size(1)):
        expert_idx = indices[:, k]
        w = weights[:, k].to(dtype=x_flat.dtype).unsqueeze(-1)
        for e in range(num_experts):
            mask = expert_idx == e
            if mask.any():
                out[mask] += w[mask] * experts[e](x_flat[mask])
    return out


def _load_balancing_loss(logits, indices, num_experts, aux_loss_coef):
    probs = F.softmax(logits, dim=-1)
    frac = F.one_hot(indices, num_experts).float().mean(dim=(0, 1))
    return aux_loss_coef * num_experts * (frac * probs.mean(0)).sum()


def _balanced_assignment(scores, capacity):
    """Assign every token to one expert with bounded expert capacity.

    This solves the rectangular balanced assignment exactly with a small
    min-cost-flow reference path. BASE routing depends on the capacity
    constraint being real; if the graph cannot route all tokens we fail at the
    boundary instead of substituting an approximate assignment.
    """
    num_tokens, num_experts = scores.shape
    require(capacity * num_experts >= num_tokens, "balanced assignment capacity cannot cover all tokens")
    require(torch.isfinite(scores).all(), "balanced assignment scores must be finite")
    device = scores.device
    values = scores.float().cpu()
    source = 0
    token_offset = 1
    expert_offset = token_offset + num_tokens
    sink = expert_offset + num_experts
    graph = [[] for _ in range(sink + 1)]

    def add_edge(src, dst, cap, cost):
        graph[src].append([dst, len(graph[dst]), cap, cost])
        graph[dst].append([src, len(graph[src]) - 1, 0, -cost])
        return graph[src][-1]

    token_expert_edges = []
    for token in range(num_tokens):
        add_edge(source, token_offset + token, 1, 0.0)
        edges = []
        for expert in range(num_experts):
            edge = add_edge(token_offset + token, expert_offset + expert, 1, -float(values[token, expert].item()))
            edges.append(edge)
        token_expert_edges.append(edges)
    for expert in range(num_experts):
        add_edge(expert_offset + expert, sink, capacity, 0.0)

    for _ in range(num_tokens):
        parent_node = [-1] * len(graph)
        parent_edge = [-1] * len(graph)
        dist = [float("inf")] * len(graph)
        in_queue = [False] * len(graph)
        queue = [source]
        dist[source] = 0.0
        in_queue[source] = True

        while queue:
            node = queue.pop(0)
            in_queue[node] = False
            for edge_idx, edge in enumerate(graph[node]):
                dst, _, cap, cost = edge
                if cap <= 0:
                    continue
                candidate = dist[node] + cost
                if candidate < dist[dst]:
                    dist[dst] = candidate
                    parent_node[dst] = node
                    parent_edge[dst] = edge_idx
                    if not in_queue[dst]:
                        queue.append(dst)
                        in_queue[dst] = True

        require(parent_node[sink] != -1, "balanced assignment failed to route every token")
        node = sink
        while node != source:
            prev = parent_node[node]
            edge = graph[prev][parent_edge[node]]
            edge[2] -= 1
            graph[node][edge[1]][2] += 1
            node = prev

    assignment = torch.full((num_tokens,), -1, dtype=torch.long)
    for token, edges in enumerate(token_expert_edges):
        for expert, edge in enumerate(edges):
            if edge[2] == 0:
                assignment[token] = expert
                break
    require((assignment != -1).all(), "balanced assignment failed to route every token")
    return assignment.to(device)
