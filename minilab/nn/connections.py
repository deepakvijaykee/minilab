import torch
import torch.nn as nn

from minilab.checks import require
from minilab.registry import register_connection


@register_connection("residual")
class Residual(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super().__init__()
        require(dim > 0, "Residual dim must be > 0")

    def forward(self, x, sublayer):
        return x + sublayer(x)


@register_connection("hc")
class HyperConnection(nn.Module):
    """Dynamic Hyper-Connections for an expanded residual stream.

    The stream state is (B, T, n, C). Each layer reads one C-dimensional
    mixture for its sublayer, keeps n residual streams, and writes the sublayer
    output back into those streams with learnable depth weights.
    """

    def __init__(self, dim, expansion=4, layer_id=0, dynamic=True):
        super().__init__()
        require(dim > 0, "HyperConnection dim must be > 0")
        require(expansion > 0, "HyperConnection expansion must be > 0")
        require(layer_id >= 0, "HyperConnection layer_id must be >= 0")
        self.expansion = expansion
        self.dynamic = dynamic
        self.norm = nn.LayerNorm(dim)
        self.dynamic_alpha_scale = nn.Parameter(torch.tensor(1e-3))
        self.dynamic_beta_scale = nn.Parameter(torch.tensor(1e-3))
        self.dynamic_alpha = nn.Linear(dim, expansion + 1, bias=False)
        self.dynamic_beta = nn.Linear(dim, 1, bias=False)
        self.reset_dynamic_parameters()

        alpha = torch.zeros(expansion, expansion + 1)
        alpha[layer_id % expansion, 0] = 1.0
        alpha[:, 1:] = torch.eye(expansion)
        self.static_alpha = nn.Parameter(alpha)
        self.static_beta = nn.Parameter(torch.ones(expansion))

    def forward(self, x, sublayer):
        alpha, beta = self._connection_weights(x)
        mixed = torch.matmul(alpha.transpose(-1, -2), x)
        out = sublayer(mixed[:, :, 0])
        return mixed[:, :, 1:] + out.unsqueeze(2) * beta.unsqueeze(-1)

    def _connection_weights(self, x):
        alpha = self.static_alpha
        beta = self.static_beta
        if self.dynamic:
            h = self.norm(x)
            alpha = alpha + self.dynamic_alpha_scale * torch.tanh(self.dynamic_alpha(h))
            beta = beta + self.dynamic_beta_scale * torch.tanh(self.dynamic_beta(h)).squeeze(-1)
        return alpha, beta

    def reset_dynamic_parameters(self):
        nn.init.zeros_(self.dynamic_alpha.weight)
        nn.init.zeros_(self.dynamic_beta.weight)


@register_connection("mhc")
class ManifoldHyperConnection(nn.Module):
    """Manifold-constrained Hyper-Connections.

    The residual stream map is projected onto the Birkhoff polytope with
    Sinkhorn-Knopp. The read/write maps are constrained to be non-negative, so
    the multi-stream residual path preserves the signal-conservation invariant
    mHC relies on while remaining readable in native PyTorch.
    """

    def __init__(self, dim, expansion=4, layer_id=0, sinkhorn_iters=20, dynamic=True):
        super().__init__()
        require(dim > 0, "ManifoldHyperConnection dim must be > 0")
        require(expansion > 0, "ManifoldHyperConnection expansion must be > 0")
        require(layer_id >= 0, "ManifoldHyperConnection layer_id must be >= 0")
        require(sinkhorn_iters > 0, "sinkhorn_iters must be > 0")
        self.expansion = expansion
        self.sinkhorn_iters = sinkhorn_iters
        self.dynamic = dynamic
        self.norm = nn.LayerNorm(dim)
        self.dynamic_scale = nn.Parameter(torch.tensor(1e-3))
        self.dynamic_maps = nn.Linear(expansion * dim, expansion * (expansion + 2), bias=False)
        self.reset_dynamic_parameters()

        pre_logits = torch.full((expansion,), -12.0)
        pre_logits[layer_id % expansion] = 12.0
        post_logits = torch.full((expansion,), 12.0)
        res_logits = torch.full((expansion, expansion), -12.0)
        res_logits.fill_diagonal_(12.0)
        self.pre_logits = nn.Parameter(pre_logits)
        self.post_logits = nn.Parameter(post_logits)
        self.res_logits = nn.Parameter(res_logits)

    def forward(self, x, sublayer):
        pre, post, res = self._connection_weights(x)
        h = (x * pre.unsqueeze(-1)).sum(dim=2)
        out = sublayer(h)
        mixed = torch.matmul(res.transpose(-1, -2), x)
        return mixed + out.unsqueeze(2) * post.unsqueeze(-1)

    def _connection_weights(self, x):
        pre_logits = self.pre_logits
        post_logits = self.post_logits
        res_logits = self.res_logits
        if self.dynamic:
            h = self.norm(x).flatten(start_dim=2)
            raw = self.dynamic_scale * torch.tanh(self.dynamic_maps(h))
            pre_delta, post_delta, res_delta = torch.split(
                raw,
                [self.expansion, self.expansion, self.expansion * self.expansion],
                dim=-1,
            )
            pre_logits = pre_logits + pre_delta
            post_logits = post_logits + post_delta
            res_logits = res_logits + res_delta.view(*res_delta.shape[:-1], self.expansion, self.expansion)
        return torch.sigmoid(pre_logits), torch.sigmoid(post_logits), _sinkhorn(res_logits, self.sinkhorn_iters)

    def reset_dynamic_parameters(self):
        nn.init.zeros_(self.dynamic_maps.weight)


def _sinkhorn(logits, iters):
    M = (logits - logits.amax(dim=(-1, -2), keepdim=True)).exp()
    for _ in range(iters):
        M = M / M.sum(dim=-1, keepdim=True)
        M = M / M.sum(dim=-2, keepdim=True)
    return M


def expand_residual_stream(x, expansion):
    require(expansion > 0, "residual stream expansion must be > 0")
    return x.unsqueeze(2).expand(-1, -1, expansion, -1).contiguous()


def reduce_residual_stream(x):
    require(x.dim() == 4, "expanded residual stream must have shape (batch, seq, streams, dim)")
    return x.sum(dim=2)
