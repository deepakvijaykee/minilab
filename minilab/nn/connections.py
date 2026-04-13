import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.registry import register_connection


@register_connection("residual")
class Residual(nn.Module):
    expansion = 1

    def __init__(self, dim):
        super().__init__()

    def forward(self, x, sublayer):
        return x + sublayer(x)


@register_connection("hc")
class HyperConnection(nn.Module):
    """Expanded residual stream with learnable mixing.
    State is (B, T, n, C) where n = expansion rate.
    Pre-weight: combine n copies into 1 for sublayer.
    Post-weight: distribute sublayer output back to n copies.
    Res-matrix: mix n copies among themselves."""

    def __init__(self, dim, expansion=4):
        super().__init__()
        self.expansion = expansion
        self.pre_weight = nn.Parameter(torch.zeros(expansion))
        self.post_weight = nn.Parameter(torch.zeros(expansion))
        self.res_weight = nn.Parameter(torch.eye(expansion))

    def forward(self, x, sublayer):
        pre_w = F.softmax(self.pre_weight, dim=0)
        h = (x * pre_w[None, None, :, None]).sum(dim=2)
        out = sublayer(h)
        post_w = F.softmax(self.post_weight, dim=0)
        post_out = out.unsqueeze(2) * post_w[None, None, :, None]
        res = torch.einsum("btnd,mn->btmd", x, self.res_weight)
        return res + post_out


@register_connection("mhc")
class ManifoldHyperConnection(nn.Module):
    """Like HC but constrains the residual mixing matrix to be doubly stochastic
    via Sinkhorn-Knopp. Preserves identity mapping, prevents gradient explosion.
    DeepSeek, 2026 (arxiv 2512.24880)."""

    def __init__(self, dim, expansion=4, sinkhorn_iters=20):
        super().__init__()
        self.expansion = expansion
        self.sinkhorn_iters = sinkhorn_iters
        self.pre_weight = nn.Parameter(torch.zeros(expansion))
        self.post_weight = nn.Parameter(torch.zeros(expansion))
        self.res_logits = nn.Parameter(torch.zeros(expansion, expansion))

    def forward(self, x, sublayer):
        pre_w = F.softmax(self.pre_weight, dim=0)
        h = (x * pre_w[None, None, :, None]).sum(dim=2)
        out = sublayer(h)
        post_w = F.softmax(self.post_weight, dim=0)
        post_out = out.unsqueeze(2) * post_w[None, None, :, None]

        # Sinkhorn-Knopp: iterative row/col normalization -> doubly stochastic
        M = self.res_logits.exp()
        for _ in range(self.sinkhorn_iters):
            M = M / M.sum(dim=-1, keepdim=True)
            M = M / M.sum(dim=-2, keepdim=True)
        res = torch.einsum("btnd,mn->btmd", x, M)
        return res + post_out
