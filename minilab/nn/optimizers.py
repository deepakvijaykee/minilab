"""Muon: momentum + Newton-Schulz orthogonalization for matrix params.
Lion: sign-based, less memory than Adam. Chen et al. 2023."""

import torch
from torch.optim import Optimizer


class Muon(Optimizer):

    def __init__(self, params, lr=0.02, momentum=0.95, ns_iters=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, ns_iters=ns_iters))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mu, ns_iters = group["lr"], group["momentum"], group["ns_iters"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)
                buf = state["buf"]
                buf.mul_(mu).add_(g)
                update = g + mu * buf
                if p.dim() >= 2:
                    update = _newton_schulz(update, ns_iters)
                p.add_(update, alpha=-lr)


class Lion(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, (b1, b2), wd = group["lr"], group["betas"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "m" not in state:
                    state["m"] = torch.zeros_like(g)
                m = state["m"]
                update = (b1 * m + (1 - b1) * g).sign_()
                if wd > 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1 - b2)


def _newton_schulz(M, iters=5):
    """Approximate M @ (M^T M)^{-1/2} — orthogonalizes M."""
    assert M.dim() == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M / (M.norm() + 1e-7)
    for _ in range(iters):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X
