"""Muon: momentum + Newton-Schulz orthogonalization for hidden matrix params.
Lion: sign-based, less memory than Adam. Chen et al. 2023."""

import torch
from torch.optim import Optimizer

from minilab.checks import require


class Muon(Optimizer):

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        ns_iters=5,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        eps=1e-8,
    ):
        _validate_muon_hparams(lr, momentum, ns_iters, weight_decay, betas, eps)
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_iters=ns_iters,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            use_muon=True,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            _validate_muon_group(group)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._step_muon_group(group)
            else:
                self._step_adamw_group(group)
        return loss

    def _step_muon_group(self, group):
        lr, mu, ns_iters, wd = group["lr"], group["momentum"], group["ns_iters"], group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.lerp_(g, 1 - mu)
            update = g.lerp(buf, mu)
            if p.dim() >= 2 and ns_iters > 0:
                update = _orthogonalized_update(update, ns_iters)
            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(update, alpha=-lr)

    def _step_adamw_group(self, group):
        lr, (b1, b2), wd, eps = group["lr"], group["betas"], group["weight_decay"], group["eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
            state["step"] += 1
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if wd > 0:
                p.mul_(1 - lr * wd)
            exp_avg.mul_(b1).add_(g, alpha=1 - b1)
            exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
            bias_correction1 = 1 - b1 ** state["step"]
            bias_correction2 = 1 - b2 ** state["step"]
            denom = exp_avg_sq.sqrt().div_(bias_correction2 ** 0.5).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


class Lion(Optimizer):

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        _validate_lr_betas_weight_decay(lr, betas, weight_decay)
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        for group in self.param_groups:
            _validate_lr_betas_weight_decay(group["lr"], group["betas"], group["weight_decay"])

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
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
        return loss


def _validate_muon_hparams(lr, momentum, ns_iters, weight_decay, betas, eps):
    require(0 <= momentum < 1, "Muon momentum must be in [0, 1)")
    require(ns_iters >= 0, "Muon ns_iters must be >= 0")
    require(eps > 0, "Muon eps must be > 0")
    _validate_lr_betas_weight_decay(lr, betas, weight_decay)


def _validate_muon_group(group):
    require(
        group["use_muon"] is True or group["use_muon"] is False,
        "Muon param group use_muon must be a bool",
    )
    if group["use_muon"]:
        _validate_muon_hparams(
            group["lr"],
            group["momentum"],
            group["ns_iters"],
            group["weight_decay"],
            group["betas"],
            group["eps"],
        )
    else:
        _validate_adamw_hparams(group["lr"], group["betas"], group["weight_decay"], group["eps"])


def _validate_adamw_hparams(lr, betas, weight_decay, eps):
    require(eps > 0, "optimizer eps must be > 0")
    _validate_lr_betas_weight_decay(lr, betas, weight_decay)


def _validate_lr_betas_weight_decay(lr, betas, weight_decay):
    require(lr >= 0, "optimizer lr must be >= 0")
    require(weight_decay >= 0, "optimizer weight_decay must be >= 0")
    require(len(betas) == 2, "optimizer betas must contain two values")
    b1, b2 = betas
    require(0 <= b1 < 1 and 0 <= b2 < 1, "optimizer betas must be in [0, 1)")


def _newton_schulz(M, iters=5):
    """Approximate the zeroth-power / semi-orthogonal form of a 2D matrix."""
    require(M.dim() == 2, "Newton-Schulz orthogonalization expects a 2D matrix")
    a, b, c = (3.4445, -4.7750, 2.0315)
    should_transpose = M.size(0) > M.size(1)
    X = M.T if should_transpose else M
    X = X / (X.norm() + 1e-7)
    for _ in range(iters):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if should_transpose else X


def _orthogonalized_update(update, ns_iters):
    original_shape = update.shape
    if update.dim() > 2:
        update = update.view(update.size(0), -1)
    update = _newton_schulz(update, ns_iters)
    update *= max(1.0, update.size(0) / update.size(1)) ** 0.5
    return update.view(original_shape)
