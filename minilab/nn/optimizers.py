"""Muon-family optimizers for hidden matrix params and Lion."""

import torch
from torch.optim import Optimizer

from minilab.checks import require, require_finite_number, require_integer


DEFAULT_SOFT_MUON_POWER = 0.4
DEFAULT_SOFT_MUON_NS_ITERS = 12
DEFAULT_SOFT_MUON_NS_COEFFICIENTS = (2.0, -1.5, 0.5)
DEFAULT_SOFT_MUON_EPS = 1e-7
DEFAULT_SOFT_MUON_P04_COEFFICIENTS = (
    0.427359225629,
    0.16510668279,
    0.0950365524083,
    0.0794622422344,
    0.0546397807059,
    0.0442774112372,
    0.0318743547215,
    0.0251008327807,
    0.0184953624306,
    0.014245458414,
    0.0137481403409,
    0.0,
)
DEFAULT_SOFT_MUON_P04_TAIL_COEFFICIENT = 0.0306539563075


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
        soft_muon=False,
        soft_muon_power=DEFAULT_SOFT_MUON_POWER,
        soft_muon_mix=1.0,
        soft_muon_ns_iters=DEFAULT_SOFT_MUON_NS_ITERS,
        soft_muon_ns_coefficients=DEFAULT_SOFT_MUON_NS_COEFFICIENTS,
        soft_muon_coefficients=DEFAULT_SOFT_MUON_P04_COEFFICIENTS,
        soft_muon_tail_coefficient=DEFAULT_SOFT_MUON_P04_TAIL_COEFFICIENT,
        soft_muon_eps=DEFAULT_SOFT_MUON_EPS,
    ):
        _validate_muon_hparams(
            lr,
            momentum,
            ns_iters,
            weight_decay,
            betas,
            eps,
            soft_muon,
            soft_muon_power,
            soft_muon_mix,
            soft_muon_ns_iters,
            soft_muon_ns_coefficients,
            soft_muon_coefficients,
            soft_muon_tail_coefficient,
            soft_muon_eps,
        )
        defaults = dict(
            lr=lr,
            momentum=momentum,
            ns_iters=ns_iters,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            use_muon=True,
            soft_muon=soft_muon,
            soft_muon_power=soft_muon_power,
            soft_muon_mix=soft_muon_mix,
            soft_muon_ns_iters=soft_muon_ns_iters,
            soft_muon_ns_coefficients=tuple(float(v) for v in soft_muon_ns_coefficients),
            soft_muon_coefficients=tuple(float(v) for v in soft_muon_coefficients),
            soft_muon_tail_coefficient=float(soft_muon_tail_coefficient),
            soft_muon_eps=soft_muon_eps,
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

    def load_state_dict(self, state_dict):
        result = super().load_state_dict(state_dict)
        for group in self.param_groups:
            _upgrade_muon_group_defaults(group, self.defaults)
            _validate_muon_group(group)
        return result

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
            if group["soft_muon"]:
                require(p.dim() >= 2, "Soft-Muon requires matrix-shaped parameters")
                update = _soft_muon_update(
                    update,
                    power=group["soft_muon_power"],
                    mix=group["soft_muon_mix"],
                    ns_iters=group["soft_muon_ns_iters"],
                    ns_coefficients=group["soft_muon_ns_coefficients"],
                    soft_coefficients=group["soft_muon_coefficients"],
                    soft_tail_coefficient=group["soft_muon_tail_coefficient"],
                    eps=group["soft_muon_eps"],
                )
            elif p.dim() >= 2 and ns_iters > 0:
                update = _orthogonalized_update(update, ns_iters)
            if wd > 0:
                p.mul_(1 - lr * wd)
            if update.dtype != p.dtype:
                update = update.to(dtype=p.dtype)
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


def _validate_muon_hparams(
    lr,
    momentum,
    ns_iters,
    weight_decay,
    betas,
    eps,
    soft_muon=False,
    soft_muon_power=DEFAULT_SOFT_MUON_POWER,
    soft_muon_mix=1.0,
    soft_muon_ns_iters=DEFAULT_SOFT_MUON_NS_ITERS,
    soft_muon_ns_coefficients=DEFAULT_SOFT_MUON_NS_COEFFICIENTS,
    soft_muon_coefficients=DEFAULT_SOFT_MUON_P04_COEFFICIENTS,
    soft_muon_tail_coefficient=DEFAULT_SOFT_MUON_P04_TAIL_COEFFICIENT,
    soft_muon_eps=DEFAULT_SOFT_MUON_EPS,
):
    require_finite_number(momentum, "Muon momentum")
    require_integer(ns_iters, "Muon ns_iters")
    require_finite_number(eps, "Muon eps")
    require(0 <= momentum < 1, "Muon momentum must be in [0, 1)")
    require(ns_iters >= 0, "Muon ns_iters must be >= 0")
    require(eps > 0, "Muon eps must be > 0")
    _validate_soft_muon_hparams(
        soft_muon,
        soft_muon_power,
        soft_muon_mix,
        soft_muon_ns_iters,
        soft_muon_ns_coefficients,
        soft_muon_coefficients,
        soft_muon_tail_coefficient,
        soft_muon_eps,
    )
    _validate_lr_betas_weight_decay(lr, betas, weight_decay)


def _upgrade_muon_group_defaults(group, defaults):
    for key in (
        "soft_muon",
        "soft_muon_power",
        "soft_muon_mix",
        "soft_muon_ns_iters",
        "soft_muon_ns_coefficients",
        "soft_muon_coefficients",
        "soft_muon_tail_coefficient",
        "soft_muon_eps",
    ):
        if key not in group:
            group[key] = defaults[key]


def _validate_muon_group(group):
    _upgrade_muon_group_defaults(group, {
        "soft_muon": False,
        "soft_muon_power": DEFAULT_SOFT_MUON_POWER,
        "soft_muon_mix": 1.0,
        "soft_muon_ns_iters": DEFAULT_SOFT_MUON_NS_ITERS,
        "soft_muon_ns_coefficients": DEFAULT_SOFT_MUON_NS_COEFFICIENTS,
        "soft_muon_coefficients": DEFAULT_SOFT_MUON_P04_COEFFICIENTS,
        "soft_muon_tail_coefficient": DEFAULT_SOFT_MUON_P04_TAIL_COEFFICIENT,
        "soft_muon_eps": DEFAULT_SOFT_MUON_EPS,
    })
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
            group["soft_muon"],
            group["soft_muon_power"],
            group["soft_muon_mix"],
            group["soft_muon_ns_iters"],
            group["soft_muon_ns_coefficients"],
            group["soft_muon_coefficients"],
            group["soft_muon_tail_coefficient"],
            group["soft_muon_eps"],
        )
    else:
        _validate_adamw_hparams(group["lr"], group["betas"], group["weight_decay"], group["eps"])


def _validate_soft_muon_hparams(
    soft_muon,
    power,
    mix,
    ns_iters,
    ns_coefficients,
    soft_coefficients,
    tail_coefficient,
    eps,
):
    require(soft_muon is True or soft_muon is False, "Muon param group soft_muon must be a bool")
    require_integer(ns_iters, "Soft-Muon ns_iters")
    require_finite_number(power, "Soft-Muon power")
    require_finite_number(mix, "Soft-Muon mix")
    require_finite_number(tail_coefficient, "Soft-Muon tail coefficient")
    require_finite_number(eps, "Soft-Muon eps")
    require(power == DEFAULT_SOFT_MUON_POWER, "Soft-Muon currently supports the fixed p=0.4 coefficient profile")
    require(0.0 <= mix <= 1.0, "Soft-Muon mix must be in [0, 1]")
    require(ns_iters >= 1, "Soft-Muon ns_iters must be >= 1")
    require(len(ns_coefficients) == 3, "Soft-Muon ns_coefficients must contain exactly 3 values")
    require(len(soft_coefficients) == ns_iters, "Soft-Muon coefficients length must match ns_iters")
    require(tail_coefficient >= 0.0, "Soft-Muon tail coefficient must be >= 0")
    require(eps > 0, "Soft-Muon eps must be > 0")
    for idx, coefficient in enumerate(ns_coefficients):
        require_finite_number(coefficient, f"Soft-Muon ns_coefficients[{idx}]")
    for idx, coefficient in enumerate(soft_coefficients):
        require_finite_number(coefficient, f"Soft-Muon coefficients[{idx}]")
    require(all(coefficient >= 0.0 for coefficient in soft_coefficients), "Soft-Muon coefficients must be >= 0")
    total = sum(float(coefficient) for coefficient in soft_coefficients) + float(tail_coefficient)
    require(abs(total - 1.0) <= 1e-6, "Soft-Muon coefficients must form a convex stack")


def _validate_adamw_hparams(lr, betas, weight_decay, eps):
    require(eps > 0, "optimizer eps must be > 0")
    _validate_lr_betas_weight_decay(lr, betas, weight_decay)


def _validate_lr_betas_weight_decay(lr, betas, weight_decay):
    require_finite_number(lr, "optimizer lr")
    require_finite_number(weight_decay, "optimizer weight_decay")
    require(lr >= 0, "optimizer lr must be >= 0")
    require(weight_decay >= 0, "optimizer weight_decay must be >= 0")
    require(len(betas) == 2, "optimizer betas must contain two values")
    b1, b2 = betas
    require_finite_number(b1, "optimizer beta1")
    require_finite_number(b2, "optimizer beta2")
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


def _soft_muon_update(
    update,
    *,
    power,
    mix,
    ns_iters,
    ns_coefficients,
    soft_coefficients,
    soft_tail_coefficient,
    eps,
):
    require(power == DEFAULT_SOFT_MUON_POWER, "Soft-Muon currently supports the fixed p=0.4 coefficient profile")
    original_shape = update.shape
    if update.dim() > 2:
        update = update.view(update.size(0), -1)
    sign_update = _zeropower_via_newton_schulz_stack(
        update,
        ns_iters=ns_iters,
        ns_coefficients=ns_coefficients,
        eps=eps,
    )
    soft_update = _softpower_via_newton_schulz_stack(
        update,
        ns_iters=ns_iters,
        ns_coefficients=ns_coefficients,
        soft_coefficients=soft_coefficients,
        soft_tail_coefficient=soft_tail_coefficient,
        eps=eps,
    )

    target_norm = _gram_frobenius_norm_estimate(sign_update, eps=eps)
    soft_norm = _gram_frobenius_norm_estimate(soft_update, eps=eps)
    soft_update = soft_update * (target_norm / soft_norm).to(dtype=soft_update.dtype)
    mixed = torch.lerp(sign_update, soft_update, mix)
    mixed_norm = _gram_frobenius_norm_estimate(mixed, eps=eps)
    mixed = mixed * (target_norm / mixed_norm).to(dtype=mixed.dtype)
    mixed *= max(1.0, mixed.size(0) / mixed.size(1)) ** 0.5
    return mixed.view(original_shape)


def _gram_frobenius_norm_estimate(update, *, keepdim=False, eps=1e-10):
    update_float = update.float()
    if update_float.size(0) > update_float.size(1):
        gram = update_float.T @ update_float
    else:
        gram = update_float @ update_float.T
    return gram.norm(dim=(-2, -1), keepdim=keepdim).sqrt().clamp_min(eps)


def _newton_schulz_stack_input(update, *, eps):
    require(update.dim() == 2, "Soft-Muon Newton-Schulz stack expects a 2D matrix")
    transposed = update.size(0) > update.size(1)
    dtype = torch.bfloat16 if update.device.type == "cuda" else torch.float32
    ns_update = update.to(dtype=dtype)
    if transposed:
        ns_update = ns_update.T
    ns_update = ns_update / _gram_frobenius_norm_estimate(ns_update, keepdim=True, eps=eps).to(dtype=ns_update.dtype)
    return ns_update, transposed


def _newton_schulz_stack_step(update, ns_coefficients):
    a, b, c = ns_coefficients
    gram = update @ update.T
    basis = b * gram + c * (gram @ gram)
    return a * update + basis @ update


def _zeropower_via_newton_schulz_stack(update, *, ns_iters, ns_coefficients, eps):
    ns_update, transposed = _newton_schulz_stack_input(update, eps=eps)
    for _ in range(ns_iters):
        ns_update = _newton_schulz_stack_step(ns_update, ns_coefficients)
    if transposed:
        ns_update = ns_update.T
    return ns_update


def _softpower_via_newton_schulz_stack(
    update,
    *,
    ns_iters,
    ns_coefficients,
    soft_coefficients,
    soft_tail_coefficient,
    eps,
):
    ns_update, transposed = _newton_schulz_stack_input(update, eps=eps)
    basis = [ns_update]
    for _ in range(ns_iters):
        ns_update = _newton_schulz_stack_step(ns_update, ns_coefficients)
        basis.append(ns_update)

    out = soft_tail_coefficient * basis[-1]
    for coefficient, basis_term in zip(soft_coefficients, basis[:-1], strict=True):
        out = out + coefficient * basis_term

    if transposed:
        out = out.T
    return out


def _orthogonalized_update(update, ns_iters):
    original_shape = update.shape
    if update.dim() > 2:
        update = update.view(update.size(0), -1)
    update = _newton_schulz(update, ns_iters)
    update *= max(1.0, update.size(0) / update.size(1)) ** 0.5
    return update.view(original_shape)
