import json
import math
from pathlib import Path

import torch

from minilab.checks import require
from minilab.registry import register_scheduler, get_scheduler


_SCHEDULE_ALPHA_FUNCTIONS = {}
_SCHEDULE_ALPHA_DERIVATIVES = {}


def register_continuous_alpha(name):
    def decorator(fn):
        require(name not in _SCHEDULE_ALPHA_FUNCTIONS, f"continuous alpha for '{name}' is already registered")
        _SCHEDULE_ALPHA_FUNCTIONS[name] = fn
        return fn
    return decorator


def register_alpha_derivative(name):
    def decorator(fn):
        require(
            name not in _SCHEDULE_ALPHA_DERIVATIVES,
            f"alpha derivative for '{name}' is already registered",
        )
        _SCHEDULE_ALPHA_DERIVATIVES[name] = fn
        return fn
    return decorator


@register_continuous_alpha("cosine")
def cosine_alpha(t):
    offset = 0.008
    scale = 1.008
    u = (t + offset) / scale * math.pi / 2
    u0 = offset / scale * math.pi / 2
    alpha = torch.cos(u) ** 2 / (math.cos(u0) ** 2)
    alpha = torch.where(t == 0, torch.ones_like(alpha), alpha)
    return torch.where(t == 1, torch.zeros_like(alpha), alpha)


@register_scheduler("cosine")
def cosine_schedule(T):
    t = torch.linspace(0, 1, T + 1)
    return cosine_alpha(t)


@register_alpha_derivative("cosine")
def cosine_alpha_derivative(t):
    offset = 0.008
    scale = 1.008
    u = (t + offset) / scale * math.pi / 2
    u0 = offset / scale * math.pi / 2
    return -torch.sin(2 * u) * (math.pi / (2 * scale)) / (math.cos(u0) ** 2)


@register_continuous_alpha("linear")
def linear_alpha(t):
    return 1 - t


@register_scheduler("linear")
def linear_schedule(T):
    return linear_alpha(torch.linspace(0, 1, T + 1))


@register_alpha_derivative("linear")
def linear_alpha_derivative(t):
    return torch.full_like(t, -1.0)


@register_continuous_alpha("geometric")
def geometric_alpha(t):
    return torch.exp(-t * math.log(1000.0))


@register_scheduler("geometric")
def geometric_schedule(T):
    return geometric_alpha(torch.linspace(0, 1, T + 1))


@register_alpha_derivative("geometric")
def geometric_alpha_derivative(t):
    return -math.log(1000.0) * geometric_alpha(t)


@register_continuous_alpha("log_linear")
def log_linear_alpha(t):
    return 1 - (1 - 1e-3) * t


@register_scheduler("log_linear")
def log_linear_schedule(T):
    return log_linear_alpha(torch.linspace(0, 1, T + 1))


@register_alpha_derivative("log_linear")
def log_linear_alpha_derivative(t):
    return torch.full_like(t, -(1 - 1e-3))


class _ScheduledForwardProcessMixin:
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_state()))

    def sample_time(self, batch_size, device, mode="continuous"):
        require(batch_size > 0, f"batch_size must be > 0, got {batch_size}")
        if mode == "continuous":
            return torch.rand(batch_size, device=device) * 0.999 + 0.001
        if mode == "discrete":
            idx = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=device)
            return idx.float() / self.num_timesteps
        raise ValueError(f"Unknown diffusion time sampling mode: {mode!r}")

    def get_alpha(self, t):
        idx = self.time_index(t)
        return self.alpha.to(t.device)[idx]

    def alpha_at(self, t):
        if self.schedule in _SCHEDULE_ALPHA_FUNCTIONS:
            return self.get_alpha_continuous(t)
        return self.get_alpha(t)

    def get_alpha_continuous(self, t):
        require(self.schedule in _SCHEDULE_ALPHA_FUNCTIONS, (
            f"schedule '{self.schedule}' must register a continuous alpha function "
            "before it can be used by continuous-time diffusion objectives"
        ))
        alpha = _SCHEDULE_ALPHA_FUNCTIONS[self.schedule](t)
        require(alpha.shape == t.shape, (
            f"continuous alpha for schedule '{self.schedule}' must return shape {tuple(t.shape)}, "
            f"got {tuple(alpha.shape)}"
        ))
        require(torch.isfinite(alpha).all(), f"continuous alpha for schedule '{self.schedule}' returned non-finite values")
        require(((0 <= alpha) & (alpha <= 1)).all(), (
            f"continuous alpha for schedule '{self.schedule}' must be in [0, 1]"
        ))
        return alpha

    def time_index(self, t, min_index=0, max_index=None):
        """Map normalized times to schedule indices using nearest-grid lookup."""
        if max_index is None:
            max_index = self.num_timesteps
        require(0 <= min_index <= max_index <= self.num_timesteps, (
            f"time index clamp must satisfy 0 <= min_index <= max_index <= {self.num_timesteps}, "
            f"got min_index={min_index}, max_index={max_index}"
        ))
        return (t * self.num_timesteps).round().long().clamp(min=min_index, max=max_index)


class ForwardProcess(_ScheduledForwardProcessMixin):
    """Absorbing noise: each token kept with prob alpha(t), replaced with mask_token_id otherwise."""

    process_type = "absorbing"

    def __init__(self, mask_token_id, num_timesteps=1000, schedule="cosine"):
        require(mask_token_id >= 0, "mask_token_id must be >= 0")
        require(num_timesteps > 1, "num_timesteps must be > 1")
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.alpha = get_scheduler(schedule)(num_timesteps)
        self._validate_alpha_schedule()

    def to_state(self):
        return {
            "process_type": self.process_type,
            "mask_token_id": self.mask_token_id,
            "num_timesteps": self.num_timesteps,
            "schedule": self.schedule,
        }

    @classmethod
    def load(cls, path):
        s = _load_process_state(path, cls.process_type, {"mask_token_id"})
        require(type(s["mask_token_id"]) is int and s["mask_token_id"] >= 0, (
            "Forward process mask_token_id must be a non-negative integer"
        ))
        return cls(s["mask_token_id"], num_timesteps=s["num_timesteps"], schedule=s["schedule"])

    def q_sample(self, x_0, t):
        B, T = x_0.shape
        if self.schedule in _SCHEDULE_ALPHA_FUNCTIONS:
            alpha_t = self.get_alpha_continuous(t).view(B, 1)
        else:
            idx = self.time_index(t)
            alpha_t = self.alpha.to(x_0.device)[idx].view(B, 1)
        keep = torch.rand(B, T, device=x_0.device) < alpha_t
        z_t = torch.where(keep, x_0, self.mask_token_id)
        return z_t, ~keep

    def has_terminal_mask_prior(self):
        zero = torch.zeros((), dtype=self.alpha.dtype, device=self.alpha.device)
        return bool(torch.isclose(self.alpha[-1], zero).item())

    def get_sigma(self, t):
        """Continuous-time CTMC noise level for absorbing diffusion, sigma(t) = -log alpha(t)."""
        return -self.get_alpha_continuous(t).clamp(min=1e-12).log()

    def get_alpha_derivative(self, t):
        require(self.schedule in _SCHEDULE_ALPHA_DERIVATIVES, (
            f"schedule '{self.schedule}' must register an exact alpha derivative "
            "before it can be used by continuous-time diffusion objectives"
        ))
        derivative = _SCHEDULE_ALPHA_DERIVATIVES[self.schedule](t)
        require(derivative.shape == t.shape, (
            f"alpha derivative for schedule '{self.schedule}' must return shape {tuple(t.shape)}, "
            f"got {tuple(derivative.shape)}"
        ))
        require(torch.isfinite(derivative).all(), (
            f"alpha derivative for schedule '{self.schedule}' returned non-finite values"
        ))
        return derivative

    def get_sigma_derivative(self, t):
        """Exact d sigma / dt used by SEDD's score-entropy integral."""
        alpha_t = self.get_alpha_continuous(t).clamp(min=1e-12)
        return -self.get_alpha_derivative(t) / alpha_t

    def get_weight(self, t):
        """Returns -α'(t) / (1 - α(t)) — the MDLM continuous-time NELBO weight.
        Schedules own α' so continuous objectives do not depend on the discretized
        training grid used for sampling and checkpoint compatibility."""
        alpha_t = self.get_alpha_continuous(t)
        return -self.get_alpha_derivative(t) / (1 - alpha_t).clamp(min=1e-5)

    def _validate_alpha_schedule(self):
        _validate_alpha_schedule(self.alpha, self.num_timesteps, self.schedule)


class UniformForwardProcess(_ScheduledForwardProcessMixin):
    """Categorical corruption: alpha(t) keeps x_0, otherwise sample from the uniform vocabulary prior."""

    process_type = "uniform"

    def __init__(self, vocab_size, num_timesteps=1000, schedule="cosine"):
        require(vocab_size > 1, "vocab_size must be > 1")
        require(num_timesteps > 1, "num_timesteps must be > 1")
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.alpha = get_scheduler(schedule)(num_timesteps)
        _validate_alpha_schedule(self.alpha, num_timesteps, schedule)

    def to_state(self):
        return {
            "process_type": self.process_type,
            "vocab_size": self.vocab_size,
            "num_timesteps": self.num_timesteps,
            "schedule": self.schedule,
        }

    @classmethod
    def load(cls, path):
        s = _load_process_state(path, cls.process_type, {"vocab_size"})
        require(type(s["vocab_size"]) is int and s["vocab_size"] > 1, (
            "Uniform forward process vocab_size must be an integer > 1"
        ))
        return cls(s["vocab_size"], num_timesteps=s["num_timesteps"], schedule=s["schedule"])

    def q_sample(self, x_0, t):
        require(((0 <= x_0) & (x_0 < self.vocab_size)).all(), "uniform diffusion tokens are outside the vocabulary")
        B, T = x_0.shape
        alpha_t = self.alpha_at(t).to(x_0.device).view(B, 1)
        keep = torch.rand(B, T, device=x_0.device) < alpha_t
        noise = torch.randint(self.vocab_size, x_0.shape, device=x_0.device)
        return torch.where(keep, x_0, noise), ~keep

    def q_probs(self, x_0, t):
        require(((0 <= x_0) & (x_0 < self.vocab_size)).all(), "uniform diffusion tokens are outside the vocabulary")
        B, T = x_0.shape
        alpha_t = self.alpha_at(t).to(x_0.device).view(B, 1, 1)
        probs = torch.full((*x_0.shape, self.vocab_size), 1.0, device=x_0.device, dtype=alpha_t.dtype)
        probs = probs * ((1.0 - alpha_t) / self.vocab_size)
        probs.scatter_add_(-1, x_0.unsqueeze(-1), alpha_t.expand(B, T, 1))
        return probs

    def marginal_transition_matrix(self, t):
        """Return the cumulative uniform corruption kernel q(z_t | x_0).

        This is the D3PM-style forward marginal for the uniform corruption
        family, not a learned reverse process or one-step posterior.
        """
        alpha_t = self.alpha_at(t).view(-1, 1, 1)
        eye = torch.eye(self.vocab_size, device=t.device, dtype=alpha_t.dtype).unsqueeze(0)
        uniform = torch.full_like(eye, 1.0 / self.vocab_size)
        return alpha_t * eye + (1.0 - alpha_t) * uniform

    def transition_matrix(self, t):
        return self.marginal_transition_matrix(t)

    def has_terminal_mask_prior(self):
        return False


def load_forward_process(path):
    s = json.loads(Path(path).read_text())
    require(isinstance(s, dict), "Forward process state must be a JSON object")
    require("process_type" in s, "Forward process state is missing required field: process_type")
    process_type = s["process_type"]
    if process_type == ForwardProcess.process_type:
        return ForwardProcess.load(path)
    if process_type == UniformForwardProcess.process_type:
        return UniformForwardProcess.load(path)
    raise ValueError(f"Unsupported forward process type: {process_type!r}")


def forward_process_signature(forward_process):
    return json.dumps(forward_process.to_state(), sort_keys=True)


def _validate_alpha_schedule(alpha, num_timesteps, schedule):
    require(alpha.dim() == 1 and alpha.numel() == num_timesteps + 1, (
        f"schedule '{schedule}' must return {num_timesteps + 1} alpha values"
    ))
    require(torch.isfinite(alpha).all(), f"schedule '{schedule}' returned non-finite alpha values")
    require(((0 <= alpha) & (alpha <= 1)).all(), (
        f"schedule '{schedule}' alpha values must be probabilities in [0, 1]"
    ))
    one = torch.ones((), dtype=alpha.dtype, device=alpha.device)
    require(torch.isclose(alpha[0], one), f"schedule '{schedule}' must start at alpha[0] = 1")
    require(alpha[-1] < alpha[0], f"schedule '{schedule}' must add noise over time")
    require((alpha[1:] <= alpha[:-1]).all(), (
        f"schedule '{schedule}' alpha values must be monotonically non-increasing"
    ))


def _load_process_state(path, process_type, extra_fields):
    s = json.loads(Path(path).read_text())
    require(isinstance(s, dict), "Forward process state must be a JSON object")
    required = {"process_type", "num_timesteps", "schedule"} | set(extra_fields)
    missing = required - set(s)
    unknown = set(s) - required
    require(not missing, f"Missing forward process fields: {sorted(missing)}")
    require(not unknown, f"Unknown forward process fields: {sorted(unknown)}")
    require(s["process_type"] == process_type, f"Unsupported forward process type: {s['process_type']}")
    require(type(s["num_timesteps"]) is int and s["num_timesteps"] > 1, (
        "Forward process num_timesteps must be an integer > 1"
    ))
    require(type(s["schedule"]) is str and s["schedule"], (
        "Forward process schedule must be a non-empty string"
    ))
    return s
