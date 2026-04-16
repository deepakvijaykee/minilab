import json
import math
from pathlib import Path

import torch

from minilab.registry import register_scheduler, get_scheduler


@register_scheduler("cosine")
def cosine_schedule(T):
    t = torch.linspace(0, 1, T + 1)
    alpha = torch.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return alpha / alpha[0]


@register_scheduler("linear")
def linear_schedule(T):
    return 1 - torch.linspace(0, 1, T + 1)


@register_scheduler("log_linear")
def log_linear_schedule(T):
    return torch.exp(-torch.linspace(0, 1, T + 1) * math.log(1000.0))


class ForwardProcess:
    """Absorbing noise: each token kept with prob alpha(t), replaced with mask_token_id otherwise."""

    def __init__(self, mask_token_id, num_timesteps=1000, schedule="cosine"):
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.alpha = get_scheduler(schedule)(num_timesteps)

    def save(self, path):
        Path(path).write_text(json.dumps({
            "mask_token_id": self.mask_token_id,
            "num_timesteps": self.num_timesteps,
            "schedule": self.schedule,
        }))

    @classmethod
    def load(cls, path):
        s = json.loads(Path(path).read_text())
        return cls(s["mask_token_id"], num_timesteps=s["num_timesteps"], schedule=s["schedule"])

    def q_sample(self, x_0, t):
        B, T = x_0.shape
        idx = (t * self.num_timesteps).long().clamp(max=self.num_timesteps)
        alpha_t = self.alpha.to(x_0.device)[idx].view(B, 1)
        keep = torch.rand(B, T, device=x_0.device) < alpha_t
        z_t = torch.where(keep, x_0, self.mask_token_id)
        return z_t, ~keep

    def get_alpha(self, t):
        idx = (t * self.num_timesteps).long().clamp(max=self.num_timesteps)
        return self.alpha.to(t.device)[idx]

    def get_weight(self, t):
        """Returns -α'(t) / (1 - α(t)) — the MDLM continuous-time NELBO weight.
        α' is estimated by central difference on the cached schedule so this works
        for any registered schedule without needing a closed-form derivative."""
        alpha = self.alpha.to(t.device)
        idx = (t * self.num_timesteps).long().clamp(min=1, max=self.num_timesteps - 1)
        alpha_t = alpha[idx]
        dalpha_dt = (alpha[idx + 1] - alpha[idx - 1]) * (self.num_timesteps / 2.0)
        return -dalpha_dt / (1 - alpha_t).clamp(min=1e-5)
