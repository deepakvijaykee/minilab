import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require


class MambaMixer(nn.Module):
    """Mamba selective state-space mixer, reference PyTorch scan."""

    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank=None,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        require(dim > 0, "Mamba dim must be > 0")
        require(d_state > 0, "Mamba d_state must be > 0")
        require(d_conv > 0, "Mamba d_conv must be > 0")
        require(expand > 0, "Mamba expand must be > 0")
        require(dt_min > 0 and dt_max > dt_min, "Mamba requires 0 < dt_min < dt_max")
        require(dt_init_floor > 0, "Mamba dt_init_floor must be > 0")
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * dim
        self.dt_rank = math.ceil(dim / 16) if dt_rank is None else dt_rank
        require(self.dt_rank > 0, "Mamba dt_rank must be > 0")

        self.in_proj = nn.Linear(dim, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

        A = torch.arange(1, d_state + 1).float().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(A.log())
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self._init_delta_bias(dt_min, dt_max, dt_init_floor)

    @torch.no_grad()
    def _init_delta_bias(self, dt_min, dt_max, dt_init_floor):
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_proj.bias.copy_(inv_dt)

    def forward(self, x):
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        u = self.conv1d(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        u = F.silu(u)

        x_dbl = self.x_proj(u)
        dt, B_t, C_t = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        y = selective_scan_ref(u, dt, -torch.exp(self.A_log.float()), B_t, C_t, self.D)
        y = y * F.silu(z)
        return self.out_proj(y)


def selective_scan_ref(u, delta, A, B, C, D):
    """Reference selective scan used by Mamba.

    u, delta: (batch, seq, d_inner)
    A: (d_inner, d_state)
    B, C: (batch, seq, d_state)
    D: (d_inner,)
    """
    require(u.shape == delta.shape, "selective scan u and delta must have the same shape")
    require(A.dim() == 2, "selective scan A must be (d_inner, d_state)")
    require(B.shape[:2] == u.shape[:2], "selective scan B must match batch and sequence")
    require(C.shape == B.shape, "selective scan B and C must have the same shape")
    Bsz, T, D_inner = u.shape
    d_state = A.size(1)
    require(A.size(0) == D_inner, "selective scan A d_inner mismatch")
    require(B.size(2) == d_state, "selective scan B d_state mismatch")
    state = torch.zeros(Bsz, D_inner, d_state, device=u.device, dtype=u.dtype)
    A = A.to(device=u.device, dtype=u.dtype)
    outputs = []
    for t in range(T):
        delta_t = delta[:, t]
        u_t = u[:, t]
        state = (
            torch.exp(delta_t.unsqueeze(-1) * A.unsqueeze(0)) * state
            + delta_t.unsqueeze(-1) * B[:, t].unsqueeze(1) * u_t.unsqueeze(-1)
        )
        y_t = (state * C[:, t].unsqueeze(1)).sum(dim=-1) + D.to(u.dtype).unsqueeze(0) * u_t
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)
