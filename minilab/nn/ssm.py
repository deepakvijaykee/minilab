import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minilab.checks import require
from minilab.nn.norm import RMSNorm


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


class Mamba2Mixer(nn.Module):
    """Mamba-2 SSD mixer, reference PyTorch scan.

    This mirrors the unfused Mamba-2 block structure: a single input projection
    emits gate, SSM input, B/C state projections, and per-head time steps; a
    causal depthwise convolution mixes the SSM input and B/C streams before the
    head-wise structured state-space recurrence.
    """

    def __init__(
        self,
        dim,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=64,
        ngroups=1,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_init_min=1.0,
        A_init_max=16.0,
    ):
        super().__init__()
        require(dim > 0, "Mamba2 dim must be > 0")
        require(d_state > 0, "Mamba2 d_state must be > 0")
        require(d_conv > 0, "Mamba2 d_conv must be > 0")
        require(expand > 0, "Mamba2 expand must be > 0")
        require(headdim > 0, "Mamba2 headdim must be > 0")
        require(ngroups > 0, "Mamba2 ngroups must be > 0")
        require(dt_min > 0 and dt_max > dt_min, "Mamba2 requires 0 < dt_min < dt_max")
        require(dt_init_floor > 0, "Mamba2 dt_init_floor must be > 0")
        require(0 < A_init_min <= A_init_max, "Mamba2 A init range must satisfy 0 < min <= max")
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * dim
        self.headdim = headdim
        require(self.d_inner % headdim == 0, "Mamba2 expanded dim must be divisible by headdim")
        self.nheads = self.d_inner // headdim
        require(self.nheads % ngroups == 0, "Mamba2 nheads must be divisible by ngroups")
        self.ngroups = ngroups

        self.in_proj = nn.Linear(dim, 2 * self.d_inner + 2 * ngroups * d_state + self.nheads, bias=False)
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=True,
        )
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)
        self.norm = RMSNorm(self.d_inner, eps=1e-5)

        dt = torch.exp(
            torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        A = torch.empty(self.nheads).uniform_(A_init_min, A_init_max)
        self.A_log = nn.Parameter(A.log())
        self.D = nn.Parameter(torch.ones(self.nheads))
        heads_per_group = self.nheads // ngroups
        self.register_buffer(
            "head_to_group",
            torch.arange(self.nheads, dtype=torch.long) // heads_per_group,
            persistent=False,
        )

    def forward(self, x):
        B, T, _ = x.shape
        projected = self.in_proj(x)
        z, x_ssm, B_t, C_t, dt = torch.split(
            projected,
            [self.d_inner, self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state, self.nheads],
            dim=-1,
        )

        xbc = torch.cat([x_ssm, B_t, C_t], dim=-1)
        xbc = self.conv1d(xbc.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xbc = F.silu(xbc)
        x_ssm, B_t, C_t = torch.split(
            xbc,
            [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state],
            dim=-1,
        )

        x_ssm = x_ssm.view(B, T, self.nheads, self.headdim)
        B_t = B_t.view(B, T, self.ngroups, self.d_state)[:, :, self.head_to_group]
        C_t = C_t.view(B, T, self.ngroups, self.d_state)[:, :, self.head_to_group]
        dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))
        y = mamba2_ssd_scan_ref(
            x_ssm,
            dt,
            -torch.exp(self.A_log.float()),
            B_t,
            C_t,
            self.D,
        )
        y = y.reshape(B, T, self.d_inner)
        y = self.norm(y * F.silu(z))
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


def mamba2_ssd_scan_ref(x, delta, A, B, C, D):
    """Reference Mamba-2/SSD scan.

    x: (batch, seq, nheads, headdim)
    delta: (batch, seq, nheads)
    A, D: (nheads,)
    B, C: (batch, seq, nheads, d_state)
    """
    require(x.dim() == 4, "Mamba2 scan x must be (batch, seq, nheads, headdim)")
    require(delta.shape == x.shape[:3], "Mamba2 delta must match x batch/seq/head")
    require(B.shape[:3] == x.shape[:3], "Mamba2 B must match x batch/seq/head")
    require(C.shape == B.shape, "Mamba2 B and C must have the same shape")
    require(A.shape == (x.size(2),), "Mamba2 A must have shape (nheads,)")
    require(D.shape == (x.size(2),), "Mamba2 D must have shape (nheads,)")
    Bsz, T, nheads, headdim = x.shape
    d_state = B.size(-1)
    state = torch.zeros(Bsz, nheads, headdim, d_state, device=x.device, dtype=x.dtype)
    A = A.to(device=x.device, dtype=torch.float32)
    D = D.to(device=x.device, dtype=x.dtype)
    outputs = []
    for t in range(T):
        delta_t = delta[:, t].float()
        decay = torch.exp(delta_t * A.view(1, nheads)).to(dtype=x.dtype)
        x_t = x[:, t]
        B_t = B[:, t]
        C_t = C[:, t]
        state = (
            state * decay.view(Bsz, nheads, 1, 1)
            + delta[:, t].to(x.dtype).view(Bsz, nheads, 1, 1)
            * x_t.unsqueeze(-1)
            * B_t.unsqueeze(2)
        )
        y_t = (state * C_t.unsqueeze(2)).sum(dim=-1) + D.view(1, nheads, 1) * x_t
        outputs.append(y_t)
    return torch.stack(outputs, dim=1)
