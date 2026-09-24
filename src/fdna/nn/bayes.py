"""Exact differentiable Bayesian layer over the KNOWN dependency structure (I8): posterior over the 16384 hidden states from a prior
with learnable per-component failure probabilities and learnable common-cause group probabilities, and an emission with a learnable
stale probability; aggregated to the 1024 control vectors. Only ~17 free parameters."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import v2

NEG = -30.0


class BayesStructure(nn.Module):
    def __init__(self, world: v2.World, init_fail: float = 0.10, init_group: float = 0.10, init_stale: float = 0.10, robust: bool = False):
        super().__init__()
        n = v2.N_COMP
        down = torch.tensor(~world.X, dtype=torch.float32)                    # (S, 14)
        self.register_buffer("down", down)
        self.register_buffer("T", torch.tensor(world.T, dtype=torch.float32))  # (S, 13) true flag values
        self.register_buffer("cidx", torch.tensor(world.cidx, dtype=torch.long))
        n_ev = 1 << len(v2.GROUPS)
        cov = np.zeros((n_ev, n), np.float32)
        ev = np.zeros((n_ev, len(v2.GROUPS)), np.float32)
        for e in range(n_ev):
            for g, grp in enumerate(v2.GROUPS):
                if e >> g & 1:
                    cov[e, list(grp)] = 1.0
                    ev[e, g] = 1.0
        self.register_buffer("cov", torch.tensor(cov)); self.register_buffer("ev", torch.tensor(ev))
        logit = lambda p: float(np.log(p / (1 - p)))
        self.fail = nn.Parameter(torch.full((n,), logit(init_fail)))
        self.group = nn.Parameter(torch.full((len(v2.GROUPS),), logit(init_group)))
        self.stale = nn.Parameter(torch.tensor(logit(init_stale)))
        self.robust = robust
        if robust:                                   # ONE extra learnable parameter: P(report 'down' | truth 'up')
            self.leak = nn.Parameter(torch.tensor(logit(0.05)))
        self.n_cv = len(world.CV)

    def log_prior(self) -> torch.Tensor:
        pf = torch.sigmoid(self.fail)
        lpf, l1pf = torch.log(pf), torch.log1p(-pf)
        pg = torch.sigmoid(self.group)
        lev = (self.ev * torch.log(pg) + (1 - self.ev) * torch.log1p(-pg)).sum(1)          # (E,)
        a = torch.where(self.cov > 0, torch.zeros_like(self.cov), lpf[None].expand_as(self.cov))     # log P(down | e)
        b = torch.where(self.cov > 0, torch.full_like(self.cov, NEG), l1pf[None].expand_as(self.cov))  # log P(up | e)
        ll = self.down @ a.T + (1 - self.down) @ b.T                                          # (S, E)
        return torch.logsumexp(ll + lev[None], dim=1)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        s = torch.sigmoid(self.stale).clamp(1e-6, 1 - 1e-6)
        A = torch.where(self.T > 0, torch.zeros_like(self.T), torch.log(s).expand_as(self.T))            # log e(1 | truth)
        up_down = torch.log(torch.sigmoid(self.leak).clamp(1e-6, 1 - 1e-6)) if self.robust else NEG
        B = torch.where(self.T > 0, torch.full_like(self.T, 1.0) * up_down, torch.log1p(-s).expand_as(self.T))   # log e(0 | truth)
        ll = e[:, :, 0] @ A.T + e[:, :, 1] @ B.T
        post = torch.softmax(self.log_prior()[None] + ll, dim=1)
        pc = torch.zeros(len(e), self.n_cv, device=e.device).index_add_(1, self.cidx, post)
        return torch.log(pc.clamp_min(1e-12))
