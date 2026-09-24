"""Phase 3 Step 2: FDNA inference repairs, one variable at a time (registry/phase3_step2.yaml).
head: 'meanfield' (existing: independent unit Bernoullis) | 'joint' (FDNA on each binary state, posterior over the 16384 states) | 'logic' (ServiceGraph truth, no FDNA).
or_aware: gateway parents of a generator are combined by OR (noisy-OR of operabilities; max when tau=0) before the unit node.
prior: 'indep' (independent component failures) | 'cc' (common-cause mixture over v2.GROUPS, as in BayesStructure)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import comm, v2
from .bayes import BayesStructure
from .belief import N_C, N_F, N_U, flag_ancestor_mask, unit_input_mask
from .inference import MeanFieldControls
from .layers import FDNALayer


def or_input_mask() -> np.ndarray:
    """(10 units, 14 comps + 5 gateway-group nodes): unit inputs = CC, its RTU, its generator's gateway group."""
    m = np.zeros((N_U, N_C + comm.N_REMOTE_GEN), np.float32)
    for u in range(N_U):
        m[u, [comm.CC, comm.rtu(u), N_C + u // comm.UNITS_PER_GEN]] = 1
    return m


class ReprInf(nn.Module):
    def __init__(self, world: v2.World, level_idx: np.ndarray, head: str = "meanfield", or_aware: bool = False, tau: float = 0.05, prior: str = "indep", or_combine: str = "max"):
        super().__init__()
        assert head in ("meanfield", "joint", "logic") and prior in ("indep", "cc") and or_combine in ("max", "noisy_or")
        self.head, self.or_aware, self.tau, self.prior_kind, self.or_combine = head, or_aware, tau, prior, or_combine
        self.register_buffer("mf", torch.tensor(flag_ancestor_mask()))
        self.prior = nn.Parameter(torch.full((N_C,), 1.5))
        self.w1 = nn.Parameter(torch.randn(N_F, N_C) * 0.5 + 1.0)
        self.w0 = nn.Parameter(torch.randn(N_F, N_C) * 0.5 - 2.0)
        if head != "logic":
            self.fdna = FDNALayer(torch.tensor(or_input_mask() if or_aware else unit_input_mask()), tau=tau)
            self.register_buffer("share", torch.tensor([comm.SHARES[u % comm.UNITS_PER_GEN] for u in range(N_U)], dtype=torch.float32))
        self.register_buffer("lv", torch.tensor(level_idx, dtype=torch.long))
        if head != "logic":
            self.mfc = MeanFieldControls(level_idx)
        if head != "meanfield":
            self.bs = BayesStructure(world)          # learnable prior parameters (fail / group); emission parts unused
            self.register_buffer("X", torch.tensor(world.X, dtype=torch.float32))
            self.register_buffer("cidx", torch.tensor(world.cidx, dtype=torch.long))
            self.n_cv = len(world.CV)

    def unit_operability(self, o_comp: torch.Tensor) -> torch.Tensor:
        """or_combine: 'max' (operability/logic OR, used for every or_aware arm so the tau ablation changes ONLY the
        softmin temperature downstream, per Phase 4 correction 4.3 -- previously this silently switched to noisy-OR
        for tau>0, conflating OR semantics with temperature) | 'noisy_or' (independence-interpretable probability OR;
        only meaningful when o_comp are genuinely probabilities, kept as an explicit opt-in, never the default)."""
        if not self.or_aware:
            return self.fdna(o_comp)
        gws = torch.stack([o_comp[:, comm.gw(g)] for g in range(comm.N_GW)], 1)
        grp = []
        for g in range(comm.N_REMOTE_GEN):
            par = gws[:, list(comm.PARENT_GW[g])]
            grp.append(1 - torch.prod(1 - par, 1) if self.or_combine == "noisy_or" else par.max(1).values)
        return self.fdna(torch.cat([o_comp, torch.stack(grp, 1)], 1))

    def evidence(self, e: torch.Tensor) -> torch.Tensor:
        return (e[:, :, 0:1] * (self.mf * self.w1)[None]).sum(1) + (e[:, :, 1:2] * (self.mf * self.w0)[None]).sum(1)

    def level_probs(self, o_unit: torch.Tensor) -> torch.Tensor:
        """(N, 1024) log p(control vector) from independent unit Bernoullis with success probabilities o_unit."""
        return self.mfc(o_unit)

    def log_state_prior(self) -> torch.Tensor:
        if self.prior_kind == "cc":
            return self.bs.log_prior()
        pf = torch.sigmoid(self.bs.fail)
        return (1 - self.X) @ torch.log(pf) + self.X @ torch.log1p(-pf)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        t = self.evidence(e)
        if self.head == "meanfield":
            o_unit = self.unit_operability(torch.sigmoid(self.prior[None] + t))
            return torch.log_softmax(self.level_probs(o_unit), dim=1)
        post = torch.softmax(self.log_state_prior()[None] + t @ self.X.T, dim=1)          # (B, 16384)
        if self.head == "logic":
            pc = torch.zeros(len(e), self.n_cv, device=e.device).index_add_(1, self.cidx, post)
        else:
            M = torch.exp(self.mfc(self.unit_operability(self.X)))                       # (16384, 1024) p(c | state) under FDNA
            pc = post @ M
        return torch.log(pc.clamp_min(1e-12))
