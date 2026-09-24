"""Differentiable service calculation with learnable dependency edges (noisy-OR over redundant parents, AND over critical
dependencies), initialised from an ASSUMED wiring that may be wrong."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import comm, wiring
from .layers import MonotoneValue


class ServiceGraph(nn.Module):
    def __init__(self, parents: dict, unit_gen: np.ndarray, learn_edges: bool = True, init: float = 4.0):
        super().__init__()
        a = np.full((wiring.N_UNIT, wiring.N_GW), -init, np.float32)
        w = np.full((wiring.N_UNIT, wiring.N_GEN), -init, np.float32)
        for u in range(wiring.N_UNIT):
            g = unit_gen[u]
            a[u, list(parents[g])] = init
            w[u, g] = init
        self.a = nn.Parameter(torch.tensor(a), requires_grad=learn_edges)
        self.w = nn.Parameter(torch.tensor(w), requires_grad=learn_edges)
        self.register_buffer("share", torch.tensor([comm.SHARES[u % comm.UNITS_PER_GEN] for u in range(wiring.N_UNIT)], dtype=torch.float32))

    def forward(self, flags: torch.Tensor) -> torch.Tensor:
        cc, gw, rtu = flags[:, 0:1], flags[:, 1:1 + wiring.N_GW], flags[:, 1 + wiring.N_GW:]
        ok = 1 - torch.prod(1 - torch.sigmoid(self.a)[None] * gw[:, None, :], dim=2)
        r = cc * rtu * ok
        return (r * self.share[None]) @ torch.softmax(self.w, dim=1)

    def edge_recovery(self):
        pred = (torch.sigmoid(self.a) > 0.5).cpu().numpy()
        true = wiring.true_unit_parent_matrix() > 0
        tp = (pred & true).sum(); fp = (pred & ~true).sum(); fn = (~pred & true).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        gen_acc = float((torch.argmax(self.w, 1).cpu().numpy() == wiring.TRUE_UNIT_GEN).mean())
        return float(f1), gen_acc


class ServiceValueNet(nn.Module):
    """service graph -> commandable fractions -> monotone value head."""

    def __init__(self, parents, unit_gen, d_x: int, learn_edges: bool = True):
        super().__init__()
        self.svc = ServiceGraph(parents, unit_gen, learn_edges)
        self.head = MonotoneValue(d_x, wiring.N_GEN)

    def forward(self, x: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        return self.head(x, self.svc(flags))
