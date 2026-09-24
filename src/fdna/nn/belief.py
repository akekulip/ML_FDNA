"""v2 neural arms operating on partial/stale observations: FDNA belief network (A5), shuffled (A6), unconstrained (A7),
and a generic min/max-aggregation GNN (A4). All share the same MonotoneValue head and identical inputs."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import comm, v2
from .layers import FDNALayer, MonotoneValue

N_C, N_F, N_U = comm.N_COMP, v2.N_FLAG, comm.N_UNIT


def flag_ancestor_mask() -> np.ndarray:
    """(13 flags, 14 components): which components determine each flag's true value."""
    m = np.zeros((N_F, N_C), np.float32)
    for u in range(N_U):
        g = u // comm.UNITS_PER_GEN
        m[u, [comm.CC, comm.rtu(u), *[comm.gw(p) for p in comm.PARENT_GW[g]]]] = 1
    for g in range(comm.N_GW):
        m[N_U + g, [comm.CC, comm.gw(g)]] = 1
    return m


def unit_input_mask() -> np.ndarray:
    return flag_ancestor_mask()[:N_U].copy()   # (10 units, 14 components): same dependency graph


def _shuffle_columns(m: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(m)
    for i in range(m.shape[0]):
        out[i, rng.choice(m.shape[1], int(m[i].sum()), replace=False)] = 1
    return out


def obs_tensor(obs: np.ndarray) -> np.ndarray:
    """(B,13) int8 in {-1,0,1} -> (B,13,2) float: [reported up, reported down]; missing = zeros."""
    return np.stack([(obs == 1), (obs == 0)], axis=2).astype(np.float32)


class BeliefFDNA(nn.Module):
    """variant: 'fdna' (true wiring masks), 'shuffled' (edge-shuffled masks), 'unconstrained' (all-ones masks, same size)."""

    def __init__(self, d_x: int, variant: str = "fdna", tau: float = 0.05, seed: int = 0):
        super().__init__()
        mf, mu = flag_ancestor_mask(), unit_input_mask()
        if variant == "shuffled":
            mf, mu = _shuffle_columns(mf, seed), _shuffle_columns(mu, seed + 1)
        elif variant == "unconstrained":
            mf, mu = np.ones_like(mf), np.ones_like(mu)
        self.register_buffer("mf", torch.tensor(mf))
        self.prior = nn.Parameter(torch.full((N_C,), 1.5))
        self.w1 = nn.Parameter(torch.randn(N_F, N_C) * 0.5 + 1.0)
        self.w0 = nn.Parameter(torch.randn(N_F, N_C) * 0.5 - 2.0)
        self.fdna = FDNALayer(torch.tensor(mu), tau=tau)
        self.register_buffer("share", torch.tensor([comm.SHARES[u % comm.UNITS_PER_GEN] for u in range(N_U)], dtype=torch.float32))
        onehot = torch.zeros(N_U, comm.N_REMOTE_GEN)
        onehot[torch.arange(N_U), torch.arange(N_U) // comm.UNITS_PER_GEN] = 1
        self.register_buffer("unit_gen", onehot)
        self.head = MonotoneValue(d_x, comm.N_REMOTE_GEN)

    def controls(self, e: torch.Tensor) -> torch.Tensor:
        logit = self.prior[None] + (e[:, :, 0:1] * (self.mf * self.w1)[None]).sum(1) + (e[:, :, 1:2] * (self.mf * self.w0)[None]).sum(1)
        o_comp = torch.sigmoid(logit)
        o_unit = self.fdna(o_comp)
        return (o_unit * self.share[None]) @ self.unit_gen

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.head(x, self.controls(e))


class MinMaxGNN(nn.Module):
    """Generic message passing with min/max neighbourhood aggregation on the dependency graph (no FDNA algebra)."""

    def __init__(self, d_x: int, hidden: int = 16, layers: int = 3):
        super().__init__()
        mf = flag_ancestor_mask()
        A = np.zeros((N_C + N_F, N_C + N_F), np.float32)
        A[N_C:, :N_C] = mf
        A[:N_C, N_C:] = mf.T
        A += np.eye(N_C + N_F, dtype=np.float32)
        self.register_buffer("A", torch.tensor(A > 0))
        self.emb = nn.Parameter(torch.randn(N_C + N_F, hidden) * 0.1)
        self.inp = nn.Linear(2, hidden)
        self.lins = nn.ModuleList([nn.Linear(3 * hidden, hidden) for _ in range(layers)])
        self.out = nn.Linear(N_F * hidden, comm.N_REMOTE_GEN)
        self.head = MonotoneValue(d_x, comm.N_REMOTE_GEN)

    def states(self, e: torch.Tensor) -> torch.Tensor:
        B = e.shape[0]
        h = self.emb[None].expand(B, -1, -1).clone()
        h[:, N_C:, :] = h[:, N_C:, :] + self.inp(e)
        big = 1e4
        for lin in self.lins:
            nb = self.A[None, :, :, None]                                 # (1,N,N,1)
            hx = h[:, None, :, :]                                         # (B,1,N,H)
            mn = torch.where(nb, hx, torch.full_like(hx, big)).min(dim=2).values
            mx = torch.where(nb, hx, torch.full_like(hx, -big)).max(dim=2).values
            h = torch.relu(lin(torch.cat([h, mn, mx], dim=2)))
        return h[:, N_C:, :].reshape(B, -1)

    def controls(self, e: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(self.states(e)))

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.head(x, self.controls(e))
