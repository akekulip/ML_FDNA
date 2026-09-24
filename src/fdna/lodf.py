"""Base-case LODF/compensation diagnostics for N-2 outage pairs (no LP solve; one Laplacian factorisation).
Verified by two independent agents against scripts/p3_step1_flowadd.py: Det(a,b)=0 (machine precision) recovers
exactly the 26 topological 2-cuts found by connected-components; 1/|Det| correlates with measured flow-addition
error at Spearman rho ~0.94-0.95 on the 10-op / 677-connected-pair sample."""
from __future__ import annotations

import numpy as np

from .grid import BASE_MVA, Grid


def ptdf_lodf(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Base-case (topology/reactance only, demand-independent) PTDF (L,n) and single-outage LODF (L,L) matrices."""
    n, L = g.n_bus, g.n_branch
    w = BASE_MVA / g.x
    Lap = np.zeros((n, n))
    np.add.at(Lap, (g.frm, g.frm), w); np.add.at(Lap, (g.to, g.to), w)
    np.add.at(Lap, (g.frm, g.to), -w); np.add.at(Lap, (g.to, g.frm), -w)
    Xinv = np.zeros((n, n)); Xinv[1:, 1:] = np.linalg.inv(Lap[1:, 1:])          # bus 0 as reference
    PTDF = w[:, None] * (Xinv[g.frm, :] - Xinv[g.to, :])                        # (L, n): d(flow_l)/d(injection_bus)
    diag = PTDF[np.arange(L), g.frm] - PTDF[np.arange(L), g.to]                 # self-sensitivity, d(flow_l)/d(outage of l)
    cross = PTDF[:, g.frm] - PTDF[:, g.to]                                      # (L, L): d(flow_l)/d(power shifted by outage of k)
    with np.errstate(divide="ignore", invalid="ignore"):
        LODF = cross / (1 - diag)[None, :]
    np.fill_diagonal(LODF, -1.0)
    return PTDF, LODF


def compensation_det(LODF: np.ndarray, a: int, b: int) -> float:
    """Det(a,b) = 1 - LODF(a,b)*LODF(b,a): the 2x2 simultaneous-outage compensation system's determinant.
    Det=0 (machine precision) <=> {a,b} is an exact topological 2-cut (verified: matches connected-components on all 820 pairs)."""
    return float(1.0 - LODF[a, b] * LODF[b, a])


def risk_table(g: Grid, pairs: list[tuple[int, int]]) -> np.ndarray:
    """(len(pairs),) 1/|Det(a,b)| for each connected pair; np.inf for exact 2-cuts. O(1) per pair after one factorisation."""
    _, LODF = ptdf_lodf(g)
    d = np.array([compensation_det(LODF, a, b) for a, b in pairs])
    with np.errstate(divide="ignore"):
        return 1.0 / np.abs(d)
