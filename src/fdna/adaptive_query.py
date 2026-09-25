"""Phase 5 Stage 3: bounded adaptive oracle queries, exploiting exact control-monotonicity (proven from the
LP's bound structure, verified exhaustively -- src/fdna/tests/test_monotone_control.py: zero violations across
415.7M ordered pairs). For a fixed (op, outage), given a set of QUERIED exact values {(c_j, V(c_j))}:

  L(c) = max(F(S), max_{c_j >= c} V(c_j))    (a queried point that DOMINATES c gives a valid lower bound,
                                                since more control can only lower shed further; F(S) is the
                                                exact structural floor, also a valid unconditional lower bound)
  U(c) = min(1, min_{c_j <= c} V(c_j))       (a queried point DOMINATED BY c gives a valid upper bound)

Both are provably valid regardless of the guiding heuristic (g2) -- monotonicity, not the surrogate, is what
makes them exact bounds. g2 is used only to CHOOSE which c_j to query next (an acquisition heuristic), never to
adjust the bounds themselves.
"""
from __future__ import annotations

import numpy as np


def bounds(CV: np.ndarray, queried_idx: np.ndarray, queried_val: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """CV: (n_cv,5) all control vectors. queried_idx/queried_val: the subset actually queried so far.
    Returns (L, U), each (n_cv,), valid for EVERY control vector, not just the queried ones. Vectorised (no
    Python loop over n_cv): for n_q queried points this is O(n_cv * n_q), not O(n_cv^2)."""
    n_cv = len(CV)
    if len(queried_idx) == 0:
        return np.full(n_cv, floor), np.full(n_cv, 1.0)
    Cq = CV[queried_idx]                                            # (n_q, 5)
    dom_by_c = (Cq[None, :, :] >= CV[:, None, :]).all(2)             # (n_cv, n_q): queried j dominates target i
    dominated_by_c = (Cq[None, :, :] <= CV[:, None, :]).all(2)       # (n_cv, n_q): queried j dominated by target i
    qv = queried_val
    L = np.where(dom_by_c, qv[None, :], -np.inf).max(1); L = np.maximum(L, floor)
    U = np.where(dominated_by_c, qv[None, :], np.inf).min(1); U = np.minimum(U, 1.0)
    return L, U


def posterior_mass_bounds(pc: np.ndarray, L: np.ndarray, U: np.ndarray, tau: float) -> tuple[float, float]:
    """qL = P(L(c) > tau), qU = P(U(c) > tau) under the posterior pc over control vectors."""
    return float((pc * (L > tau)).sum()), float((pc * (U > tau)).sum())
