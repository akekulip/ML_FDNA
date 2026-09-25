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


PROB_DECISION_THRESHOLD = 0.5
"""The probability-decision cutoff `predict_from_bounds` and `resolve` use to call a candidate severe/not-
severe. Named and factored out separately from `spec.SEVERE` (the LOAD-SHED severity threshold, ~0.01) after
an external review (repair round 3) found `resolve`'s stopping rule had conflated the two -- `qU <= spec.SEVERE`
where the correct negative-certification condition is `qU <= PROB_DECISION_THRESHOLD`. The bug made the stopping
rule needlessly conservative (wasted queries) but never produced an invalid certificate, since spec.SEVERE (0.01)
< PROB_DECISION_THRESHOLD (0.5) made the old condition MORE restrictive, not less. Giving the two thresholds
distinct names/locations is the actual fix -- it makes the conflation a type-level impossibility, not just a
corrected literal."""


def predict_from_bounds(qL: float, qU: float) -> bool:
    """The severe/not-severe prediction rule used by scripts/p5_adaptive_query_experiment_v2.py, factored out
    so the script and its regression test (tests/test_adaptive_query_integrity.py) call the exact same function
    rather than two hand-copied copies of the formula that could silently drift apart. Its signature is a
    structural guarantee against external review finding 4 (hidden-state leakage): this function has no
    parameter through which a hidden true control state could enter -- it can only ever see (qL,qU), the
    posterior-mass bounds any real policy actually observes."""
    t = PROB_DECISION_THRESHOLD
    return bool((qL > t) if qL > t else ((qL + qU) / 2 > t))


def resolve(policy: str, y_true_grid: np.ndarray, floor: float, g2grid: np.ndarray, pc_support: np.ndarray,
            budget: int, guided_rng, CV: np.ndarray, tau: float) -> tuple[int, float, float]:
    """Adaptively query up to `budget` controls; stop once qL==qU over pc_support OR the binary decision is
    already certified. Returns (n_queries_used, qL, qU) ONLY -- L/U at the true index are never exposed to the
    caller's prediction logic.

    Moved here from scripts/p5_adaptive_query_experiment_v2.py (repair round 3) alongside the threshold fix, so
    both the script and any future script (e.g. a control-variate policy comparison) share one implementation
    instead of risking a second hand-copied version drifting out of sync -- the exact class of bug this review
    round is fixing elsewhere (see PROB_DECISION_THRESHOLD's own docstring).

    `tau`: the LOAD-SHED severity threshold (spec.SEVERE), used to test whether a control state's shed exceeds
    the severity definition. Never confused with PROB_DECISION_THRESHOLD (0.5), the probability-space cutoff
    used to certify the binary decision -- external review, repair round 3, finding 2: `resolve`'s stopping
    rule used to compare `qU` (a probability) against `tau` (a load-shed fraction), an overly conservative but
    not incorrect condition (wasted queries, never an invalid certificate, since tau=0.01 < 0.5 made the old
    condition MORE restrictive). Decision-aware early exit checked BEFORE the full-resolution check, so it
    fires first whenever it applies (repair round 2)."""
    pn = pc_support / pc_support.sum() if pc_support.sum() > 0 else pc_support
    queried_idx, queried_val = [], []
    extremes = [int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))]
    for e in extremes:
        queried_idx.append(e); queried_val.append(y_true_grid[e])
    L, U = bounds(CV, np.array(queried_idx), np.array(queried_val), floor)
    while len(queried_idx) < budget:
        qL, qU = posterior_mass_bounds(pn, L, U, tau)
        if qL > PROB_DECISION_THRESHOLD or qU <= PROB_DECISION_THRESHOLD:
            break  # decision already certified -- further queries cannot change the binary prediction
        support = np.flatnonzero(pc_support)
        unresolved_support = support[(L[support] <= tau) & (U[support] > tau)]
        if len(unresolved_support) == 0:
            break
        unqueried_unresolved = np.array([i for i in unresolved_support if i not in queried_idx])
        if len(unqueried_unresolved) == 0:
            break
        if policy == "g2_guided_bounds":
            nxt = unqueried_unresolved[np.argmin(np.abs(g2grid[unqueried_unresolved] - tau))]
        else:
            nxt = unqueried_unresolved[guided_rng.integers(len(unqueried_unresolved))]
        queried_idx.append(int(nxt)); queried_val.append(y_true_grid[nxt])
        L, U = bounds(CV, np.array(queried_idx), np.array(queried_val), floor)
    qL, qU = posterior_mass_bounds(pn, L, U, tau)
    return len(queried_idx), qL, qU
