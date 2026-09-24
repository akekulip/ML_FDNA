"""Phase 4 Stage 3 (brief section 6.2): versioned schema for HIGHER-K outage sets (k>2), kept disjoint from the
pair-only v2/v2data/dataset layout so Phase 1-3 remain exactly reproducible. ScenarioLP already accepts an
arbitrary removed-branch tuple (src/fdna/lp.py) -- this module adds canonical IDs, manifests and sampling on top
of it, deriving every dimension from the grid instead of hard-coding IEEE-30/pair-specific constants."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .dataset import G, RATING
from .grid import Grid
from .lp import ScenarioLP
from .opgen import sample_op
from .spec import PARAMS


def outage_key(branches: tuple[int, ...]) -> tuple[int, ...]:
    """Canonical sorted, deduplicated outage-set identifier. () = N-0."""
    return tuple(sorted(set(int(b) for b in branches)))


def n_possible_ksets(n_branch: int, k: int) -> int:
    from math import comb
    return comb(n_branch, k)


def is_feasible(grid: Grid, branches: tuple[int, ...]) -> bool:
    """Cheap pre-check: every branch id is distinct and in range. Does not run the LP (islanding/infeasible
    control combinations are counted, not filtered, by the caller -- see sample_kset_manifest)."""
    key = outage_key(branches)
    return len(key) == len(branches) and all(0 <= b < grid.n_branch for b in key)


@dataclass(frozen=True)
class KSetManifest:
    """A frozen (seed-reproducible) list of outage sets for one cardinality k, sampled BEFORE any label is generated."""
    k: int
    op_ids: np.ndarray          # (n_op,)
    outage_sets: list[tuple[int, ...]]   # (n_sets,) canonical sorted tuples, distinct
    strata: np.ndarray          # (n_sets,) str labels: 'representative' | 'stress' (topology-flagged via lodf risk / islanding)
    seed: int


def sample_kset_manifest(grid: Grid, k: int, n_sets: int, seed: int, stress_frac: float = 0.3) -> KSetManifest:
    """Freeze n_sets distinct k-subsets of branches: (1-stress_frac) uniformly at random, stress_frac biased toward
    high LODF-pair risk / known-bridge branches (from src/fdna/lodf.py, no LP calls) so genuinely hard higher-k
    cases are represented without exhaustively enumerating C(n_branch, k). Reported, not silently dropped: the
    achieved stratum counts (rounding + rejection sampling on duplicates) may differ slightly from the nominal split."""
    from .lodf import ptdf_lodf
    rng = np.random.default_rng(seed)
    total = n_possible_ksets(grid.n_branch, k)
    if n_sets > total:
        raise ValueError(f"requested {n_sets} distinct {k}-sets but only {total} exist")
    _, LODF = ptdf_lodf(grid)
    pair_risk = np.zeros(grid.n_branch)
    for a, b in combinations(range(grid.n_branch), 2):
        det = 1.0 - LODF[a, b] * LODF[b, a]
        r = 1.0 / max(abs(det), 1e-9)
        pair_risk[a] = max(pair_risk[a], r); pair_risk[b] = max(pair_risk[b], r)
    branch_p = pair_risk / pair_risk.sum()

    seen: set[tuple[int, ...]] = set()
    sets_, strata = [], []
    n_stress = int(round(n_sets * stress_frac))
    attempts, max_attempts = 0, n_sets * 200
    while len(sets_) < n_sets and attempts < max_attempts:
        attempts += 1
        want_stress = len(strata) < n_stress
        if want_stress:
            branches = tuple(rng.choice(grid.n_branch, size=k, replace=False, p=branch_p))
        else:
            branches = tuple(rng.choice(grid.n_branch, size=k, replace=False))
        key = outage_key(branches)
        if key in seen:
            continue
        seen.add(key); sets_.append(key); strata.append("stress" if want_stress else "representative")
    if len(sets_) < n_sets:
        raise RuntimeError(f"only sampled {len(sets_)}/{n_sets} distinct {k}-sets in {max_attempts} attempts")
    return KSetManifest(k=k, op_ids=np.array([]), outage_sets=sets_, strata=np.array(strata), seed=seed)


def label_kset(grid: Grid, op_id: int, outage: tuple[int, ...], cv: np.ndarray) -> tuple[np.ndarray, bool, str]:
    """Exact oracle label: shed/total for every row of cv (n_cv, G-1). Returns (y (n_cv,), any_infeasible, note).
    Infeasible/failed LP solves are COUNTED, not silently dropped (brief 6.2)."""
    op = sample_op(grid, RATING, np.random.default_rng(op_id))
    lp = ScenarioLP(grid, op, outage, PARAMS)
    y = np.zeros(len(cv)); failed = 0
    for i in range(len(cv)):
        try:
            y[i] = lp.y(cv[i])
        except RuntimeError:
            y[i] = np.nan; failed += 1
    return y, failed > 0, f"{failed}/{len(cv)} control vectors infeasible" if failed else ""
