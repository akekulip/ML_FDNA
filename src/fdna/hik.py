"""Phase 4 Stage 3 (brief section 6.2): versioned schema for HIGHER-K outage sets (k>2), kept disjoint from the
pair-only v2/v2data/dataset layout so Phase 1-3 remain exactly reproducible. ScenarioLP already accepts an
arbitrary removed-branch tuple (src/fdna/lp.py) -- this module adds canonical IDs, manifests and sampling on top
of it, deriving every dimension from the grid instead of hard-coding IEEE-30/pair-specific constants."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .dataset import G, RATING
from .grid import Grid, validate_branch_ids, validate_outage
from .lp import ScenarioLP
from .opgen import BASE_COST, sample_op
from .spec import PARAMS


def outage_key(branches: tuple[int, ...]) -> tuple[int, ...]:
    """Canonical sorted outage-set identifier. () = N-0."""
    return validate_branch_ids(None, branches)


def n_possible_ksets(n_branch: int, k: int) -> int:
    from math import comb
    return comb(n_branch, k)


def is_feasible(grid: Grid, branches: tuple[int, ...]) -> bool:
    """Cheap pre-check: every branch id is distinct and in range. Does not run the LP (islanding/infeasible
    control combinations are counted, not filtered, by the caller -- see sample_kset_manifest)."""
    try:
        validate_outage(grid, branches)
    except ValueError:
        return False
    return True


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


def label_kset(grid: Grid, op_id: int, outage: tuple[int, ...], cv: np.ndarray, *,
               rating: np.ndarray | None = None, params=None) -> tuple[np.ndarray, bool, str]:
    """Exact oracle label: shed/total for every row of cv (n_cv, G-1). Returns (y (n_cv,), any_infeasible, note).
    Infeasible/failed LP solves are COUNTED, not silently dropped (brief 6.2).

    `rating` permits alternative topologies that keep the same six-generator dispatch contract used by opgen's
    frozen BASE_COST vector. It is not a general arbitrary-generator-grid API.
    """
    outage = validate_outage(grid, outage)
    if grid.n_gen != len(BASE_COST):
        raise ValueError(
            f"label_kset requires the six-generator dispatch contract from opgen.BASE_COST; got {grid.n_gen} generators"
        )
    if rating is None:
        if grid is not G:
            raise ValueError("label_kset requires an explicit rating for non-default grids")
        rating = RATING
    if params is None:
        params = PARAMS
    op = sample_op(grid, rating, np.random.default_rng(op_id))
    lp = ScenarioLP(grid, op, outage, params)
    y = np.zeros(len(cv)); failed = 0
    for i in range(len(cv)):
        try:
            y[i] = lp.y(cv[i])
        except RuntimeError:
            y[i] = np.nan; failed += 1
    return y, failed > 0, f"{failed}/{len(cv)} control vectors infeasible" if failed else ""
