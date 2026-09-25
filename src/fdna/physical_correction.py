"""Phase 5 Stage 2: physical (island-floor) correction to the g2 Mobius composition. F(S) is the exact
demand fraction stranded in generator-less islands after outage set S (src/fdna/hik_diag.minimal_cut_struct,
already correctness-gated against ScenarioLP.struct_mw). Three composable variants of the N-3 value estimate
given N-1/N-2 information:

  g2_plain(S,c)   = sum_i V({i},c) + sum_{pairs} I(pair,c)          (order-2 Mobius truncation, as before)
  g2_clipped(S,c) = clip(g2_plain(S,c), F(S), 1)                     (known topological floor enforced)
  g2_residual(S,c)= F(S) + [order-2 Mobius truncation of (V(T,c)-F(T)) over T subset S]
                  = F(S) + sum_i (V({i},c)-F({i})) + sum_{pairs} ((V(pair,c)-F(pair)) - (V({a},c)-F({a})) - (V({b},c)-F({b})))

F({i}) and F(pair) are computed the same way (graph connectivity only, no LP) so this never double-counts the
floor: it truncates the RESIDUAL series (V-F), not V itself, then adds F(S) back exactly once.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .grid import Grid, validate_outage
from .hik_diag import minimal_cut_struct
from .lp import OperatingPoint, Params


def island_floor(grid: Grid, demand: np.ndarray, outage: tuple[int, ...]) -> float:
    """F(S) = exact fraction of total demand stranded in generator-less islands after removing `outage`."""
    outage = validate_outage(grid, outage)
    struct_mw, _ = minimal_cut_struct(grid, outage, demand=demand)
    return struct_mw / demand.sum()


def capacity_floor(grid: Grid, op: OperatingPoint, params: Params, outage: tuple[int, ...], c: np.ndarray) -> float:
    """Stage R5 (external review's proposed mechanism, independently verified against the review's own toy LP:
    generator-less floor 0%, capacity floor 20%, repo's exact LP 20% -- exact agreement).

    island_floor only catches ISLANDS WITH NO GENERATOR AT ALL. An island can contain a generator and still lack
    enough reachable capacity to serve its own load. F_capacity(S,c) = sum_I max(0, D_I - sum_{g in I} U_g(c)) /
    total_demand, where U_g(c) = min(p0_g + rng_g(c), pmax_g) is generator g's maximum deliverable output under
    control c -- the SAME upper corrective bound ScenarioLP._bounds enforces (lp.py:119-127), so this uses the
    real control-to-capacity mapping, not an approximation of it.

    Like island_floor, this ignores intra-island congestion (routing/branch-rating limits inside a still-
    connected island), so it is a valid LOWER bound on the true LP shed, not an exact value -- gated in
    tests/test_capacity_floor.py against real ScenarioLP solves (capacity_floor <= exact shed) and against
    island_floor (capacity_floor >= island_floor, since a generator-less island's own term is identical and
    every other island can only add further deficit)."""
    outage = validate_outage(grid, outage)
    demand = op.demand
    pmax, p0 = grid.gen_pmax, op.p0
    c = np.asarray(c, float)
    rng = np.r_[params.local_mw, params.ramp_frac * pmax[1:] * c]
    U = np.minimum(p0 + rng, pmax)

    alive = np.ones(grid.n_branch, bool)
    alive[list(outage)] = False
    k = np.flatnonzero(alive)
    adj = coo_matrix((np.ones(len(k)), (grid.frm[k], grid.to[k])), shape=(grid.n_bus, grid.n_bus))
    n_isl, isl = connected_components(adj, directed=False)
    gen_isl = isl[grid.gen_bus]
    cap_by_isl = np.bincount(gen_isl, weights=U, minlength=n_isl)
    dem_by_isl = np.bincount(isl, weights=demand, minlength=n_isl)
    deficit = np.maximum(dem_by_isl - cap_by_isl, 0.0)
    return float(deficit.sum() / demand.sum())


def g2_clipped(g2_value: float, grid: Grid, demand: np.ndarray, outage: tuple[int, ...]) -> float:
    f = island_floor(grid, demand, outage)
    return float(np.clip(g2_value, f, 1.0))


def g2_residual(y_singles: dict, y_pairs: dict, grid: Grid, demand: np.ndarray, outage: tuple[int, ...]) -> float:
    """Raw residual-series estimate.

    y_singles: {branch: V({branch},c)}; y_pairs: {(a,b): V({a,b},c)} for the 3 sub-pairs of `outage`.
    This does not enforce the floor bound; use g2_clipped(raw, ...) when the caller needs the lower bound.
    """
    outage = validate_outage(grid, outage)
    a, b, c = outage
    F = lambda S: island_floor(grid, demand, S)
    Fs = F(outage)
    r_single = {i: y_singles[i] - F((i,)) for i in outage}
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    r_pair_interactions = []
    for p in pairs:
        r_pair = y_pairs[p] - F(p)
        r_pair_interactions.append(r_pair - r_single[p[0]] - r_single[p[1]])
    return float(Fs + sum(r_single.values()) + sum(r_pair_interactions))
