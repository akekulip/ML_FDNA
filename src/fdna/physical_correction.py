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

from .grid import Grid
from .hik_diag import minimal_cut_struct


def island_floor(grid: Grid, demand: np.ndarray, outage: tuple[int, ...]) -> float:
    """F(S) = exact fraction of total demand stranded in generator-less islands after removing `outage`."""
    struct_mw, _ = minimal_cut_struct(grid, outage, demand=demand)
    return struct_mw / demand.sum()


def g2_clipped(g2_value: float, grid: Grid, demand: np.ndarray, outage: tuple[int, ...]) -> float:
    f = island_floor(grid, demand, outage)
    return float(np.clip(g2_value, f, 1.0))


def g2_residual(y_singles: dict, y_pairs: dict, grid: Grid, demand: np.ndarray, outage: tuple[int, ...]) -> float:
    """y_singles: {branch: V({branch},c)}; y_pairs: {(a,b): V({a,b},c)} for the 3 sub-pairs of `outage`."""
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
