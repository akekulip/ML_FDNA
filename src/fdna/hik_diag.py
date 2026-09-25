"""Phase 4 Stage 2->3: two independent brainstorming agents (numerical-methods/control; compositional-learning/
reliability) both independently ranked, as their #1 idea, generalising the already-verified k=2 mechanisms to
arbitrary k -- both cheap, exact, and LP-solve-free (reuse the one Laplacian factorisation / plain graph
connectivity already used at k=2). This module implements both, each gated by an exact-reproduction check against
the already-verified k=2 facts before being trusted at k=3/4.

1. generalized_det(S): k-way compensation determinant det(M_S), M_S[i,i]=1, M_S[i,j]=-LODF(i,j). At k=2 this is
   EXACTLY lodf.compensation_det; MUST reproduce it and the 26 known 2-cuts before use.
2. minimal_cut_struct(S): exact generator-less-island shed floor for an outage set of any size, via plain
   connected_components (no LP). At k<=2 MUST exactly reproduce ScenarioLP.struct_mw for the same removed set."""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .grid import Grid, validate_branch_ids, validate_outage
from .lodf import ptdf_lodf


def generalized_det(LODF: np.ndarray, S: tuple[int, ...]) -> float:
    """det(M_S) for M_S[i,i]=1, M_S[i,j]=-LODF(S[i],S[j]) i!=j. At |S|=2 this equals 1-LODF(a,b)*LODF(b,a)
    (lodf.compensation_det) exactly -- checked in tests/test_hik_diag.py."""
    S = validate_branch_ids(LODF.shape[0], S)
    k = len(S)
    if k <= 1:
        return 1.0
    M = np.eye(k)
    for i in range(k):
        for j in range(k):
            if i != j:
                M[i, j] = -LODF[S[i], S[j]]
    return float(np.linalg.det(M))


def minimal_cut_struct(grid: Grid, S: tuple[int, ...], demand: np.ndarray | None = None) -> tuple[float, int]:
    """(struct_mw, n_islands) for outage set S: exact demand of generator-less islands after removing S, via
    connected_components only (no LP). demand defaults to the grid's nominal load (topology-only use, e.g. the
    new-cut check); pass the OPERATING POINT's actual demand (op.demand) to match ScenarioLP.struct_mw exactly --
    struct_mw depends on the sampled operating point's demand, not the nominal grid.load, and the two differ
    (checked in tests/test_hik_diag.py, which caught this)."""
    if demand is None:
        demand = grid.load
    S = validate_outage(grid, S)
    alive = np.ones(grid.n_branch, bool); alive[list(S)] = False
    k = np.flatnonzero(alive)
    adj = coo_matrix((np.ones(len(k)), (grid.frm[k], grid.to[k])), shape=(grid.n_bus, grid.n_bus))
    n_isl, isl = connected_components(adj, directed=False)
    gen_isl = isl[grid.gen_bus]
    has_gen = np.bincount(gen_isl, minlength=n_isl) > 0
    demand_by_isl = np.bincount(isl, weights=demand, minlength=n_isl)
    return float(demand_by_isl[~has_gen].sum()), int(n_isl)


def is_new_cut(grid: Grid, S: tuple[int, ...]) -> bool:
    """True iff S disconnects the grid AND no proper subset of S alone does (a genuinely NEW k-way cut, not one
    already implied by a smaller subset -- matches STEP1's 'new_2cut' axis, generalized to any k)."""
    from itertools import combinations
    S = validate_outage(grid, S)
    _, n_isl = minimal_cut_struct(grid, S)
    if n_isl <= 1:
        return False
    for r in range(1, len(S)):
        for sub in combinations(S, r):
            _, n_sub = minimal_cut_struct(grid, sub)
            if n_sub > 1:
                return False
    return True
