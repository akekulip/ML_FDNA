"""Feature builders for Milestone 1 baselines. No LP outputs are used as inputs."""
from __future__ import annotations

import numpy as np

from . import comm, spec
from .grid import BASE_MVA, Grid

def post_outage_flows(g: Grid, demand: np.ndarray, p0: np.ndarray, removed: tuple[int, ...], rating: np.ndarray):
    """Cheap DC flow after outages (no LP). Island imbalance is spread over that island's generators.
    Returns (flow/rating per branch (0 if removed), structural MW of gen-less islands)."""
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse import coo_matrix

    n = g.n_bus
    alive = np.ones(g.n_branch, bool)
    alive[list(removed)] = False
    k = np.flatnonzero(alive)
    w = BASE_MVA / g.x[k]
    adj = coo_matrix((np.ones(len(k)), (g.frm[k], g.to[k])), shape=(n, n))
    n_isl, isl = connected_components(adj, directed=False)
    inj = -demand.copy()
    np.add.at(inj, g.gen_bus, p0)
    struct = 0.0
    theta = np.zeros(n)
    Lap = np.zeros((n, n))
    np.add.at(Lap, (g.frm[k], g.frm[k]), w)
    np.add.at(Lap, (g.to[k], g.to[k]), w)
    np.add.at(Lap, (g.frm[k], g.to[k]), -w)
    np.add.at(Lap, (g.to[k], g.frm[k]), -w)
    gen_isl = isl[g.gen_bus]
    for i in range(n_isl):
        buses = np.flatnonzero(isl == i)
        gens = np.flatnonzero(gen_isl == i)
        if len(gens) == 0:
            struct += demand[buses].sum()
            inj[buses] = 0.0
            continue
        imb = inj[buses].sum()
        share = p0[gens] / p0[gens].sum() if p0[gens].sum() > 1e-9 else np.full(len(gens), 1 / len(gens))
        np.add.at(inj, g.gen_bus[gens], -imb * share)
        if len(buses) > 1:
            sub = Lap[np.ix_(buses[1:], buses[1:])]
            theta[buses[1:]] = np.linalg.solve(sub, inj[buses[1:]])
    flow = np.zeros(g.n_branch)
    flow[k] = w * (theta[g.frm[k]] - theta[g.to[k]])
    return np.abs(flow) / rating, struct / demand.sum()


def control_features(fs_ids: np.ndarray):
    """Explicit service calculation: (commandable fraction per remote generator, unit reachability, total)."""
    C = np.array([comm.control_fraction(comm.state_of(spec.FAILURE_SETS[i])) for i in fs_ids])
    U = np.array([comm.unit_reachable(comm.state_of(spec.FAILURE_SETS[i])) for i in fs_ids]).astype(float)
    return C, U, C.sum(axis=1, keepdims=True)


def raw_comm_states(fs_ids: np.ndarray):
    S = np.ones((len(fs_ids), comm.N_COMP))
    for r, i in enumerate(fs_ids):
        S[r, list(spec.FAILURE_SETS[i])] = 0.0
    return S
