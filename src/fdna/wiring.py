"""Assumed-wiring generator with edge corruption (Branch 2) and control features under a given wiring."""
from __future__ import annotations

import numpy as np

from . import comm, spec

N_GEN, N_UNIT, N_GW = comm.N_REMOTE_GEN, comm.N_UNIT, comm.N_GW
TRUE_PARENTS = {g: tuple(p) for g, p in comm.PARENT_GW.items()}
TRUE_UNIT_GEN = np.arange(N_UNIT) // comm.UNITS_PER_GEN
N_EDGES = sum(len(p) for p in TRUE_PARENTS.values()) + N_UNIT  # 7 gateway->generator + 10 unit->generator = 17


def corrupt(rho: float, seed: int):
    """Rewire round(rho * 17) randomly chosen edges to a wrong endpoint. Returns (parents dict, unit_gen array)."""
    rng = np.random.default_rng(seed)
    parents = {g: list(p) for g, p in TRUE_PARENTS.items()}
    unit_gen = TRUE_UNIT_GEN.copy()
    edges = [("p", g, i) for g, p in TRUE_PARENTS.items() for i in range(len(p))] + [("u", u, 0) for u in range(N_UNIT)]
    k = int(round(rho * N_EDGES))
    for e in rng.choice(len(edges), k, replace=False):
        kind, a, i = edges[e]
        if kind == "p":
            choices = [p for p in range(N_GW) if p not in parents[a]]
            parents[a][i] = int(rng.choice(choices))
        else:
            unit_gen[a] = int(rng.choice([g for g in range(N_GEN) if g != unit_gen[a]]))
    return {g: tuple(p) for g, p in parents.items()}, unit_gen


def control_vectors_under(parents: dict, unit_gen: np.ndarray) -> np.ndarray:
    """(n_failure_sets, 5) commandable fractions computed with an ASSUMED wiring, for every failure set."""
    out = np.zeros((len(spec.FAILURE_SETS), N_GEN))
    for i, f in enumerate(spec.FAILURE_SETS):
        up = comm.state_of(f)
        for u in range(N_UNIT):
            g = unit_gen[u]
            if up[comm.CC] and up[comm.rtu(u)] and any(up[comm.gw(p)] for p in parents[g]):
                out[i, g] += comm.SHARES[u % comm.UNITS_PER_GEN]
    return out


def true_unit_parent_matrix() -> np.ndarray:
    m = np.zeros((N_UNIT, N_GW))
    for u in range(N_UNIT):
        m[u, list(TRUE_PARENTS[TRUE_UNIT_GEN[u]])] = 1
    return m
