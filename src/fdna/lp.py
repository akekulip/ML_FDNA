"""DC corrective load-shedding LP (reference labels).

Minimise total shed subject to nodal balance, post-contingency branch ratings,
generator limits and bounded corrective ranges. The local balancer (generator 0)
is always commandable within +-local_mw; remote generators are commandable only
within their available control fraction. Generation may trip (protection) only to rebalance an island's
shed plus its initial surplus, so tripping is not free downward redispatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
from scipy.sparse.csgraph import connected_components

from .grid import BASE_MVA, Grid


@dataclass(frozen=True)
class Params:
    ramp_frac: float  # corrective range of a remote generator = ramp_frac * Pmax
    local_mw: float  # local balancer range (+-)


@dataclass
class OperatingPoint:
    demand: np.ndarray  # (n_bus,) MW
    p0: np.ndarray  # (G,) pre-contingency dispatch MW
    rating: np.ndarray  # (L,) MW post-contingency limit


class ScenarioLP:
    """LP for one (operating point, set of removed branches); only bounds change with control."""

    def __init__(self, grid: Grid, op: OperatingPoint, removed: tuple[int, ...], params: Params):
        self.g, self.op, self.params = grid, op, params
        n, G = grid.n_bus, grid.n_gen
        alive = np.ones(grid.n_branch, dtype=bool)
        alive[list(removed)] = False
        self.alive = alive
        k = np.flatnonzero(alive)
        f, t = grid.frm[k], grid.to[k]
        w = BASE_MVA / grid.x[k]

        adj = coo_matrix((np.ones(len(k)), (f, t)), shape=(n, n))
        self.n_isl, self.isl = connected_components(adj, directed=False)

        # Laplacian in MW/rad
        L = coo_matrix(
            (np.r_[w, w, -w, -w], (np.r_[f, t, f, t], np.r_[f, t, t, f])), shape=(n, n)
        ).tocsr()
        gen_inc = coo_matrix((np.ones(G), (grid.gen_bus, np.arange(G))), shape=(n, G)).tocsr()
        # variables: theta(n) | s(n) | delta(G) | kappa(G)
        I = csr_matrix(np.eye(n))
        self.A_eq = hstack([L, -I, -gen_inc, gen_inc], format="csr")
        p0_bus = np.zeros(n)
        np.add.at(p0_bus, grid.gen_bus, op.p0)
        self.b_eq = p0_bus - op.demand

        # flow limits: +-w(theta_f - theta_t) <= rating
        m = len(k)
        ar = np.arange(m)
        flow = coo_matrix(
            (np.r_[w, -w, -w, w], (np.r_[ar, ar, ar + m, ar + m], np.r_[f, t, f, t])), shape=(2 * m, n)
        ).tocsr()
        rating = op.rating[k]
        A_flow = hstack([flow, csr_matrix((2 * m, n + 2 * G))], format="csr")
        b_flow = np.r_[rating, rating]

        # per-generator: -delta + kappa <= p0  (output stays >= 0)
        gz = csr_matrix((G, 2 * n))
        A_g = hstack([gz, -csr_matrix(np.eye(G)), csr_matrix(np.eye(G))], format="csr")
        b_g = op.p0.copy()

        # island surplus: sum kappa in island <= surplus
        gen_isl = self.isl[grid.gen_bus]
        dem_isl = np.bincount(self.isl, weights=op.demand, minlength=self.n_isl)
        gen_out = np.bincount(gen_isl, weights=op.p0, minlength=self.n_isl)
        self.surplus = np.maximum(gen_out - dem_isl, 0.0)
        # protection trips: kappa in an island may only rebalance the island's shed plus its initial
        # surplus, so it cannot act as free downward redispatch: sum(kappa) - sum(s) <= surplus
        rows_s = np.zeros((self.n_isl, 2 * n + 2 * G))
        for i in range(self.n_isl):
            rows_s[i, np.flatnonzero(self.isl == i) + n] = -1.0
            rows_s[i, 2 * n + G + np.flatnonzero(gen_isl == i)] = 1.0
        blocks = [A_flow, A_g, csr_matrix(rows_s)]
        rhs = [b_flow, b_g, self.surplus]
        self.A_ub = vstack(blocks, format="csr")
        self.b_ub = np.concatenate(rhs)

        self.c_obj = np.r_[np.zeros(n), np.ones(n), np.zeros(2 * G)]
        # static bounds
        self.lb = np.r_[np.full(n, -np.inf), np.zeros(n), np.zeros(2 * G)]
        self.ub = np.r_[np.full(n, np.inf), op.demand, np.zeros(2 * G)]
        ref = {}
        for b in range(n):
            ref.setdefault(self.isl[b], b)
        for b in ref.values():
            self.lb[b] = self.ub[b] = 0.0
        self.ub[2 * n + G :] = op.p0
        # structural shed: demand of islands with no generator
        has_gen = np.bincount(gen_isl, minlength=self.n_isl) > 0
        self.struct_mw = float(dem_isl[~has_gen].sum())
        self.total = float(op.demand.sum())
        self._n, self._G = n, G

    def _bounds(self, c: np.ndarray):
        n, G = self._n, self._G
        p = self.params
        lb, ub = self.lb.copy(), self.ub.copy()
        pmax, p0 = self.g.gen_pmax, self.op.p0
        rng = np.r_[p.local_mw, p.ramp_frac * pmax[1:] * c]
        lb[2 * n : 2 * n + G] = -np.minimum(rng, p0)
        ub[2 * n : 2 * n + G] = np.minimum(rng, pmax - p0)
        return np.c_[lb, ub]

    def shed_mw(self, c: np.ndarray) -> float:
        """c: (G-1,) commandable fraction of each remote generator's range."""
        res = linprog(
            self.c_obj,
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            A_eq=self.A_eq,
            b_eq=self.b_eq,
            bounds=self._bounds(np.asarray(c, float)),
            method="highs",
        )
        if res.status != 0:
            raise RuntimeError(f"LP status {res.status}: {res.message}")
        return float(res.fun)

    def y(self, c: np.ndarray) -> float:
        return self.shed_mw(c) / self.total
