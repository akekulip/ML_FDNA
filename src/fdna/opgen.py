"""Operating-point generation: nominal ratings, random demand, cost-minimal base dispatch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from .grid import BASE_MVA, Grid
from .lp import OperatingPoint

BASE_COST = np.array([40.0, 20.0, 25.0, 30.0, 35.0, 28.0])
SLACK_MIN_MW = 80.0  # local balancer is must-run (spinning reserve), so it has range both ways


@dataclass(frozen=True)
class RatingSpec:
    kappa: float  # rating = max(kappa * |nominal flow|, floor_mw)
    floor_mw: float


def _dc_dispatch(grid: Grid, demand: np.ndarray, cost: np.ndarray, rating: np.ndarray | None):
    n, G, L = grid.n_bus, grid.n_gen, grid.n_branch
    w = BASE_MVA / grid.x
    Lap = coo_matrix(
        (np.r_[w, w, -w, -w], (np.r_[grid.frm, grid.to, grid.frm, grid.to], np.r_[grid.frm, grid.to, grid.to, grid.frm])),
        shape=(n, n),
    ).tocsr()
    gen_inc = coo_matrix((np.ones(G), (grid.gen_bus, np.arange(G))), shape=(n, G)).tocsr()
    A_eq = hstack([Lap, -gen_inc], format="csr")  # L theta - P = -d
    b_eq = -demand
    A_ub = b_ub = None
    if rating is not None:
        flow = coo_matrix(
            (np.r_[w, -w, -w, w], (np.r_[np.arange(L), np.arange(L), np.arange(L) + L, np.arange(L) + L],
                                   np.r_[grid.frm, grid.to, grid.frm, grid.to])),
            shape=(2 * L, n),
        ).tocsr()
        A_ub = hstack([flow, csr_matrix((2 * L, G))], format="csr")
        b_ub = np.r_[rating, rating]
    bounds = [(None, None)] * n + [(0.0, float(pm)) for pm in grid.gen_pmax]
    bounds[0] = (0.0, 0.0)
    bounds[n] = (SLACK_MIN_MW, float(grid.gen_pmax[0]))
    res = linprog(np.r_[np.zeros(n), cost], A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.status != 0:
        return None, None
    theta, P = res.x[:n], res.x[n:]
    flows = w * (theta[grid.frm] - theta[grid.to])
    return P, flows


def nominal_rating(grid: Grid, spec: RatingSpec) -> np.ndarray:
    P, flows = _dc_dispatch(grid, grid.load, BASE_COST, None)
    assert P is not None
    return np.maximum(spec.kappa * np.abs(flows), spec.floor_mw)


def sample_op(grid: Grid, rating: np.ndarray, rng: np.random.Generator,
              load_range=(0.85, 1.15), bus_noise=0.1, max_tries=50) -> OperatingPoint:
    for _ in range(max_tries):
        lam = rng.uniform(*load_range)
        d = grid.load * lam * np.clip(1 + bus_noise * rng.standard_normal(grid.n_bus), 0.7, 1.3)
        cost = BASE_COST * rng.uniform(0.7, 1.3, size=grid.n_gen)
        P, _ = _dc_dispatch(grid, d, cost, rating)
        if P is not None:
            return OperatingPoint(demand=d, p0=P, rating=rating)
    raise RuntimeError("no feasible operating point found")
