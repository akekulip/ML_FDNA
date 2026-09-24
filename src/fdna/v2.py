"""v2 world: hidden communication state, partial/stale observations, exact Bayes posterior (see prereg/SPEC_V2.md)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import comm

P_FAIL = 0.15
P_GROUP = 0.08
GROUPS = ((comm.gw(0), *(comm.rtu(u) for u in range(4))), (comm.gw(1), comm.gw(2)))
Q_GRID = (0.1, 0.3)
S_GRID = (0.0, 0.2)
CELLS = [(q, s) for q in Q_GRID for s in S_GRID]
N_COMP, N_UNIT, N_GW = comm.N_COMP, comm.N_UNIT, comm.N_GW
N_FLAG = N_UNIT + N_GW  # 10 unit poll flags + 3 gateway heartbeats
NEG = -60.0


def truth_flags(up: np.ndarray) -> np.ndarray:
    """up: (..., 14) bool -> (..., 13) bool true value of every observed flag."""
    up = np.atleast_2d(up)
    out = np.zeros((up.shape[0], N_FLAG), bool)
    for i, x in enumerate(up):
        out[i, :N_UNIT] = comm.unit_reachable(x)
        out[i, N_UNIT:] = [x[comm.CC] and x[comm.gw(g)] for g in range(N_GW)]
    return out


@dataclass
class World:
    X: np.ndarray  # (16384, 14) up flags
    T: np.ndarray  # (16384, 13) true flags
    logprior: np.ndarray  # (16384,)
    cidx: np.ndarray  # (16384,) control-vector index
    CV: np.ndarray  # (n_cv, 5) distinct commandable-fraction vectors
    A: np.ndarray  # (16384, 13) log e(1|T)
    B: np.ndarray  # (16384, 13) log e(0|T) (s-dependent, set by with_s)

    def with_s(self, s: float) -> "World":
        A = np.where(self.T, 0.0, np.log(s) if s > 0 else NEG)
        B = np.where(self.T, NEG, np.log(1.0 - s))
        return World(self.X, self.T, self.logprior, self.cidx, self.CV, np.maximum(A, NEG), B)


def build_world() -> World:
    n = N_COMP
    X = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    down = ~X
    # prior: mixture over common-cause events (each active with prob P_GROUP)
    p = np.zeros(len(X))
    for e in range(1 << len(GROUPS)):
        cov = np.zeros(n, bool)
        pe = 1.0
        for g, grp in enumerate(GROUPS):
            on = bool(e >> g & 1)
            pe *= P_GROUP if on else 1 - P_GROUP
            if on:
                cov[list(grp)] = True
        lik = np.ones(len(X))
        for i in range(n):
            if cov[i]:
                lik *= down[:, i]
            else:
                lik *= np.where(down[:, i], P_FAIL, 1 - P_FAIL)
        p += pe * lik
    assert abs(p.sum() - 1) < 1e-9
    C = np.array([comm.control_fraction(x) for x in X]).round(6)
    CV, cidx = np.unique(C, axis=0, return_inverse=True)
    T = truth_flags(X)
    w = World(X, T, np.log(np.maximum(p, 1e-300)), cidx.astype(np.int32), CV, None, None)
    return w.with_s(0.0)


def sample_states(w: World, rng: np.random.Generator, n: int) -> np.ndarray:
    """Draw n state indices from the exact prior."""
    return rng.choice(len(w.X), size=n, p=np.exp(w.logprior))


def emit(w: World, states: np.ndarray, q: float, s: float, rng: np.random.Generator) -> np.ndarray:
    """Observations (n, 13) int8: -1 missing, 0 down, 1 up."""
    t = w.T[states]
    obs = t.astype(np.int8)
    stale = (~t) & (rng.random(t.shape) < s)
    obs[stale] = 1
    obs[rng.random(t.shape) < q] = -1
    return obs


def posterior(w: World, obs: np.ndarray, s: float) -> np.ndarray:
    """Exact posterior over the 16384 states for a batch of observations (B, 13) -> (B, 16384) float32."""
    ws = w.with_s(s)
    B = len(obs)
    ll = np.tile(ws.logprior, (B, 1)).astype(np.float64)
    for j in range(N_FLAG):
        o = obs[:, j]
        ll += (o == 1)[:, None] * ws.A[None, :, j] + (o == 0)[:, None] * ws.B[None, :, j]
    ll -= ll.max(axis=1, keepdims=True)
    p = np.exp(ll)
    p /= p.sum(axis=1, keepdims=True)
    return p.astype(np.float32)


def posterior_over_controls(w: World, post: np.ndarray) -> np.ndarray:
    """(B, 16384) -> (B, n_cv): posterior mass of every control vector (sparse aggregation)."""
    from scipy.sparse import coo_matrix

    M = coo_matrix((np.ones(len(w.cidx), np.float32), (np.arange(len(w.cidx)), w.cidx)), shape=(len(w.cidx), len(w.CV))).tocsr()
    return np.asarray((M.T @ post.T).T, dtype=np.float32)


def control_marginals(w: World, pc: np.ndarray) -> np.ndarray:
    """(B, n_cv) -> (B, 5) posterior mean commandable fraction per generator."""
    return pc @ w.CV.astype(np.float32)
