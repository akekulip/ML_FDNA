"""v2 dataset assembly from the LP value table, plus the tree-arm feature sets."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import comm, v2
from .baseline_data import _phys_chunk, N_PHYS
from .evalutil import scenario_uniform

SHARES = comm.SHARES
PARENTS = comm.PARENT_GW
DATA_V1 = Path("data")


def load_vtable(dirpath: str, split: str) -> dict:
    z = np.load(Path(dirpath) / f"vtable_{split}.npz")
    return {k: z[k] for k in z.files}


def op_arrays(ops_npz: str):
    z = np.load(ops_npz)
    idx = {int(k): i for i, k in enumerate(z["op_id"])}
    return z["demand"], z["p0"], idx


def cont_features(vt: dict, ops_npz: str) -> np.ndarray:
    """(n_op, n_cont, D) electrical + physics features per (operating point, outage)."""
    demand, p0, idx = op_arrays(ops_npz)
    n_op, n_cont = vt["conts"].shape[:2]
    out = np.zeros((n_op, n_cont, 30 + 6 + 41 + 1 + N_PHYS), np.float32)
    for i, op in enumerate(vt["op_ids"]):
        oi = idx[int(op)]
        conts = vt["conts"][i]
        keys = np.array([(oi, int(a), int(b)) for a, b in conts])
        phys = _phys_chunk((demand, p0, keys))
        mh = np.zeros((n_cont, 41), np.float32)
        for j, (a, b) in enumerate(conts):
            if a >= 0: mh[j, a] = 1
            if b >= 0: mh[j, b] = 1
        el = np.hstack([np.tile(demand[oi], (n_cont, 1)), np.tile(p0[oi], (n_cont, 1)), mh, mh.sum(1, keepdims=True), phys])
        out[i] = el
    return out


def logic_features(obs: np.ndarray) -> np.ndarray:
    """A2: deterministic structure-derived bounds from observations only (no probabilities).
    per generator: lower bound (units observed up, sound only if no staleness), upper bound (units not ruled out),
    plus counts of missing / observed-down flags."""
    B = len(obs)
    lo = np.zeros((B, 5)); hi = np.zeros((B, 5))
    for u in range(comm.N_UNIT):
        g = u // comm.UNITS_PER_GEN
        flag = obs[:, u]
        hb = np.stack([obs[:, comm.N_UNIT + p] for p in PARENTS[g]], 1)  # parent heartbeats
        ruled_out = (flag == 0) | (hb == 0).all(1)
        lo[:, g] += SHARES[u % comm.UNITS_PER_GEN] * (flag == 1)
        hi[:, g] += SHARES[u % comm.UNITS_PER_GEN] * (~ruled_out)
    nmiss = (obs < 0).sum(1, keepdims=True)
    ndown = (obs == 0).sum(1, keepdims=True)
    return np.hstack([lo, hi, nmiss, ndown]).astype(np.float32)


def oracle_features(w: v2.World, obs: np.ndarray, s: float, chunk: int = 2000):
    """Exact posterior: returns (pc (B, n_cv) control posterior, marginal features (B, 10))."""
    pcs = []
    for a in range(0, len(obs), chunk):
        pcs.append(v2.posterior_over_controls(w, v2.posterior(w, obs[a:a + chunk], s)))
    pc = np.vstack(pcs)
    marg = pc @ w.CV.astype(np.float32)
    p_zero = pc @ (w.CV == 0).astype(np.float32)
    return pc, np.hstack([marg, p_zero]).astype(np.float32)


def build(vt: dict, feats: np.ndarray, w: v2.World, q: float, s: float, k: int, seed: int, only_n2: bool = False, coverage=None):
    """Draw k hidden states + observations per (op, outage); labels from the value table."""
    rng = np.random.default_rng(seed)
    n_op, n_cont = vt["conts"].shape[:2]
    rows_op, rows_cont, states = [], [], []
    for i in range(n_op):
        for j in range(n_cont):
            if only_n2 and vt["conts"][i, j, 1] < 0:
                continue
            rows_op.append(np.full(k, i)); rows_cont.append(np.full(k, j)); states.append(v2.sample_states(w, rng, k))
    io, jc, st = np.concatenate(rows_op), np.concatenate(rows_cont), np.concatenate(states)
    obs = v2.emit(w, st, q, s, rng, coverage)
    y = vt["V"][io, jc, w.cidx[st]]
    draw = np.tile(np.arange(k), len(io) // k)
    key = scenario_uniform(vt["op_ids"][io], vt["conts"][io, jc, 0], vt["conts"][io, jc, 1] + 100 * draw, draw + 7, salt=3)
    return dict(io=io, jc=jc, st=st, obs=obs, y=y.astype(np.float64), Xc=feats[io, jc], op=vt["op_ids"][io], key=key)
