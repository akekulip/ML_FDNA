"""Stress calibration on a development pilot (seed 1000+). Never touches test operating points."""
from __future__ import annotations

import itertools
import sys
from multiprocessing import Pool

import numpy as np

from fdna import comm
from fdna.grid import load_case30
from fdna.lp import Params, ScenarioLP
from fdna.opgen import RatingSpec, nominal_rating, sample_op

TAU = 0.005
N_OP, N_N2, N_CTRL = 10, 150, 12
G = load_case30()


def control_vectors(rng, k):
    sets = comm.failure_sets(4)
    idx = rng.choice(len(sets), size=k, replace=False)
    cs = [comm.control_fraction(comm.state_of(sets[i])) for i in idx]
    return np.array([np.ones(5), np.zeros(5), *cs])


def work(args):
    kappa, ramp, local, floor, opi = args
    rating = nominal_rating(G, RatingSpec(kappa, floor))
    rng = np.random.default_rng(1000 + opi)
    op = sample_op(G, rating, rng)
    prm = Params(ramp, local)
    C = control_vectors(rng, N_CTRL)
    conts = [(b,) for b in range(G.n_branch)]
    pairs = [(a, b) for a in range(G.n_branch) for b in range(a + 1, G.n_branch)]
    conts += [pairs[i] for i in rng.choice(len(pairs), size=N_N2, replace=False)]
    out = []
    for r in conts:
        lp = ScenarioLP(G, op, r, prm)
        ys = np.array([lp.y(c) for c in C])
        out.append((len(r), ys[0], ys[1], ys.max() - ys.min(), len(np.unique(np.round(ys, 6))), lp.struct_mw / lp.total, (np.abs(ys[2:] - ys[0]) > TAU).mean()))
    return (kappa, ramp, local, floor), out


if __name__ == "__main__":
    grid = list(itertools.product([1.6, 2.0, 2.5], [0.15, 0.3, 0.6], [20.0, 40.0], [20.0]))
    jobs = [(*p, o) for p in grid for o in range(N_OP)]
    res: dict = {}
    with Pool(30) as pool:
        for key, out in pool.imap_unordered(work, jobs):
            res.setdefault(key, []).extend(out)
    print("kappa ramp local | N2: shed>0 sens>tau  none-full>tau  >=3vals | N1: shed>0 sens | struct>0 | N2 frac(state) differing")
    for key in sorted(res):
        a = np.array(res[key])
        n2, n1 = a[a[:, 0] == 2], a[a[:, 0] == 1]
        f = lambda m: (m[:, 1] > 1e-6).mean()
        s = lambda m: (m[:, 3] > TAU).mean()
        print(key[:3], f"| {f(n2):.2f} {s(n2):.2f} {((n2[:,2]-n2[:,1])>TAU).mean():.2f} {(n2[:,4]>=3).mean():.2f}"
              f" | {f(n1):.2f} {s(n1):.2f} | {(a[:,5]>0).mean():.2f} | {n2[:,6].mean():.2f}")
