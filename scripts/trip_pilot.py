"""Step 2 sensitivity pilot on DEVELOPMENT seeds (1000+): how much do the value-of-control findings depend on the
protection-trip rule? Never touches test operating points."""
from multiprocessing import Pool

import numpy as np

from fdna import comm, spec
from fdna.grid import load_case30
from fdna.lp import Params, ScenarioLP
from fdna.opgen import nominal_rating, sample_op

G = load_case30()
RATING = nominal_rating(G, spec.RATING)
MODES = ["shed_plus_surplus", "surplus_only", "free"]


def work(opi):
    rng = np.random.default_rng(1000 + opi)
    op = sample_op(G, RATING, rng)
    sets = comm.failure_sets(4)
    C = np.array([np.ones(5), np.zeros(5)] + [comm.control_fraction(comm.state_of(sets[i]))
                                              for i in rng.choice(len(sets), 12, replace=False)])
    pairs = [(a, b) for a in range(G.n_branch) for b in range(a + 1, G.n_branch)]
    conts = [(b,) for b in range(G.n_branch)] + [pairs[i] for i in rng.choice(len(pairs), 150, replace=False)]
    out = []
    for r in conts:
        row = [len(r)]
        for mode in MODES:
            lp = ScenarioLP(G, op, r, Params(spec.PARAMS.ramp_frac, spec.PARAMS.local_mw, mode))
            ys = []
            for c in C:
                try:
                    ys.append(lp.y(c))
                except RuntimeError:
                    ys.append(np.nan)
            ys = np.array(ys)
            row += [np.isnan(ys).mean(), ys[0], ys[1], np.nanmax(ys) - np.nanmin(ys) if not np.isnan(ys).all() else np.nan]
        out.append(row)
    return out


if __name__ == "__main__":
    with Pool(30) as p:
        a = np.array([r for o in p.map(work, range(10)) for r in o])
    print("mode | N-2: unresolved-share, shed>0 at full control, none-vs-full gap>tau, sensitive(range>tau), mean(y_none-y_full)")
    for k, mode in enumerate(MODES):
        n2 = a[a[:, 0] == 2][:, 1 + 4 * k: 5 + 4 * k]
        unres, yf, yn, rg = n2.T
        ok = ~np.isnan(yf) & ~np.isnan(yn)
        print(f"{mode:18s} unresolved={np.nanmean(unres):.3f} shed>0={np.mean(yf[ok] > 1e-6):.3f} "
              f"gap>tau={np.mean((yn[ok] - yf[ok]) > spec.TAU):.3f} sensitive={np.nanmean(rg > spec.TAU):.3f} "
              f"mean_gap={np.mean(yn[ok] - yf[ok]):.4f}")
