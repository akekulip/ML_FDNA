"""Reference-label generation (used by scripts/gen_dataset.py and tests)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import comm, spec
from .grid import load_case30
from .lp import ScenarioLP
from .opgen import nominal_rating, sample_op

G = load_case30()
RATING = nominal_rating(G, spec.RATING)
PAIRS = [(a, b) for a in range(G.n_branch) for b in range(a + 1, G.n_branch)]
CTRL = np.array([comm.control_fraction(comm.state_of(f)) for f in spec.FAILURE_SETS])
COLS = ["op", "split", "cell", "b1", "b2", "fs", "y", "y_struct"]


def strat_test_fs(rng, per_size):
    ids = spec.fs_ids("test")
    sizes = np.array([len(spec.FAILURE_SETS[i]) for i in ids])
    out = []
    for sz, k in per_size.items():
        pool = ids[sizes == sz]
        out.extend(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.array(out)


def plan(split, rng):
    """-> list of (contingency tuple, failure-set ids, cell tag)."""
    n1 = [(b,) for b in range(G.n_branch)]
    tr, va = spec.fs_ids("train"), spec.fs_ids("val")
    if split == "train":
        return [((), tr, "n0_seen")] + [(c, tr, "n1_seen") for c in n1]
    if split == "val":
        n2 = [PAIRS[i] for i in rng.choice(len(PAIRS), 150, replace=False)]
        return [(c, va, "n1_unseen") for c in n1] + [(c, va, "n2_unseen") for c in n2]
    te = strat_test_fs(rng, {2: 20, 3: 15, 4: 15})
    sub_tr = rng.choice(tr, 20, replace=False)
    return ([(c, tr, "n1_seen") for c in n1] + [(c, te, "n1_unseen") for c in n1]
            + [(c, sub_tr, "n2_seen") for c in PAIRS] + [(c, te, "n2_unseen") for c in PAIRS])


def job(args):
    """args = (split, op_id) -> (labels DataFrame, (op_id, demand, p0)). Deterministic in op_id."""
    split, op_id = args
    op = sample_op(G, RATING, np.random.default_rng(op_id))
    rows = []
    for cont, fsids, cell in plan(split, np.random.default_rng(10_000 + op_id)):
        lp = ScenarioLP(G, op, cont, spec.PARAMS)
        cache = {}
        for fi in fsids:
            key = tuple(np.round(CTRL[fi], 6))
            if key not in cache:
                cache[key] = lp.y(CTRL[fi])
            rows.append((op_id, split, cell, cont[0] if cont else -1, cont[1] if len(cont) > 1 else -1,
                         int(fi), cache[key], lp.struct_mw / lp.total))
    return pd.DataFrame(rows, columns=COLS), (op_id, op.demand, op.p0)


def assemble(results):
    """Order-independent assembly: sort labels by scenario id and operating points by id."""
    dfs = [r[0] for r in results]
    ops = sorted((r[1] for r in results), key=lambda o: o[0])
    lab = pd.concat(dfs, ignore_index=True).sort_values(["op", "b1", "b2", "fs"]).reset_index(drop=True)
    assert not lab.duplicated(["op", "b1", "b2", "fs"]).any(), "scenario ids must be unique"
    return lab, dict(op_id=np.array([o[0] for o in ops]), demand=np.array([o[1] for o in ops]),
                     p0=np.array([o[2] for o in ops]))
