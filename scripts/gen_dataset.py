"""Generate reference labels. Usage: gen_dataset.py [out_dir] [n_train n_val n_test]"""
from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from fdna import comm, spec
from fdna.grid import load_case30
from fdna.lp import ScenarioLP
from fdna.opgen import nominal_rating, sample_op

G = load_case30()
RATING = nominal_rating(G, spec.RATING)
PAIRS = [(a, b) for a in range(G.n_branch) for b in range(a + 1, G.n_branch)]
CTRL = np.array([comm.control_fraction(comm.state_of(f)) for f in spec.FAILURE_SETS])


def strat_test_fs(rng, per_size):
    ids = spec.fs_ids("test")
    sizes = np.array([len(spec.FAILURE_SETS[i]) for i in ids])
    out = []
    for sz, k in per_size.items():
        pool = ids[sizes == sz]
        out.extend(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.array(out)


def plan(split, rng):
    """-> list of (contingency tuple, fs id array, cell tag)."""
    n1 = [(b,) for b in range(G.n_branch)]
    tr, va = spec.fs_ids("train"), spec.fs_ids("val")
    if split == "train":
        return [((), tr, "n0_seen")] + [(c, tr, "n1_seen") for c in n1]
    if split == "val":
        n2 = [PAIRS[i] for i in rng.choice(len(PAIRS), 150, replace=False)]
        return [(c, va, "n1_unseen") for c in n1] + [(c, va, "n2_unseen") for c in n2]
    te = strat_test_fs(rng, {2: 20, 3: 15, 4: 15})
    sub_tr = rng.choice(tr, 20, replace=False)
    return (
        [(c, tr, "n1_seen") for c in n1]
        + [(c, te, "n1_unseen") for c in n1]
        + [(c, sub_tr, "n2_seen") for c in PAIRS]
        + [(c, te, "n2_unseen") for c in PAIRS]
    )


def job(args):
    split, op_id = args
    rng = np.random.default_rng(op_id)
    op = sample_op(G, RATING, rng)
    rows = []
    cache_lp = {}
    for cont, fsids, cell in plan(split, np.random.default_rng(10_000 + op_id)):
        lp = ScenarioLP(G, op, cont, spec.PARAMS)
        cache = {}
        for fi in fsids:
            key = tuple(np.round(CTRL[fi], 6))
            if key not in cache:
                cache[key] = lp.y(CTRL[fi])
            rows.append((op_id, split, cell, cont[0] if cont else -1, cont[1] if len(cont) > 1 else -1,
                         int(fi), cache[key], lp.struct_mw / lp.total))
    df = pd.DataFrame(rows, columns=["op", "split", "cell", "b1", "b2", "fs", "y", "y_struct"])
    return df, (op_id, op.demand, op.p0)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    n_tr, n_va, n_te = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (100, 20, 80)
    out.mkdir(exist_ok=True)
    jobs = [("train", i) for i in range(0, n_tr)] + [("val", i) for i in range(100, 100 + n_va)] \
        + [("test", i) for i in range(200, 200 + n_te)]
    dfs, ops = [], []
    with Pool(30) as pool:
        for k, (df, op) in enumerate(pool.imap_unordered(job, jobs)):
            dfs.append(df)
            ops.append(op)
            if k % 20 == 0:
                print(k, "/", len(jobs), flush=True)
    pd.concat(dfs, ignore_index=True).to_parquet(out / "labels.parquet")
    np.savez(out / "ops.npz", op_id=np.array([o[0] for o in ops]), demand=np.array([o[1] for o in ops]),
             p0=np.array([o[2] for o in ops]))
    (out / "spec_hash.txt").write_text(spec.spec_hash())
    print("spec", spec.spec_hash(), "rows", sum(len(d) for d in dfs))
