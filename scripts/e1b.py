"""Gate E1b (screen, NO FDNA): inference headroom = R(GBM given the EXACT control vector) - R(best tuned tree on observations).
env: VARIANT in {v2, v2b, v2c}, CELL_IDX (index into the variant's (q,s) grid), N_JOBS. n=100 training operating points only."""
import json, os, sys
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import op_metrics

D = "data_v2"
VARIANT = os.environ.get("VARIANT", "v2"); CELL_IDX = int(os.environ["CELL_IDX"]); NJ = int(os.environ.get("N_JOBS", 6))
GRID = {"v2": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)], "v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)],
        "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
n, K_TR, K_VA, K_TE, N_TRIALS = 100, 20, 10, 8, 12
w = v2.build_world()
tr, va, te = (v2data.load_vtable(D, x) for x in ("train", "val", "test"))
F = {x: v2data.cont_features(v, "data/ops.npz") for x, v in (("train", tr), ("val", va), ("test", te))}
rng0 = np.random.default_rng(1)
grid = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
        for a, b, c, e in zip(rng0.choice([15, 31, 63], N_TRIALS), rng0.choice([20, 50, 100, 200], N_TRIALS),
                              rng0.choice([0.5, 0.8, 1.0], N_TRIALS), rng0.choice([0.0, 1.0, 10.0], N_TRIALS))]


def fit_tuned(Xtr, ytr, Xva, yva):
    best = None
    for p in grid:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
        e = float(((m.predict(Xva) - yva) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, m)
    return best[1]


def feats(d, kind):
    c_true = w.CV[w.cidx[d["st"]]]
    if kind == "exactc":
        return np.hstack([d["Xc"], c_true, c_true.sum(1, keepdims=True)]).astype(np.float32)
    if kind == "A1_obs":
        return np.hstack([d["Xc"], np.where(d["obs"] < 0, np.nan, d["obs"]).astype(np.float32)])
    return np.hstack([d["Xc"], v2data.logic_features(d["obs"])])


arms = ["exactc"] if VARIANT == "v2" else ["exactc", "A1_obs", "A2_logic"]
dva = v2data.build(va, F["val"], w, q, s, K_VA, seed=11, coverage=COV)
dte = v2data.build(te, F["test"], w, q, s, K_TE, seed=13, only_n2=True, coverage=COV)
idx = np.sort(np.random.default_rng(1000 * n).choice(len(tr["op_ids"]), n, replace=False))
vsub = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
dtr = v2data.build(vsub, F["train"][idx], w, q, s, K_TR, seed=100 + n, coverage=COV)
pc_te = v2data.oracle_features(w, dte["obs"], s)[0]
preds = {"oracle": (pc_te * te["V"][dte["io"], dte["jc"]]).sum(1)}
for a in arms:
    preds[a] = fit_tuned(feats(dtr, a), dtr["y"], feats(dva, a), dva["y"]).predict(feats(dte, a))
rows = [dict(variant=VARIANT, q=q, s=s, n=n, arm=a, op=int(o), **op_metrics(dte["y"][dte["op"] == o], p[dte["op"] == o], dte["key"][dte["op"] == o]))
        for a, p in preds.items() for o in np.unique(dte["op"])]
pd.DataFrame(rows).to_parquet(f"{D}/e1b_{VARIANT}_{CELL_IDX}.parquet")
print("done", VARIANT, q, s)
