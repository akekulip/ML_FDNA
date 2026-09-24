"""Gate E1 (screen, NO FDNA): exact Bayes oracle vs tuned tree arms on identical partial/stale observations.
Test data: inspected exploratory block 200-279 (screening only)."""
import json, sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import op_metrics

D = sys.argv[1] if len(sys.argv) > 1 else "data_v2"
import os
N_LIST = [25, 100]
CELL_IDX = os.environ.get("CELL_IDX")
NJ = int(os.environ.get("N_JOBS", 8))
K_TR, K_VA, K_TE = 20, 10, 8
N_TRIALS = 12
w = v2.build_world()
tr, va, te = (v2data.load_vtable(D, s) for s in ("train", "val", "test"))
F = {s: v2data.cont_features(v, "data/ops.npz") for s, v in (("train", tr), ("val", va), ("test", te))}
rng0 = np.random.default_rng(1)
grid = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
        for a, b, c, e in zip(rng0.choice([15, 31, 63], N_TRIALS), rng0.choice([20, 50, 100, 200], N_TRIALS),
                              rng0.choice([0.5, 0.8, 1.0], N_TRIALS), rng0.choice([0.0, 1.0, 10.0], N_TRIALS))]


def sub(vt, feats, idx):
    return {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in vt.items()}, feats[idx]


def arm_features(d, s):
    obs = d["obs"]
    A1 = np.hstack([d["Xc"], np.where(obs < 0, np.nan, obs).astype(np.float32)])
    A2 = np.hstack([d["Xc"], v2data.logic_features(obs)])
    pc, marg = v2data.oracle_features(w, obs, s)
    return {"A1_obs": A1, "A2_logic": A2, "A2b_oraclefeat": np.hstack([d["Xc"], marg])}, pc


def fit_tuned(Xtr, ytr, Xva, yva, seed):
    best = None
    for p in grid:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=seed, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
        e = float(((m.predict(Xva) - yva) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, m, p)
    return best[1], best[2]


rows, info = [], {}
t0 = time.time()
for (q, s) in ([v2.CELLS[int(CELL_IDX)]] if CELL_IDX is not None else v2.CELLS):
    dva = v2data.build(va, F["val"], w, q, s, K_VA, seed=11)
    fva, _ = arm_features(dva, s)
    dte = v2data.build(te, F["test"], w, q, s, K_TE, seed=13, only_n2=True)
    fte, pc_te = arm_features(dte, s)
    oracle = (pc_te * te["V"][dte["io"], dte["jc"]]).sum(1)
    for n in N_LIST:
        idx = np.sort(np.random.default_rng(1000 * n).choice(len(tr["op_ids"]), n, replace=False))
        vsub, fsub = sub(tr, F["train"], idx)
        dtr = v2data.build(vsub, fsub, w, q, s, K_TR, seed=100 + n)
        ftr, _ = arm_features(dtr, s)
        preds = {"oracle": oracle}
        for name in ftr:
            m, p = fit_tuned(ftr[name], dtr["y"], fva[name], dva["y"], seed=0)
            preds[name] = m.predict(fte[name])
            info[f"q{q}_s{s}_n{n}_{name}"] = p
        for name, pr in preds.items():
            for o in np.unique(dte["op"]):
                mm = dte["op"] == o
                rows.append(dict(q=q, s=s, n=n, arm=name, op=int(o), **op_metrics(dte["y"][mm], pr[mm], dte["key"][mm])))
        print(f"cell q={q} s={s} n={n} done {time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows)
sfx = f"_{CELL_IDX}" if CELL_IDX is not None else ""
r.to_parquet(f"{D}/e1_results{sfx}.parquet")
json.dump(info, open(f"{D}/e1_params{sfx}.json", "w"), default=float)
if CELL_IDX is not None:
    sys.exit(0)
rng = np.random.default_rng(3)
print("\nE1: mean per-op R-precision (test ops 200-279, N-2 outages, K=8 draws) and oracle-minus-best-tree gap [95% cluster CI]")
print("prevalence:", r.groupby(["q", "s"]).prev.mean().round(3).to_dict())
tab = r.groupby(["q", "s", "n", "arm"]).rprec.mean().unstack("arm").round(3)
print(tab.to_string())
print()
for (q, s, n), g in r.groupby(["q", "s", "n"]):
    pv = g.pivot(index="op", columns="arm", values="rprec")
    best = pv[["A1_obs", "A2_logic"]].max(axis=1)
    dlt = (pv["oracle"] - best).values
    idx = rng.integers(0, len(dlt), (10000, len(dlt)))
    lo, hi = np.percentile(dlt[idx].mean(1), [2.5, 97.5])
    ref = (pv["A2b_oraclefeat"] - best).values.mean()
    print(f"q={q} s={s} n={n}: oracle - best(A1,A2) = {dlt.mean():+.4f} [{lo:+.4f},{hi:+.4f}] | oracle-feature GBM - best = {ref:+.4f}")
