"""EXPLORATORY (idle-core task): is the ~0.02 raw-flag edge over the explicit control vector real when both use
IDENTICAL hyper-parameters? 5 hyper-parameter configs x 5 seeds x n in {25,100}; exploratory test block 200-279."""
import time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data
from fdna.baseline_data import evaluate, strata

d = baseline_data.load("data")
F = {"explicit": d.F["elec+phys+ctrl"], "raw": d.F["elec+phys+raw_comm"], "explicit_C_only": None}
X = d.F["elec+phys+ctrl"]
F["explicit_C_only"] = np.hstack([X[:, :123], X[:, 123:128]])          # 5 commandable fractions only
S = {k: v for k, v in strata(d).items() if k in ("n2_unseen|novel", "n2_unseen|familiar")}
ops_tr = np.unique(d.lab.op.values[d.tr])
cfgs = [dict(num_leaves=15, min_child_samples=100, colsample_bytree=0.5), dict(num_leaves=31, min_child_samples=50, colsample_bytree=0.8),
        dict(num_leaves=63, min_child_samples=20, colsample_bytree=1.0), dict(num_leaves=15, min_child_samples=200, colsample_bytree=0.5),
        dict(num_leaves=31, min_child_samples=100, colsample_bytree=0.5)]
rows, t0 = [], time.time()
for n in (25, 100):
    for ci, cfg in enumerate(cfgs):
        for seed in range(5):
            sub = ops_tr if n == 100 else np.random.default_rng(1000 * n + seed).choice(ops_tr, n, replace=False)
            mtr = d.tr & np.isin(d.lab.op.values, sub)
            for name, X_ in F.items():
                m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.1, reg_lambda=1.0, bagging_fraction=0.8, bagging_freq=1,
                                      random_state=seed, verbose=-1, n_jobs=8, **cfg)
                m.fit(X_[mtr], d.y[mtr], eval_set=[(X_[d.va], d.y[d.va])], callbacks=[lgb.early_stopping(30, verbose=False)])
                full = np.full(len(d.y), np.nan); full[d.te] = m.predict(X_[d.te])
                for r in evaluate(d, name, full, seed, S):
                    r.update(n=n, cfg=ci); rows.append(r)
        print("n", n, "cfg", ci, f"{time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows); r.to_parquet("data/raw_edge.parquet")
rng = np.random.default_rng(9)
print("\nraw - explicit R-precision differences (identical hyper-parameters), per-op paired, cluster 95% CI")
for st in ("n2_unseen|novel", "n2_unseen|familiar"):
    for n in (25, 100):
        out = []
        for ci in range(len(cfgs)):
            g = r[(r.stratum == st) & (r.n == n) & (r.cfg == ci)]
            a = g[g.model == "raw"].groupby("op").rprec.mean(); b = g[g.model == "explicit"].groupby("op").rprec.mean()
            dl = (a - b).values; ix = rng.integers(0, len(dl), (4000, len(dl)))
            lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5]); out.append(f"cfg{ci}: {dl.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
        print(st, f"n={n}", " ".join(out))
print("\nexplicit_C_only vs explicit (does adding U/T help?):")
for st in ("n2_unseen|novel",):
    for n in (25, 100):
        g = r[(r.stratum == st) & (r.n == n)]
        print(st, n, g.groupby("model").rprec.mean().round(3).to_dict())
