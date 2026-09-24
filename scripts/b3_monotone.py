"""Branch 3 screen: monotone-in-control value model vs monotone-constrained GBM (v1 benchmark, exploratory test 200-279)."""
import sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data
from fdna.baseline_data import evaluate, strata
from fdna.nn.layers import MLP, MonotoneValue
from fdna.nn.train import fit, predict

DATA = "data"
NS, REPS = [5, 10, 25, 100], 3
d = baseline_data.load(DATA)
X = d.F["elec+phys+ctrl"]
NX = 123                      # electrical + physics block; then C(5) U(10) T(1)
C = X[:, 123:128]
mono = [0] * NX + [-1] * 5 + [-1] * 10 + [-1]      # shed non-increasing in every control feature
mu, sd = X[d.tr].mean(0), X[d.tr].std(0) + 1e-6
Z = ((X - mu) / sd).astype(np.float32)
S = {k: v for k, v in strata(d).items() if k in ("n2_unseen|novel", "n2_unseen|familiar", "n2_unseen")}
ops_tr = np.unique(d.lab.op.values[d.tr])
GBM = dict(n_estimators=400, num_leaves=15, min_child_samples=100, colsample_bytree=0.5, learning_rate=0.1,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=8)
rows, t0 = [], time.time()
for n in NS:
    for rep in range(REPS):
        sub = ops_tr if n == 100 else np.random.default_rng(1000 * n + rep).choice(ops_tr, n, replace=False)
        mtr = d.tr & np.isin(d.lab.op.values, sub)
        preds = {}
        for name, mc in (("gbm_free", None), ("gbm_monotone", mono)):
            m = lgb.LGBMRegressor(random_state=rep, monotone_constraints=mc, **GBM)
            m.fit(X[mtr], d.y[mtr], eval_set=[(X[d.va], d.y[d.va])], callbacks=[lgb.early_stopping(30, verbose=False)])
            preds[name] = m.predict(X[d.te])
        mlp, _ = fit(MLP(X.shape[1]), (Z[mtr],), d.y[mtr], (Z[d.va],), d.y[d.va], seed=rep)
        preds["mlp_free"] = predict(mlp, (Z[d.te],))
        mv, _ = fit(MonotoneValue(NX, 5), (Z[mtr][:, :NX], C[mtr]), d.y[mtr], (Z[d.va][:, :NX], C[d.va]), d.y[d.va], seed=rep)
        preds["nn_monotone"] = predict(mv, (Z[d.te][:, :NX], C[d.te]))
        for name, p in preds.items():
            full = np.full(len(d.y), np.nan); full[d.te] = p
            for r in evaluate(d, name, full, rep, S):
                r.update(n=n, rep=rep); rows.append(r)
        print("n", n, "rep", rep, f"{time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows); r.to_parquet(f"{DATA}/b3_results.parquet")
rng = np.random.default_rng(4)
print("\nBranch 3 screen: mean R-precision by model and n (rows: model, stratum n2_unseen|novel / familiar)")
for st in ("n2_unseen|novel", "n2_unseen|familiar"):
    print("--", st); print(r[r.stratum == st].groupby(["model", "n"]).rprec.mean().unstack("n").round(3).to_string())
print("\npaired differences (per-op, reps averaged) with 95% cluster CI")
for a, b in (("nn_monotone", "gbm_monotone"), ("gbm_monotone", "gbm_free"), ("nn_monotone", "mlp_free")):
    for st in ("n2_unseen|novel", "n2_unseen|familiar"):
        out = []
        for n in NS:
            g = r[(r.stratum == st) & (r.n == n)]
            pa = g[g.model == a].groupby("op").rprec.mean(); pb = g[g.model == b].groupby("op").rprec.mean()
            dl = (pa - pb).values; ix = rng.integers(0, len(dl), (5000, len(dl)))
            lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5]); out.append(f"n={n}: {dl.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
        print(f"{a} - {b} | {st}:", "  ".join(out))
