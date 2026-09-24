"""Few-shot N-2 label efficiency (registered before coding). Exact control known; ops 0-24; exploratory block 200-279 as test."""
import os, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import rprec

D, FS = "data_v2", "data_v2_fs"
KS = [0, 10, 30, 100]; REPS = 3; NJ = int(os.environ.get("N_JOBS", 8)); K_TR, K_VA, K_TE = 20, 10, 8
w = v2.build_world()
tr, va, te = (v2data.load_vtable(D, x) for x in ("train", "val", "test"))
fs = v2data.load_vtable(FS, "train2")
F = {"train": v2data.cont_features(tr, "data/ops.npz"), "fs": v2data.cont_features(fs, "data/ops.npz"),
     "val": v2data.cont_features(va, "data/ops.npz"), "test": v2data.cont_features(te, "data/ops.npz")}
idx = np.arange(25)
trs = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
assert (fs["op_ids"] == trs["op_ids"]).all()
rng0 = np.random.default_rng(1)
GB = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
      for a, b, c, e in zip(rng0.choice([15, 31, 63], 12), rng0.choice([20, 50, 100, 200], 12), rng0.choice([0.5, 0.8, 1.0], 12), rng0.choice([0.0, 1.0, 10.0], 12))]
feat = lambda d_, Fs: np.hstack([Fs[d_["io"], d_["jc"]], w.CV[w.cidx[d_["st"]]], w.CV[w.cidx[d_["st"]]].sum(1, keepdims=True)]).astype(np.float32)
dva = v2data.build(va, F["val"], w, 0.5, 0.0, K_VA, seed=11); Xva = feat(dva, F["val"])
dte = v2data.build(te, F["test"], w, 0.5, 0.0, K_TE, seed=13, only_n2=True); Xte = feat(dte, F["test"])
d1 = v2data.build(trs, F["train"][idx], w, 0.5, 0.0, K_TR, seed=100); X1 = feat(d1, F["train"][idx])
n2_cols = np.flatnonzero(fs["conts"][0, :, 1] >= 0)               # positions of N-2 pairs in the fs table (same layout for all ops)
rows, t0, tuned = [], time.time(), {}
for k in KS:
    for rep in range(REPS):
        if k == 0:
            Xtr, ytr = X1, d1["y"]
        else:
            sel = np.sort(np.random.default_rng(7 + rep).choice(n2_cols, k, replace=False))
            sub = {kk: (v[:, sel] if kk in ("conts", "V") else v) for kk, v in fs.items()}
            d2 = v2data.build(sub, F["fs"][:, sel], w, 0.5, 0.0, K_TR, seed=300 + rep)
            Xtr, ytr = np.vstack([X1, feat(d2, F["fs"][:, sel])]), np.concatenate([d1["y"], d2["y"]])
        if k not in tuned:
            best = None
            for p in GB:
                m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
                m.fit(Xtr, ytr, eval_set=[(Xva, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
                e = float(((m.predict(Xva) - dva["y"]) ** 2).mean())
                if best is None or e < best[0]:
                    best = (e, p)
            tuned[k] = best[1]
        m = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **tuned[k])
        m.fit(Xtr, ytr, eval_set=[(Xva, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
        pred = m.predict(Xte)
        for o in np.unique(dte["op"]):
            mm = dte["op"] == o
            for thr in (0.01, 0.02):
                rows.append(dict(k=k, rep=rep, op=int(o), thr=thr, rprec=rprec(dte["y"][mm], pred[mm], dte["key"][mm], thr)))
        print(f"k={k} rep={rep} {time.time()-t0:.0f}s rows={len(ytr)}", flush=True)
r = pd.DataFrame(rows); r.to_parquet(f"{D}/fewshot.parquet")
t = r.groupby(["thr", "k"]).rprec.mean().unstack("thr").round(3)
r0 = t.loc[0]
print("\nunseen-N-2 R-precision vs k labelled N-2 pairs per training operating point (25 ops; exact control known)")
t["gap_closed@1%"] = ((t[0.01] - r0[0.01]) / (1 - r0[0.01])).round(2); t["gap_closed@2%"] = ((t[0.02] - r0[0.02]) / (1 - r0[0.02])).round(2)
print(t.to_string())
