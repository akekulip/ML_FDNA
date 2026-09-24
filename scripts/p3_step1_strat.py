"""Phase 3 Step 1b: missed-severe mass by disjoint outage class for the k=0 arm (tuned GBM, exact control, trained on N-1 rows of all 100 train ops).
Exploratory (test table ops 200-279 already inspected). registry/phase3_step1.yaml defines classes/error."""
import json, sys
import lightgbm as lgb
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fdna import spec, v2, v2data
from fdna.dataset import G, PAIRS
from fdna.evalutil import rank_order

sys.path.insert(0, "scripts")
w = v2.build_world()
tr, va, te = (v2data.load_vtable("data_v2", x) for x in ("train", "val", "test"))
F = {"train": v2data.cont_features(tr, "data/ops.npz"), "val": v2data.cont_features(va, "data/ops.npz"), "test": v2data.cont_features(te, "data/ops.npz")}
feat = lambda d, Fs: np.hstack([Fs[d["io"], d["jc"]], w.CV[w.cidx[d["st"]]], w.CV[w.cidx[d["st"]]].sum(1, keepdims=True)]).astype(np.float32)
dtr = v2data.build(tr, F["train"], w, 0.5, 0.0, 20, seed=100); dva = v2data.build(va, F["val"], w, 0.5, 0.0, 10, seed=11)
dte = v2data.build(te, F["test"], w, 0.5, 0.0, 8, seed=13, only_n2=True)
Xtr, Xva, Xte = feat(dtr, F["train"]), feat(dva, F["val"]), feat(dte, F["test"])
rng0 = np.random.default_rng(1); best = None
for a, b, c, e in zip(rng0.choice([15, 31, 63], 12), rng0.choice([20, 50, 100, 200], 12), rng0.choice([0.5, 0.8, 1.0], 12), rng0.choice([0.0, 1.0, 10.0], 12)):
    p = dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
    m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=8, bagging_fraction=0.8, bagging_freq=1, **p)
    m.fit(Xtr, dtr["y"], eval_set=[(Xva, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
    er = float(((m.predict(Xva) - dva["y"]) ** 2).mean())
    if best is None or er < best[0]: best = (er, p)
m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=8, bagging_fraction=0.8, bagging_freq=1, **best[1])
m.fit(Xtr, dtr["y"], eval_set=[(Xva, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
pred = m.predict(Xte)


def klass(rem):
    k = np.array([i for i in range(G.n_branch) if i not in rem])
    n, lab = connected_components(coo_matrix((np.ones(len(k)), (G.frm[k], G.to[k])), shape=(G.n_bus,) * 2), directed=False)
    if n == 1: return "connected"
    hg = np.array([(lab[G.gen_bus] == i).any() for i in range(n)])
    return "gen_less_only" if hg.sum() == 1 else ("all_gen" if (~hg).sum() == 0 else "mixed")


cls = {p: klass(p) for p in PAIRS}
pair = [tuple(int(x) for x in te["conts"][i, j]) for i, j in zip(dte["io"], dte["jc"])]
rc = np.array([cls[p] for p in pair]); y = dte["y"]; sev = y > spec.SEVERE
miss = np.zeros(len(y), bool); tot_rp = []
for o in np.unique(dte["op"]):
    mm = np.flatnonzero(dte["op"] == o); k = int(sev[mm].sum())
    if k == 0: continue
    top = np.zeros(len(mm), bool); top[rank_order(pred[mm], dte["key"][mm])[:k]] = True
    miss[mm] = sev[mm] & ~top; tot_rp.append(sev[mm][top].mean())
out = {"rprec_pooled_mean_over_ops": float(np.mean(tot_rp)), "n_rows": int(len(y)), "n_severe": int(sev.sum()), "n_missed": int(miss.sum()), "tuned": best[1]}
for c in ("connected", "gen_less_only", "all_gen", "mixed"):
    mc = rc == c
    out[c] = {"row_share": float(mc.mean()), "severe_share": float((sev & mc).sum() / max(sev.sum(), 1)), "missed_share": float((miss & mc).sum() / max(miss.sum(), 1)),
              "miss_rate_given_severe": float((miss & mc).sum() / max((sev & mc).sum(), 1)), "mae": float(np.abs(pred - y)[mc].mean())}
isl = rc != "connected"
out["islanding_missed_share"] = float((miss & isl).sum() / max(miss.sum(), 1)); out["G1_pass_ge_50pct"] = bool(out["islanding_missed_share"] >= 0.5)
json.dump(out, open("results/phase3/step1_strat.json", "w"), indent=1); print(json.dumps(out, indent=1))
