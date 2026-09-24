"""Value ladder (registered before coding): exact control vector known; is the value-learning gap a feature artifact?
Exploratory block 200-279; trained on N-0/N-1 outages, tested on N-1 (in-distribution) and unseen N-2. env: N_LIST, REPS, N_JOBS."""
import os, time
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fdna import spec, v2, v2data
from fdna.dataset import G, RATING
from fdna.evalutil import rprec
from fdna.features import post_outage_flows

D = "data_v2"
NS = [int(x) for x in os.environ.get("N_LIST", "25,100").split(",")]; REPS = int(os.environ.get("REPS", 3)); NJ = int(os.environ.get("N_JOBS", 8))
THR = [0.005, 0.01, 0.02, 0.05]
K_TR, K_VA, K_TE = 20, 10, 8
w = v2.build_world()
tr, va, te = (v2data.load_vtable(D, x) for x in ("train", "val", "test"))
ops = np.load("data/ops.npz"); oidx = {int(k): i for i, k in enumerate(ops["op_id"])}
F0 = {k: v2data.cont_features(v, "data/ops.npz") for k, v in (("train", tr), ("val", va), ("test", te))}
XR = G.x


def r1_block(vt):
    """(n_op, n_cont, 3 stats x 3 aggs) symmetric aggregates over the removed branches + n_out."""
    n_op, n_cont = vt["conts"].shape[:2]
    out = np.zeros((n_op, n_cont, 10), np.float32)
    for i, op in enumerate(vt["op_ids"]):
        oi = oidx[int(op)]
        r0, _ = post_outage_flows(G, ops["demand"][oi], ops["p0"][oi], (), RATING)
        for j, (a, b) in enumerate(vt["conts"][i]):
            ids = [int(x) for x in (a, b) if x >= 0]
            out[i, j, 0] = len(ids)
            if ids:
                for k, arr in enumerate((r0, RATING, XR)):
                    v = arr[ids]
                    out[i, j, 1 + 3 * k: 4 + 3 * k] = [v.sum(), v.max(), v.min()]
    return out


def island_struct(vt):
    """Per (op, cont): island id per bus (for R2)."""
    res = {}
    for i, op in enumerate(vt["op_ids"]):
        for j, (a, b) in enumerate(vt["conts"][i]):
            alive = np.ones(G.n_branch, bool)
            for x in (a, b):
                if x >= 0:
                    alive[int(x)] = False
            k = np.flatnonzero(alive)
            adj = coo_matrix((np.ones(len(k)), (G.frm[k], G.to[k])), shape=(G.n_bus, G.n_bus))
            res[(i, j)] = connected_components(adj, directed=False)[1]
    return res


def r2_rows(vt, isl, d, c):
    """Island balance / effective headroom features per row from the exact control vector c."""
    out = np.zeros((len(d["io"]), 7), np.float32)
    P = spec.PARAMS
    for r, (i, j) in enumerate(zip(d["io"], d["jc"])):
        oi = oidx[int(vt["op_ids"][i])]
        demand, p0 = ops["demand"][oi], ops["p0"][oi]
        lab = isl[(i, j)]
        gen_isl = lab[G.gen_bus]
        n_isl = lab.max() + 1
        up = np.r_[min(P.local_mw, G.gen_pmax[0] - p0[0]), np.minimum(P.ramp_frac * G.gen_pmax[1:] * c[r], G.gen_pmax[1:] - p0[1:])]
        dn = np.r_[min(P.local_mw, p0[0]), np.minimum(P.ramp_frac * G.gen_pmax[1:] * c[r], p0[1:])]
        D_i = np.bincount(lab, weights=demand, minlength=n_isl); G_i = np.bincount(gen_isl, weights=p0, minlength=n_isl)
        U_i = np.bincount(gen_isl, weights=up, minlength=n_isl); N_i = np.bincount(gen_isl, weights=dn, minlength=n_isl)
        has_gen = np.bincount(gen_isl, minlength=n_isl) > 0
        deficit = np.where(has_gen, np.maximum(D_i - G_i, 0), 0); surplus = np.where(has_gen, np.maximum(G_i - D_i, 0), 0)
        lb = np.maximum(deficit - U_i, 0)
        out[r] = [deficit.sum(), surplus.sum(), lb.sum(), U_i[has_gen].sum(), N_i[has_gen].sum(), n_isl, D_i.max() / demand.sum()]
    return out


R1 = {k: r1_block(v) for k, v in (("train", tr), ("val", va), ("test", te))}
ISL = {k: island_struct(v) for k, v in (("train", tr), ("val", va), ("test", te))}
print("features built", flush=True)
rng0 = np.random.default_rng(1)
GB = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
      for a, b, c, e in zip(rng0.choice([15, 31, 63], 12), rng0.choice([20, 50, 100, 200], 12), rng0.choice([0.5, 0.8, 1.0], 12), rng0.choice([0.0, 1.0, 10.0], 12))]


def rung_features(name, split, d, vt):
    c = w.CV[w.cidx[d["st"]]]
    base0 = F0[split][d["io"], d["jc"]]
    if name == "R0":
        return np.hstack([base0, c, c.sum(1, keepdims=True)]).astype(np.float32)
    r1 = np.hstack([base0[:, :36], R1[split][d["io"], d["jc"]], base0[:, 78:]])      # demand, dispatch | aggregates | phys block
    if name == "R1":
        return np.hstack([r1, c, c.sum(1, keepdims=True)]).astype(np.float32)
    return np.hstack([r1, c, c.sum(1, keepdims=True), r2_rows(vt, ISL[split], d, c)]).astype(np.float32)


dva = v2data.build(va, F0["val"], w, 0.5, 0.0, K_VA, seed=11)
dte = v2data.build(te, F0["test"], w, 0.5, 0.0, K_TE, seed=13)
is_n2 = dte["jc"] >= 0
b2 = te["conts"][dte["io"], dte["jc"], 1]
n2 = b2 >= 0
rows, t0, tuned = [], time.time(), {}
for n in NS:
    for rep in range(REPS):
        idx = np.sort(np.random.default_rng(1000 * n + rep).choice(len(tr["op_ids"]), n, replace=False))
        vsub = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
        # island structure indices must follow the subset: rebuild lookup with subset positions
        dtr = v2data.build(vsub, F0["train"][idx], w, 0.5, 0.0, K_TR, seed=100 + n + 7 * rep)
        full_idx = idx[dtr["io"]]
        dtr_full = dict(dtr, io=full_idx)                       # rows referring to the FULL training table
        for name in ("R0", "R1", "R2"):
            Xtr = rung_features(name, "train", dtr_full, tr); Xva = rung_features(name, "val", dva, va); Xte = rung_features(name, "test", dte, te)
            if (n, name) not in tuned:
                best = None
                for p in GB:
                    m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
                    m.fit(Xtr, dtr["y"], eval_set=[(Xva, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
                    e = float(((m.predict(Xva) - dva["y"]) ** 2).mean())
                    if best is None or e < best[0]:
                        best = (e, p)
                tuned[(n, name)] = best[1]
            m = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **tuned[(n, name)])
            m.fit(Xtr, dtr["y"], eval_set=[(Xva, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
            pred = m.predict(Xte)
            for kind, msk in (("N-1", ~n2), ("N-2", n2)):
                for o in np.unique(dte["op"][msk]):
                    mm = msk & (dte["op"] == o)
                    for thr in THR:
                        rows.append(dict(n=n, rep=rep, rung=name, kind=kind, op=int(o), thr=thr, rprec=rprec(dte["y"][mm], pred[mm], dte["key"][mm], thr)))
        print(f"n={n} rep={rep} {time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows)
r.to_parquet(f"{D}/value_ladder.parquet")
print("\nmean R-precision by rung (exact control known; ceiling 1.0)")
t = r.groupby(["kind", "thr", "n", "rung"]).rprec.mean().unstack("rung").round(3)
t["gap_closed_R1"] = ((t["R1"] - t["R0"]) / (1 - t["R0"])).round(2); t["gap_closed_R2"] = ((t["R2"] - t["R0"]) / (1 - t["R0"])).round(2)
print(t.to_string())
