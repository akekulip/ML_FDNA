"""Branch 1 arms on v2 (run ONLY after gate E1 and the v2 freeze). Screen tier; confirm/replicate use blocks 400-479/500-579.
Usage: b1_arms.py DATA_DIR TEST_SPLIT_NAME [n_list] [reps] [cell_idx ...]"""
import json, sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import op_metrics
from fdna.nn.belief import BeliefFDNA, MinMaxGNN, obs_tensor
from fdna.nn.layers import MLP
from fdna.nn.train import fit, predict

D = sys.argv[1]
TEST = sys.argv[2] if len(sys.argv) > 2 else "test"
NS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["25", "100"])]
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
CELLS = [v2.CELLS[int(i)] for i in sys.argv[5:]] or v2.CELLS
K_TR, K_VA, K_TE = 20, 10, 8
w = v2.build_world()
tr, va, te = (v2data.load_vtable(D, s) for s in ("train", "val", TEST))
ops_file = "data/ops.npz" if TEST == "test" else f"{D}/ops_{TEST}.npz"
F = {s: v2data.cont_features(v, ops_file if s == TEST else "data/ops.npz") for s, v in (("train", tr), ("val", va), (TEST, te))}
e1p = json.load(open(f"{D}/e1_params.json"))
rows, t0 = [], time.time()


def gbm_pred(name, ftr, ytr, fva, yva, fte, q, s, n, rep):
    p = e1p[f"q{q}_s{s}_n{n}_{name}"]
    m = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=16, bagging_fraction=0.8, bagging_freq=1, **p)
    m.fit(ftr, ytr, eval_set=[(fva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m.predict(fte)


for (q, s) in CELLS:
    dva = v2data.build(va, F["val"], w, q, s, K_VA, seed=11)
    dte = v2data.build(te, F[TEST], w, q, s, K_TE, seed=13, only_n2=True)
    pc_te = v2data.oracle_features(w, dte["obs"], s)[0]
    oracle = (pc_te * te["V"][dte["io"], dte["jc"]]).sum(1)
    for n in NS:
        for rep in range(REPS):
            idx = np.sort(np.random.default_rng(1000 * n + rep).choice(len(tr["op_ids"]), n, replace=False))
            vsub = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
            dtr = v2data.build(vsub, F["train"][idx], w, q, s, K_TR, seed=100 + n + 7 * rep)
            mu, sd = dtr["Xc"].mean(0), dtr["Xc"].std(0) + 1e-6
            Z = lambda d_: ((d_["Xc"] - mu) / sd).astype(np.float32)
            E = lambda d_: obs_tensor(d_["obs"])
            preds = {"oracle": oracle}
            f1 = lambda d_: np.hstack([d_["Xc"], np.where(d_["obs"] < 0, np.nan, d_["obs"]).astype(np.float32)])
            f2 = lambda d_: np.hstack([d_["Xc"], v2data.logic_features(d_["obs"])])
            preds["A1_gbm_obs"] = gbm_pred("A1_obs", f1(dtr), dtr["y"], f1(dva), dva["y"], f1(dte), q, s, n, rep)
            preds["A2_gbm_logic"] = gbm_pred("A2_logic", f2(dtr), dtr["y"], f2(dva), dva["y"], f2(dte), q, s, n, rep)
            flat = lambda d_: np.hstack([Z(d_), E(d_).reshape(len(d_["y"]), -1)]).astype(np.float32)
            m3, _ = fit(MLP(flat(dtr).shape[1]), (flat(dtr),), dtr["y"], (flat(dva),), dva["y"], seed=rep)
            preds["A3_mlp"] = predict(m3, (flat(dte),))
            for name, mk in (("A4_minmax_gnn", lambda: MinMaxGNN(Z(dtr).shape[1])),
                             ("A5_fdna_belief", lambda: BeliefFDNA(Z(dtr).shape[1], "fdna")),
                             ("A6_shuffled", lambda: BeliefFDNA(Z(dtr).shape[1], "shuffled", seed=rep)),
                             ("A7_unconstrained", lambda: BeliefFDNA(Z(dtr).shape[1], "unconstrained"))):
                net, _ = fit(mk(), (Z(dtr), E(dtr)), dtr["y"], (Z(dva), E(dva)), dva["y"], seed=rep, epochs=60)
                preds[name] = predict(net, (Z(dte), E(dte)))
            for name, pr in preds.items():
                for o in np.unique(dte["op"]):
                    mm = dte["op"] == o
                    rows.append(dict(q=q, s=s, n=n, rep=rep, arm=name, op=int(o), **op_metrics(dte["y"][mm], pr[mm], dte["key"][mm])))
            print(f"cell q={q} s={s} n={n} rep={rep} {time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows); r.to_parquet(f"{D}/b1_results_{TEST}.parquet")
rng = np.random.default_rng(8)
print("\nB1 arms: mean per-op R-precision (test split", TEST, ")")
print(r.groupby(["q", "s", "n", "arm"]).rprec.mean().unstack("arm").round(3).to_string())
print("\npaired (per-op, reps averaged) A5 - X, 95% cluster CI")
for (q, s, n), g in r.groupby(["q", "s", "n"]):
    pv = g.groupby(["op", "arm"]).rprec.mean().unstack("arm")
    best = pv[["A1_gbm_obs", "A2_gbm_logic", "A3_mlp", "A4_minmax_gnn"]].max(axis=1)
    for name, ref in (("best(A1-A4)", best), ("A6_shuffled", pv["A6_shuffled"]), ("A7_unconstrained", pv["A7_unconstrained"])):
        dl = (pv["A5_fdna_belief"] - ref).values; ix = rng.integers(0, len(dl), (5000, len(dl)))
        lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5])
        print(f"q={q} s={s} n={n}: A5 - {name} = {dl.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
