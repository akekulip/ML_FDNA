"""Branch 1 arms on the frozen v2 family. env: VARIANT (v2b|v2c|v2), CELL_IDX, TEST (test|confirm|replicate), N_LIST, REPS, N_JOBS.
Tuning: GBM arms 12-trial random search; every neural arm gets the same 6-trial (lr, weight-decay) search, all on validation
data, once per (cell, n) at replicate 0, then reused. Data: train ops 0-99, val 100-119; test = screening block 200-279 by
default; TEST=confirm/replicate use data_v2_confirm/ (locked block wrapper)."""
import json, os, sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import op_metrics
from fdna.nn.belief import BeliefFDNA, MinMaxGNN, obs_tensor
from fdna.nn.layers import MLP
from fdna.nn.train import fit, predict

D = "data_v2"
VARIANT = os.environ.get("VARIANT", "v2b"); CELL_IDX = int(os.environ["CELL_IDX"]); TEST = os.environ.get("TEST", "test")
NS = [int(x) for x in os.environ.get("N_LIST", "25,100").split(",")]; REPS = int(os.environ.get("REPS", 3)); NJ = int(os.environ.get("N_JOBS", 4))
GRID = {"v2": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)], "v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)],
        "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
K_TR, K_VA, K_TE = 20, 10, 8
w = v2.build_world()
tdir = D if TEST == "test" else "data_v2_confirm"
tr, va = (v2data.load_vtable(D, x) for x in ("train", "val"))
te = v2data.load_vtable(tdir, TEST)
ops_file = "data/ops.npz" if TEST == "test" else f"{tdir}/ops_{TEST}.npz"
F = {"train": v2data.cont_features(tr, "data/ops.npz"), "val": v2data.cont_features(va, "data/ops.npz"), "test": v2data.cont_features(te, ops_file)}
rng0 = np.random.default_rng(1)
GB = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
      for a, b, c, e in zip(rng0.choice([15, 31, 63], 12), rng0.choice([20, 50, 100, 200], 12), rng0.choice([0.5, 0.8, 1.0], 12), rng0.choice([0.0, 1.0, 10.0], 12))]
NN_GRID = [(1e-3, 0.0), (2e-3, 0.0), (5e-3, 0.0), (1e-3, 1e-4), (2e-3, 1e-4), (5e-3, 1e-4)]


def tune_gbm(Xtr, ytr, Xva, yva):
    best = None
    for p in GB:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
        e = float(((m.predict(Xva) - yva) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, p)
    return best[1]


def fit_gbm(p, Xtr, ytr, Xva, yva, seed):
    m = lgb.LGBMRegressor(n_estimators=400, random_state=seed, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m


def tune_nn(mk, tri, ytr, vai, yva):
    best = None
    for lr, wd in NN_GRID:
        net, e = fit(mk(), tri, ytr, vai, yva, seed=0, epochs=60, lr=lr, wd=wd)
        if best is None or e < best[0]:
            best = (e, (lr, wd))
    return best[1]


def f_obs(d):
    return np.hstack([d["Xc"], np.where(d["obs"] < 0, np.nan, d["obs"]).astype(np.float32)])


def f_logic(d):
    return np.hstack([d["Xc"], v2data.logic_features(d["obs"])])


def f_exact(d):
    c = w.CV[w.cidx[d["st"]]]
    return np.hstack([d["Xc"], c, c.sum(1, keepdims=True)]).astype(np.float32)


if TEST != "test":
    from fdna.blocks import open_block
    open_block(TEST, f"B1_{VARIANT}_cell{CELL_IDX}")
rows, t0 = [], time.time()
dva = v2data.build(va, F["val"], w, q, s, K_VA, seed=11, coverage=COV)
dte = v2data.build(te, F["test"], w, q, s, K_TE, seed=13, only_n2=True, coverage=COV)
pc_te, marg_te = v2data.oracle_features(w, dte["obs"], s)
oracle = (pc_te * te["V"][dte["io"], dte["jc"]]).sum(1)
A2b = lambda d, mg: np.hstack([d["Xc"], mg])
for n in NS:
    tuned = {}
    for rep in range(REPS):
        idx = np.sort(np.random.default_rng(1000 * n + rep).choice(len(tr["op_ids"]), n, replace=False))
        vsub = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
        dtr = v2data.build(vsub, F["train"][idx], w, q, s, K_TR, seed=100 + n + 7 * rep, coverage=COV)
        _, marg_tr = v2data.oracle_features(w, dtr["obs"], s)
        _, marg_va = v2data.oracle_features(w, dva["obs"], s)
        mu, sd = dtr["Xc"].mean(0), dtr["Xc"].std(0) + 1e-6
        Z = lambda d_: ((d_["Xc"] - mu) / sd).astype(np.float32)
        E = lambda d_: obs_tensor(d_["obs"])
        flat = lambda d_: np.hstack([Z(d_), E(d_).reshape(len(d_["y"]), -1)]).astype(np.float32)
        preds = {"oracle": oracle}
        gb_sets = {"A1_gbm_obs": (f_obs, None), "A2_gbm_logic": (f_logic, None), "REF_gbm_exactc": (f_exact, None),
                   "REF_gbm_oraclefeat": (None, (marg_tr, marg_va, marg_te))}
        for name, (fn, mg) in gb_sets.items():
            if mg is None:
                Xtr_, Xva_, Xte_ = fn(dtr), fn(dva), fn(dte)
            else:
                Xtr_, Xva_, Xte_ = A2b(dtr, mg[0]), A2b(dva, mg[1]), A2b(dte, mg[2])
            if name not in tuned:
                tuned[name] = tune_gbm(Xtr_, dtr["y"], Xva_, dva["y"])
            preds[name] = fit_gbm(tuned[name], Xtr_, dtr["y"], Xva_, dva["y"], rep).predict(Xte_)
        nn_sets = {"A3_mlp": (lambda: MLP(flat(dtr).shape[1]), (flat(dtr),), (flat(dva),), (flat(dte),)),
                   "A4_minmax_gnn": (lambda: MinMaxGNN(Z(dtr).shape[1]), (Z(dtr), E(dtr)), (Z(dva), E(dva)), (Z(dte), E(dte))),
                   "A5_fdna_belief": (lambda: BeliefFDNA(Z(dtr).shape[1], "fdna"), (Z(dtr), E(dtr)), (Z(dva), E(dva)), (Z(dte), E(dte))),
                   "A6_shuffled": (lambda: BeliefFDNA(Z(dtr).shape[1], "shuffled", seed=rep), (Z(dtr), E(dtr)), (Z(dva), E(dva)), (Z(dte), E(dte))),
                   "A7_unconstrained": (lambda: BeliefFDNA(Z(dtr).shape[1], "unconstrained"), (Z(dtr), E(dtr)), (Z(dva), E(dva)), (Z(dte), E(dte)))}
        for name, (mk, tri, vai, tei) in nn_sets.items():
            if name not in tuned:
                tuned[name] = tune_nn(mk, tri, dtr["y"], vai, dva["y"])
            lr, wd = tuned[name]
            net, _ = fit(mk(), tri, dtr["y"], vai, dva["y"], seed=rep, epochs=60, lr=lr, wd=wd)
            preds[name] = predict(net, tei)
        for name, pr in preds.items():
            for o in np.unique(dte["op"]):
                mm = dte["op"] == o
                rows.append(dict(variant=VARIANT, q=q, s=s, n=n, rep=rep, arm=name, op=int(o), **op_metrics(dte["y"][mm], pr[mm], dte["key"][mm])))
        print(f"{VARIANT} q={q} s={s} n={n} rep={rep} {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_parquet(f"{D}/b1_{TEST}_{VARIANT}_{CELL_IDX}.parquet")
json.dump({str(k): v for k, v in tuned.items()}, open(f"{D}/b1_tuned_{TEST}_{VARIANT}_{CELL_IDX}.json", "w"), default=str)
print("done")
