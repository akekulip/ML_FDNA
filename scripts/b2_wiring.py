"""Branch 2 screen: corrupted assumed wiring (v1 benchmark, exploratory test 200-279).
Primary: rho=0.2, 3 corruption seeds; secondary rho in {0.1,0.3} (registry: win = A5 - A1 >= 0.05 at 20% corruption)."""
import sys, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data, spec, wiring
from fdna.baseline_data import evaluate, strata
from fdna.nn.layers import MLP
from fdna.nn.service import ServiceValueNet
from fdna.nn.train import fit, predict, to_t

RHOS = [float(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.2"])]
SEEDS = 3
NS, REPS = [10, 25, 100], 2
d = baseline_data.load("data")
NX = 123
Xr = d.F["elec+phys+raw_comm"]                 # elec(78) + phys(45) + raw flags(14)
Xe, flags = Xr[:, :NX], Xr[:, NX:]
mu, sd = Xe[d.tr].mean(0), Xe[d.tr].std(0) + 1e-6
Ze = ((Xe - mu) / sd).astype(np.float32)
fs = d.lab.fs.values
S = {k: v for k, v in strata(d).items() if k in ("n2_unseen|novel", "n2_unseen|familiar", "n2_unseen")}
ops_tr = np.unique(d.lab.op.values[d.tr])
GBM = dict(n_estimators=400, num_leaves=15, min_child_samples=100, colsample_bytree=0.5, learning_rate=0.1,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, n_jobs=8)
rows, edge_rows, t0 = [], [], time.time()


def gbm(X, mtr, rep):
    m = lgb.LGBMRegressor(random_state=rep, **GBM)
    m.fit(X[mtr], d.y[mtr], eval_set=[(X[d.va], d.y[d.va])], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m.predict(X[d.te])


def record(name, pred, n, rep, rho, cseed):
    full = np.full(len(d.y), np.nan); full[d.te] = pred
    for r in evaluate(d, name, full, rep, S):
        r.update(n=n, rep=rep, rho=rho, cseed=cseed); rows.append(r)


def ctrl_feats(C_all):
    C = C_all[fs]
    return np.hstack([Xe, C, C.sum(1, keepdims=True)]).astype(np.float32)


X_true = ctrl_feats(wiring.control_vectors_under(wiring.TRUE_PARENTS, wiring.TRUE_UNIT_GEN))
for n in NS:
    for rep in range(REPS):
        sub = ops_tr if n == 100 else np.random.default_rng(1000 * n + rep).choice(ops_tr, n, replace=False)
        mtr = d.tr & np.isin(d.lab.op.values, sub)
        # wiring-independent arms (run once per n, rep)
        record("A1_gbm_raw", gbm(Xr, mtr, rep), n, rep, -1, -1)
        record("A2t_gbm_truewiring", gbm(X_true, mtr, rep), n, rep, -1, -1)
        Zr = np.hstack([Ze, flags]).astype(np.float32)
        m3, _ = fit(MLP(Zr.shape[1]), (Zr[mtr],), d.y[mtr], (Zr[d.va],), d.y[d.va], seed=rep)
        record("A3_mlp_raw", predict(m3, (Zr[d.te],)), n, rep, -1, -1)
        for rho in RHOS:
            for cs in range(SEEDS):
                par, ug = wiring.corrupt(rho, 100 + cs)
                Xa = ctrl_feats(wiring.control_vectors_under(par, ug))
                record("A2c_gbm_assumed", gbm(Xa, mtr, rep), n, rep, rho, cs)
                for name, learn in (("A5_service_learn", True), ("A6_service_fixed", False)):
                    net = ServiceValueNet(par, ug, NX, learn_edges=learn)
                    net, _ = fit(net, (Ze[mtr], flags[mtr]), d.y[mtr], (Ze[d.va], flags[d.va]), d.y[d.va], seed=rep, epochs=50)
                    record(name, predict(net, (Ze[d.te], flags[d.te])), n, rep, rho, cs)
                    if learn:
                        f1, ga = net.svc.edge_recovery()
                        edge_rows.append(dict(n=n, rep=rep, rho=rho, cseed=cs, edge_f1=f1, gen_acc=ga))
                print(f"n={n} rep={rep} rho={rho} done {time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows); r.to_parquet("data/b2_results.parquet")
pd.DataFrame(edge_rows).to_parquet("data/b2_edges.parquet")
rng = np.random.default_rng(6)
print("\nBranch 2 screen: mean R-precision by arm and n (stratum n2_unseen|novel, then familiar)")
for st in ("n2_unseen|novel", "n2_unseen|familiar"):
    print("--", st); print(r[r.stratum == st].groupby(["arm" if "arm" in r else "model", "rho", "n"]).rprec.mean().unstack("n").round(3).to_string())
print("\nedge recovery (F1 of unit-parent edges, unit->generator accuracy):")
print(pd.DataFrame(edge_rows).groupby(["rho", "n"])[["edge_f1", "gen_acc"]].mean().round(3).to_string())
for rho in RHOS:
    print(f"\npaired A5 - A1 at rho={rho} (per-op, reps and corruption seeds averaged)")
    for st in ("n2_unseen|novel", "n2_unseen|familiar"):
        out = []
        for n in NS:
            g = r[(r.stratum == st) & (r.n == n)]
            pa = g[(g.model == "A5_service_learn") & (g.rho == rho)].groupby("op").rprec.mean()
            pb = g[g.model == "A1_gbm_raw"].groupby("op").rprec.mean()
            dl = (pa - pb).values; ix = rng.integers(0, len(dl), (5000, len(dl)))
            lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5]); out.append(f"n={n}: {dl.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
        print(st, "|", "  ".join(out))
