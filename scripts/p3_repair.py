"""Phase 3 Step 2 (EXPLORATORY screen, registry/phase3_step2.yaml): FDNA repair ablations + comparators.
Inference metrics for every arm; composed R-precision (shared value model g, top-64) for COMPOSE arms.
env: VARIANT (v2b|v2c), CELL_IDX (3), REPS, N_INF, N_JOBS, N (operating points for g)."""
import os, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import op_metrics
from fdna.nn import belief, inference
from fdna.nn.bayes import BayesStructure
from fdna.nn.repair import ReprInf

D = "data_v2"
VARIANT = os.environ["VARIANT"]; CELL_IDX = int(os.environ.get("CELL_IDX", 3)); REPS = int(os.environ.get("REPS", 3))
N_INF = int(os.environ.get("N_INF", 25000)); NJ = int(os.environ.get("N_JOBS", 6)); N = int(os.environ.get("N", 100)); TOPK = 64
GRID = {"v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)], "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
K_TR, K_VA, K_TE = 20, 10, 4
w = v2.build_world(); lv = inference.level_index(w.CV)
tr, va, te = (v2data.load_vtable(D, x) for x in ("train", "val", "test"))
F = {k: v2data.cont_features(t, "data/ops.npz") for k, t in (("train", tr), ("val", va), ("test", te))}
NN_GRID = [(1e-3, 0.0), (3e-3, 0.0), (1e-3, 1e-4), (3e-3, 1e-4)]
R = lambda **k: (lambda: ReprInf(w, lv, **k))
ARMS = {  # name: (constructor, compose?)
    "I5_ref": (R(head="meanfield", or_aware=False, tau=0.05), True),
    "I5_tau0.01": (R(head="meanfield", or_aware=False, tau=0.01), False),
    "I5_tau0": (R(head="meanfield", or_aware=False, tau=0.0), True),
    "I5_OR": (R(head="meanfield", or_aware=True, tau=0.05), True),
    "J_plain": (R(head="joint", or_aware=False, tau=0.05, prior="indep"), False),
    "J_plain_CC": (R(head="joint", or_aware=False, tau=0.05, prior="cc"), True),
    "I5_OR_hard": (R(head="meanfield", or_aware=True, tau=0.0), True),
    "J_OR_hard_CC": (R(head="joint", or_aware=True, tau=0.0, prior="cc"), True),
    "L_indep": (R(head="logic", prior="indep"), True),
    "L_CC": (R(head="logic", prior="cc"), True),
    "I1_mlp": (lambda: inference.InfMLP(1024), True),
    "I8_bayes": (lambda: BayesStructure(w), True),
}
rng0 = np.random.default_rng(1)
GB = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
      for a, b, c, e in zip(rng0.choice([15, 31, 63], 12), rng0.choice([20, 50, 100, 200], 12), rng0.choice([0.5, 0.8, 1.0], 12), rng0.choice([0.0, 1.0, 10.0], 12))]
gfeat = lambda Xc, c: np.hstack([Xc, c, c.sum(1, keepdims=True)]).astype(np.float32)


def tune_gbm(Xtr, ytr, Xva, yva):
    best = None
    for p in GB:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
        e = float(((m.predict(Xva) - yva) ** 2).mean())
        if best is None or e < best[0]: best = (e, p)
    return best[1]


def compose(logp, g, Xc, K=TOPK):
    p = np.exp(logp); top = np.argpartition(-p, K, axis=1)[:, :K]
    pk = np.take_along_axis(p, top, 1); cover = float(pk.sum(1).mean()); pk = pk / pk.sum(1, keepdims=True)
    out = np.zeros(len(Xc))
    for j in range(K): out += pk[:, j] * g.predict(gfeat(Xc, w.CV[top[:, j]]))
    return out, cover


dva = v2data.build(va, F["val"], w, q, s, K_VA, seed=11, coverage=COV)
dte = v2data.build(te, F["test"], w, q, s, K_TE, seed=13, only_n2=True, coverage=COV)
pc_exact = v2data.oracle_features(w, dte["obs"], s)[0]; logp_exact = np.log(np.maximum(pc_exact, 1e-30))
Ete = belief.obs_tensor(dte["obs"]); true_c = w.cidx[dte["st"]]
def draw(n, seed):
    rng = np.random.default_rng(seed); st = v2.sample_states(w, rng, n)
    return v2.emit(w, st, q, s, rng, COV), w.cidx[st]
rows, tuned_inf, tuned_g, t0 = [], {}, {}, time.time()
for rep in range(REPS):
    obs_tr, y_tr = draw(N_INF, 5000 + rep); obs_va, y_va = draw(20000, 7)
    Etr, Eva = belief.obs_tensor(obs_tr), belief.obs_tensor(obs_va)
    logps = {"exact": logp_exact}
    for name, (mk, _) in ARMS.items():
        if name not in tuned_inf:
            best = None
            for lr, wd in NN_GRID:
                try:
                    _, e = inference.fit_ce(mk, Etr, y_tr, Eva, y_va, lr=lr, wd=wd, seed=0)
                except FloatingPointError:
                    continue
                if best is None or e < best[0]: best = (e, (lr, wd))
            tuned_inf[name] = best[1] if best else None
        if tuned_inf[name] is None:
            print("arm failed to train:", name, flush=True); continue
        lr, wd = tuned_inf[name]
        net, _ = inference.fit_ce(mk, Etr, y_tr, Eva, y_va, lr=lr, wd=wd, seed=rep)
        logps[name] = inference.predict_logp(net, Ete)
    X = lambda o_: np.where(o_ < 0, np.nan, o_).astype(np.float32)
    lp = np.zeros((len(Ete), 1024), np.float32)
    for gi in range(5):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.1, random_state=rep, verbose=-1, n_jobs=NJ)
        m.fit(X(obs_tr), lv[y_tr][:, gi])
        pg = np.full((len(Ete), 4), 1e-6); pg[:, m.classes_] = m.predict_proba(X(dte["obs"])); pg /= pg.sum(1, keepdims=True)
        lp += np.log(pg)[:, lv[:, gi]]
    logps["I2_gbm_product"] = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    idx = np.sort(np.random.default_rng(1000 * N + rep).choice(len(tr["op_ids"]), N, replace=False))
    vsub = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
    dtr = v2data.build(vsub, F["train"][idx], w, q, s, K_TR, seed=100 + N + 7 * rep, coverage=COV)
    Xg_tr, Xg_va = gfeat(dtr["Xc"], w.CV[w.cidx[dtr["st"]]]), gfeat(dva["Xc"], w.CV[w.cidx[dva["st"]]])
    if N not in tuned_g: tuned_g[N] = tune_gbm(Xg_tr, dtr["y"], Xg_va, dva["y"])
    g = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **tuned_g[N])
    g.fit(Xg_tr, dtr["y"], eval_set=[(Xg_va, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
    for name, logp in logps.items():
        nll = float(-logp[np.arange(len(true_c)), true_c].mean())
        kl = float((pc_exact * (logp_exact - logp)).sum(1).mean())
        base = dict(variant=VARIANT, q=q, s=s, rep=rep, arm=name, nll=nll, kl_exact_to_arm=kl)
        compose_it = name in ("exact", "I2_gbm_product") or ARMS.get(name, (None, False))[1]
        if compose_it:
            pred, cover = compose(logp, g, dte["Xc"])
            for o in np.unique(dte["op"]):
                m_ = dte["op"] == o
                rows.append(dict(base, op=int(o), top64_mass=cover, **op_metrics(dte["y"][m_], pred[m_], dte["key"][m_])))
        else:
            rows.append(dict(base, op=-1))
    print(f"{VARIANT} rep={rep} {time.time()-t0:.0f}s", flush=True)
pd.DataFrame(rows).to_parquet(f"{D}/p3_repair_{VARIANT}_{CELL_IDX}.parquet"); print("done")
