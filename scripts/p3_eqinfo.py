"""Phase 3 Step 1 baseline audit (EXPLORATORY, screen on inspected block): adds equal-information comparator A9 to the D contrast.
Derived from d_compose.py. Decomposed composition branch (registered before coding): g(op, outage, c) learned on full-information rows, inference
p(c|obs) learned separately on cheap (obs, c) pairs, composed over the top-16 posterior control vectors.
env: VARIANT, CELL_IDX, TEST, N_LIST (operating points for g), REPS, N_INF, N_JOBS."""
import os, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.blocks import block_seed, close_block, open_block
from fdna.evalutil import op_metrics
from fdna.nn import belief, inference

D = "data_v2"
VARIANT = os.environ.get("VARIANT", "v2b"); CELL_IDX = int(os.environ["CELL_IDX"]); TEST = os.environ.get("TEST", "test")
NS = [int(x) for x in os.environ.get("N_LIST", "25,100").split(",")]; REPS = int(os.environ.get("REPS", 3))
N_INF = int(os.environ.get("N_INF", 25000)); NJ = int(os.environ.get("N_JOBS", 6)); TOPK = 64
TUNE = os.environ.get("TUNE", "confirm" if TEST != "test" else "screen")
GRID = {"v2": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)], "v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)],
        "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
K_TR, K_VA, K_TE = 20, 10, 8
if TEST != "test":
    slot = f"D_P{1 if VARIANT == 'v2b' else 2}"
    open_block(TEST, slot, "d", VARIANT, CELL_IDX,
               env=dict(N_LIST=NS, REPS=REPS, N_INF=N_INF, TUNE=TUNE, TOPK=TOPK))            # lock BEFORE any block data is loaded
w = v2.build_world(); lv = inference.level_index(w.CV)
tdir = D if TEST == "test" else "data_v2_confirm"
tr, va = (v2data.load_vtable(D, x) for x in ("train", "val")); te = v2data.load_vtable(tdir, TEST)
ops_file = "data/ops.npz" if TEST == "test" else f"{tdir}/ops_{TEST}.npz"
F = {"train": v2data.cont_features(tr, "data/ops.npz"), "val": v2data.cont_features(va, "data/ops.npz"), "test": v2data.cont_features(te, ops_file)}
rng0 = np.random.default_rng(1)
NT = 12 if TUNE == "screen" else 40
GB = [dict(num_leaves=int(a), min_child_samples=int(b), feature_fraction=float(c), reg_lambda=float(e), learning_rate=0.1)
      for a, b, c, e in zip(rng0.choice([15, 31, 63], NT), rng0.choice([20, 50, 100, 200], NT), rng0.choice([0.5, 0.8, 1.0], NT), rng0.choice([0.0, 1.0, 10.0], NT))]
NN_GRID = [(1e-3, 0.0), (3e-3, 0.0), (1e-3, 1e-4), (3e-3, 1e-4)] if TUNE == "screen" else [(lr, wd) for lr in (1e-3, 2e-3, 5e-3) for wd in (0.0, 1e-5, 1e-4)]


def gfeat(Xc, c):
    return np.hstack([Xc, c, c.sum(1, keepdims=True)]).astype(np.float32)


def tune_gbm(Xtr, ytr, Xva, yva):
    best = None
    for p in GB:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **p)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(30, verbose=False)])
        e = float(((m.predict(Xva) - yva) ** 2).mean())
        if best is None or e < best[0]:
            best = (e, p)
    return best[1]


def compose(logp, g, Xc, K=TOPK):
    p = np.exp(logp)
    top = np.argpartition(-p, K, axis=1)[:, :K]
    pk = np.take_along_axis(p, top, 1)
    cover = float(pk.sum(1).mean())
    pk = pk / pk.sum(1, keepdims=True)
    out = np.zeros(len(Xc))
    for j in range(K):
        out += pk[:, j] * g.predict(gfeat(Xc, w.CV[top[:, j]]))
    return out, cover


dva = v2data.build(va, F["val"], w, q, s, K_VA, seed=11, coverage=COV)
dte = v2data.build(te, F["test"], w, q, s, K_TE, seed=block_seed(TEST), only_n2=True, coverage=COV)
pc_exact = v2data.oracle_features(w, dte["obs"], s)[0]
Ete = belief.obs_tensor(dte["obs"])
# inference modules (trained once per replicate on N_INF cheap (obs, c) pairs)
rows, t0, tuned_inf, tuned_g = [], time.time(), {}, {}
def make_mk(rep):
    return {"D_I1_mlp": lambda: inference.InfMLP(1024)}


def draw(n, seed):
    rng = np.random.default_rng(seed)
    st = v2.sample_states(w, rng, n)
    obs = v2.emit(w, st, q, s, rng, COV)
    return obs, w.cidx[st]


for rep in range(REPS):
    obs_tr, y_tr = draw(N_INF, 5000 + rep); obs_va, y_va = draw(20000, 7)
    Etr, Eva = belief.obs_tensor(obs_tr), belief.obs_tensor(obs_va)
    logps = {"D_exact": np.log(np.maximum(pc_exact, 1e-30))}
    for name, mk in make_mk(rep).items():
        if name not in tuned_inf:
            best = None
            for lr, wd in NN_GRID:
                _, e = inference.fit_ce(mk, Etr, y_tr, Eva, y_va, lr=lr, wd=wd, seed=0)
                if best is None or e < best[0]:
                    best = (e, (lr, wd))
            tuned_inf[name] = best[1]
        lr, wd = tuned_inf[name]
        net, _ = inference.fit_ce(mk, Etr, y_tr, Eva, y_va, lr=lr, wd=wd, seed=rep)
        logps[name] = inference.predict_logp(net, Ete)
    X = lambda o_: np.where(o_ < 0, np.nan, o_).astype(np.float32)
    lp = np.zeros((len(Ete), 1024), np.float32)
    clfs = []
    for gi in range(5):
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.1, random_state=rep, verbose=-1, n_jobs=NJ)
        m.fit(X(obs_tr), lv[y_tr][:, gi]); clfs.append(m)
        pg = np.full((len(Ete), 4), 1e-6); pg[:, m.classes_] = m.predict_proba(X(dte["obs"])); pg /= pg.sum(1, keepdims=True)
        lp += np.log(pg)[:, lv[:, gi]]

    def pgfeat(obs):
        outp = []
        for gi, m in enumerate(clfs):
            pg = np.full((len(obs), 4), 1e-6); pg[:, m.classes_] = m.predict_proba(X(obs)); pg /= pg.sum(1, keepdims=True); outp.append(pg)
        return np.hstack(outp).astype(np.float32)
    logps["D_I2_gbm_product"] = lp - np.log(np.exp(lp).sum(1, keepdims=True))
    for n in NS:
        idx = np.sort(np.random.default_rng(1000 * n + rep).choice(len(tr["op_ids"]), n, replace=False))
        vsub = {k: (v[idx] if k in ("op_ids", "conts", "V") else v) for k, v in tr.items()}
        dtr = v2data.build(vsub, F["train"][idx], w, q, s, K_TR, seed=100 + n + 7 * rep, coverage=COV)
        c_tr, c_va = w.CV[w.cidx[dtr["st"]]], w.CV[w.cidx[dva["st"]]]
        Xg_tr, Xg_va = gfeat(dtr["Xc"], c_tr), gfeat(dva["Xc"], c_va)
        if n not in tuned_g:
            tuned_g[n] = tune_gbm(Xg_tr, dtr["y"], Xg_va, dva["y"])
        g = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **tuned_g[n])
        g.fit(Xg_tr, dtr["y"], eval_set=[(Xg_va, dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
        # end-to-end comparators trained on the SAME LP rows (observations only) and, for A8, given the cheap-pair inference marginals
        def marg(logp_or_net, d_):
            lp_ = inference.predict_logp(logp_or_net, belief.obs_tensor(d_["obs"]))
            p_ = np.exp(lp_); return np.hstack([p_ @ w.CV.astype(np.float32), p_ @ (w.CV == 0).astype(np.float32)]).astype(np.float32)
        lr1, wd1 = tuned_inf["D_I1_mlp"]
        net1, _ = inference.fit_ce(make_mk(rep)["D_I1_mlp"], Etr, y_tr, Eva, y_va, lr=lr1, wd=wd1, seed=rep)
        e2e = {               "A8_hybrid_generic_marginals": (lambda d_: np.hstack([d_["Xc"], marg(net1, d_)])),
               "A9_equal_info": (lambda d_: np.hstack([d_["Xc"], np.where(d_["obs"] < 0, np.nan, d_["obs"]).astype(np.float32), pgfeat(d_["obs"])]))}
        preds = {}
        for name, fn in e2e.items():
            key = (name, n)
            if key not in tuned_g:
                tuned_g[key] = tune_gbm(fn(dtr), dtr["y"], fn(dva), dva["y"])
            m_ = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=NJ, bagging_fraction=0.8, bagging_freq=1, **tuned_g[key])
            m_.fit(fn(dtr), dtr["y"], eval_set=[(fn(dva), dva["y"])], callbacks=[lgb.early_stopping(30, verbose=False)])
            preds[name] = (m_.predict(fn(dte)), 1.0)
        for name, logp in logps.items():
            preds[name] = compose(logp, g, dte["Xc"])
        for name, (pred, cover) in preds.items():
            for o in np.unique(dte["op"]):
                m = dte["op"] == o
                rows.append(dict(variant=VARIANT, q=q, s=s, n=n, rep=rep, arm=name, op=int(o), top16_mass=cover,
                                 **op_metrics(dte["y"][m], pred[m], dte["key"][m])))
        print(f"{VARIANT} q={q} s={s} rep={rep} n={n} {time.time()-t0:.0f}s", flush=True)
out = f"{D}/p3_eqinfo_{TEST}_{VARIANT}_{CELL_IDX}.parquet"
pd.DataFrame(rows).to_parquet(out)
if TEST != "test":
    print("output sha256", close_block(TEST, slot, out))
print("done")
