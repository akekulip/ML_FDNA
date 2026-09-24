"""Inference-only branch (registered before running). env: VARIANT, CELL_IDX, TEST (test|confirm|replicate), N_LIST, REPS."""
import json, os, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import v2, v2data
from fdna.evalutil import op_metrics
from fdna.nn import belief, inference

D = "data_v2"
VARIANT = os.environ.get("VARIANT", "v2b"); CELL_IDX = int(os.environ["CELL_IDX"]); TEST = os.environ.get("TEST", "test")
NS = [int(x) for x in os.environ.get("N_LIST", "1000,5000,25000,125000").split(",")]; REPS = int(os.environ.get("REPS", 3))
GRID = {"v2": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)], "v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)],
        "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
w = v2.build_world()
lv = inference.level_index(w.CV)
tdir = D if TEST == "test" else "data_v2_confirm"
te = v2data.load_vtable(tdir, TEST)
if TEST != "test":
    from fdna.blocks import open_block
    open_block(TEST, f"INF_{VARIANT}_cell{CELL_IDX}")
dummy = np.zeros(te["conts"].shape[:2] + (1,), np.float32)
dte = v2data.build(te, dummy, w, q, s, 8, seed=13, only_n2=True, coverage=COV)
pc_exact = v2data.oracle_features(w, dte["obs"], s)[0]
Ete = belief.obs_tensor(dte["obs"])
V_rows = lambda a, b: te["V"][dte["io"][a:b], dte["jc"][a:b]]
NN_GRID = [(1e-3, 0.0), (3e-3, 0.0), (1e-3, 1e-4), (3e-3, 1e-4)]


def draw(n, seed):
    rng = np.random.default_rng(seed)
    st = v2.sample_states(w, rng, n)
    return belief.obs_tensor(v2.emit(w, st, q, s, rng, COV)), w.cidx[st], v2.emit(w, st, q, s, np.random.default_rng(seed), COV)


def evaluate(name, logp, N, rep, rows):
    p = np.exp(logp)
    val = np.concatenate([(p[a:a + 8192] * V_rows(a, a + 8192)).sum(1) for a in range(0, len(p), 8192)])
    kl = float(np.mean((pc_exact * (np.log(np.maximum(pc_exact, 1e-12)) - logp)).sum(1)))
    acc = float((p.argmax(1) == pc_exact.argmax(1)).mean())
    for o in np.unique(dte["op"]):
        m = dte["op"] == o
        rows.append(dict(variant=VARIANT, q=q, s=s, N=N, rep=rep, arm=name, op=int(o), kl_exact=kl, top1_vs_exact=acc,
                         **op_metrics(dte["y"][m], val[m], dte["key"][m])))


rows, t0, tuned = [], time.time(), {}
evaluate("oracle_exact_posterior", np.log(np.maximum(pc_exact, 1e-30)), 0, 0, rows)
Eva, yva, _ = draw(20000, 7)
MK = {"I1_mlp": lambda: inference.InfMLP(1024), "I3_minmax_gnn": lambda: inference.InfGNN(1024),
      "I5_fdna": lambda: inference.InfFDNA(lv, "fdna"), "I6_shuffled": lambda: inference.InfFDNA(lv, "shuffled"),
      "I7_unconstrained": lambda: inference.InfFDNA(lv, "unconstrained")}
for N in NS:
    for rep in range(REPS):
        Etr, ytr, otr = draw(N, 1000 + N + rep)
        for name, mk in MK.items():
            key = (N, name)
            if key not in tuned:
                best = None
                for lr, wd in NN_GRID:
                    _, e = inference.fit_ce(mk(), Etr, ytr, Eva, yva, lr=lr, wd=wd, seed=0)
                    if best is None or e < best[0]:
                        best = (e, (lr, wd))
                tuned[key] = best[1]
            lr, wd = tuned[key]
            net, _ = inference.fit_ce(mk(), Etr, ytr, Eva, yva, lr=lr, wd=wd, seed=rep)
            evaluate(name, inference.predict_logp(net, Ete), N, rep, rows)
        # I2: product of per-generator GBM classifiers on raw observations
        X = lambda o_: np.where(o_ < 0, np.nan, o_).astype(np.float32)
        Xtr, Xte = X(otr), X(dte["obs"])
        lp = np.zeros((len(Xte), 1024), np.float32)
        for g in range(5):
            m = lgb.LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.1, random_state=rep, verbose=-1, n_jobs=6)
            m.fit(Xtr, lv[ytr][:, g])
            pg = np.full((len(Xte), 4), 1e-6); pg[:, m.classes_] = m.predict_proba(Xte); pg /= pg.sum(1, keepdims=True)
            lp += np.log(pg)[:, lv[:, g]]
        lp -= np.log(np.exp(lp).sum(1, keepdims=True))
        evaluate("I2_gbm_product", lp, N, rep, rows)
        print(f"{VARIANT} q={q} s={s} N={N} rep={rep} {time.time()-t0:.0f}s", flush=True)
r = pd.DataFrame(rows)
r.to_parquet(f"{D}/inf_{TEST}_{VARIANT}_{CELL_IDX}.parquet")
print("done")
