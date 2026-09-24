"""Misspecification sweep for I8 (registered before coding): I8 built on an ASSUMED wiring with rho of the edges wrong. env: VARIANT, CELL_IDX, TEST
(test|confirm|replicate), N_LIST, REPS, TUNE (screen|confirm). Scores: rprec = expected shed under the learned posterior (registered
primary); rprec_sev = R-precision of P(severe) computed from the posterior (secondary; Bayes-optimal for the metric)."""
import os, time
import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import spec, v2, v2data, wiring
from fdna.blocks import block_seed, close_block, open_block
from fdna.evalutil import op_metrics
from fdna.nn import belief, inference
from fdna.nn.bayes import BayesStructure

D = "data_v2"
VARIANT = os.environ.get("VARIANT", "v2b"); CELL_IDX = int(os.environ["CELL_IDX"]); TEST = os.environ.get("TEST", "test")
NS = [int(x) for x in os.environ.get("N_LIST", "1000,5000").split(",")]; REPS = int(os.environ.get("REPS", 1))
ROBUST = bool(int(os.environ.get("ROBUST", "0")))
TUNE = os.environ.get("TUNE", "confirm" if TEST != "test" else "screen"); NJ = int(os.environ.get("N_JOBS", 6))
GRID = {"v2": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)], "v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)],
        "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
if TEST != "test":
    slot = f"INF8_P{1 if VARIANT == 'v2b' else 2}"
    open_block(TEST, slot, "inf8", VARIANT, CELL_IDX)           # lock BEFORE any block data is loaded
w = v2.build_world()
lv = inference.level_index(w.CV)
tdir = D if TEST == "test" else "data_v2_confirm"
te = v2data.load_vtable(tdir, TEST)
dummy = np.zeros(te["conts"].shape[:2] + (1,), np.float32)
dte = v2data.build(te, dummy, w, q, s, 8, seed=block_seed(TEST), only_n2=True, coverage=COV)
pc_exact = v2data.oracle_features(w, dte["obs"], s)[0]
Ete = belief.obs_tensor(dte["obs"])
V_rows = lambda a, b: te["V"][dte["io"][a:b], dte["jc"][a:b]]
NN_GRID = [(0.01, 0.0), (0.03, 0.0), (0.1, 0.0)]


def draw(n, seed):
    rng = np.random.default_rng(seed)
    st = v2.sample_states(w, rng, n)
    obs = v2.emit(w, st, q, s, rng, COV)
    return obs, belief.obs_tensor(obs), w.cidx[st]


def score(logp):
    p = np.exp(logp)
    ev, sv = [], []
    for a in range(0, len(p), 8192):
        Vr = V_rows(a, a + 8192)
        ev.append((p[a:a + 8192] * Vr).sum(1)); sv.append((p[a:a + 8192] * (Vr > spec.SEVERE)).sum(1))
    return np.concatenate(ev), np.concatenate(sv), p


def evaluate(name, logp, N, rep, rows, **extra):
    ev, sv, p = score(logp)
    kl = float(np.mean((pc_exact * (np.log(np.maximum(pc_exact, 1e-12)) - logp)).sum(1)))
    acc = float((p.argmax(1) == pc_exact.argmax(1)).mean())
    for o in np.unique(dte["op"]):
        m = dte["op"] == o
        a = op_metrics(dte["y"][m], ev[m], dte["key"][m]); b = op_metrics(dte["y"][m], sv[m], dte["key"][m])
        rows.append(dict(variant=VARIANT, q=q, s=s, N=N, rep=rep, arm=name, op=int(o), kl_exact=kl, top1_vs_exact=acc,
                         **a, rprec_sev=b["rprec"], **extra))


rows, t0, tuned = [], time.time(), {}
evaluate("oracle_exact_posterior", np.log(np.maximum(pc_exact, 1e-30)), 0, 0, rows)
_, Eva, yva = draw(20000, 7)
for N in NS:
    for rep in range(REPS):
        otr, Etr, ytr = draw(N, 1000 + N + rep)
        for rho in (0.0, 0.1, 0.2, 0.3):
            for cs in (range(3) if rho > 0 else range(1)):
                par, ug = wiring.corrupt(rho, 100 + cs)
                wa = v2.build_world_assumed(par, ug)
                mk = lambda wa=wa: BayesStructure(wa, robust=ROBUST)
                name = "I8_misspec_robust" if ROBUST else "I8_misspec"
                if (N, rho, cs) not in tuned:
                    best = None
                    for lr, wd in NN_GRID:
                        _, e = inference.fit_ce(mk, Etr, ytr, Eva, yva, lr=lr, wd=wd, seed=0, epochs=60, patience=8)
                        if best is None or e < best[0]:
                            best = (e, (lr, wd))
                    tuned[(N, rho, cs)] = best[1]
                lr, wd = tuned[(N, rho, cs)]
                net, _ = inference.fit_ce(mk, Etr, ytr, Eva, yva, lr=lr, wd=wd, seed=rep, epochs=60, patience=8)
                evaluate(name, inference.predict_logp(net, Ete), N, rep, rows, mis_rho=rho, cseed=cs)
                print(f"{VARIANT} N={N} rho={rho} cs={cs} {time.time()-t0:.0f}s", flush=True)
out = f"{D}/inf8mis{'rob' if ROBUST else ''}_{TEST}_{VARIANT}_{CELL_IDX}.parquet"
pd.DataFrame(rows).to_parquet(out)
if TEST != "test":
    print("output sha256", close_block(TEST, slot, out))
print("done")
