"""Phase 4 correction (brief section 4.3): small, HONEST inference-quality test, not the abandoned R-precision screen.
Reports NLL and KL-to-exact-posterior only, for each repaired-FDNA arm vs comparators. No GBM value model, no
R-precision/composition step -- the research-scientist review (this session) showed that endpoint cannot resolve an
effect this small at 3 reps; NLL/KL on tens of thousands of i.i.d. draws can. Single GPU process, one cell at a time,
per-arm progress printed. or_combine is fixed to 'max' for every or_aware arm regardless of tau (correction 4.3)."""
import os, time
import numpy as np
import pandas as pd
import torch

from fdna import v2, v2data
from fdna.nn import belief, inference
from fdna.nn.bayes import BayesStructure
from fdna.nn.repair import ReprInf

D = "data_v2"
VARIANT = os.environ["VARIANT"]; CELL_IDX = int(os.environ.get("CELL_IDX", 3)); REPS = int(os.environ.get("REPS", 3))
N_INF = int(os.environ.get("N_INF", 25000))
GRID = {"v2b": [(q, s) for q in (0.5, 0.7) for s in (0.0, 0.3)], "v2c": [(q, s) for q in (0.1, 0.3) for s in (0.0, 0.2)]}
COV = v2.COVERAGE_SPARSE if VARIANT == "v2c" else None
q, s = GRID[VARIANT][CELL_IDX]
w = v2.build_world(); lv = inference.level_index(w.CV)
te = v2data.load_vtable(D, "test")
NN_GRID = [(1e-3, 0.0), (3e-3, 0.0), (1e-3, 1e-4), (3e-3, 1e-4)]
R = lambda **k: (lambda: ReprInf(w, lv, **k))
ARMS = {
    "I5_ref": R(head="meanfield", or_aware=False, tau=0.05),
    "I5_tau0": R(head="meanfield", or_aware=False, tau=0.0),
    "I5_OR": R(head="meanfield", or_aware=True, tau=0.05),
    "I5_OR_hard": R(head="meanfield", or_aware=True, tau=0.0),
    "J_plain": R(head="joint", or_aware=False, tau=0.05, prior="indep"),
    "J_plain_CC": R(head="joint", or_aware=False, tau=0.05, prior="cc"),
    "J_OR_hard_CC": R(head="joint", or_aware=True, tau=0.0, prior="cc"),
    "L_indep": R(head="logic", prior="indep"),
    "L_CC": R(head="logic", prior="cc"),
    "I1_mlp": lambda: inference.InfMLP(1024),
    "I8_bayes": lambda: BayesStructure(w),
}
K_TE = 4
dte = v2data.build(te, v2data.cont_features(te, "data/ops.npz"), w, q, s, K_TE, seed=13, only_n2=True, coverage=COV)
pc_exact = v2data.oracle_features(w, dte["obs"], s)[0]; logp_exact = np.log(np.maximum(pc_exact, 1e-30))
Ete = belief.obs_tensor(dte["obs"]); true_c = w.cidx[dte["st"]]


def draw(n, seed):
    rng = np.random.default_rng(seed); st = v2.sample_states(w, rng, n)
    return v2.emit(w, st, q, s, rng, COV), w.cidx[st]


rows, tuned_inf, t0 = [], {}, time.time()
for rep in range(REPS):
    obs_tr, y_tr = draw(N_INF, 5000 + rep); obs_va, y_va = draw(20000, 7)
    Etr, Eva = belief.obs_tensor(obs_tr), belief.obs_tensor(obs_va)
    for name, mk in ARMS.items():
        ta = time.time()
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
            print(f"  {name}: FAILED to train (no finite val loss)", flush=True); continue
        lr, wd = tuned_inf[name]
        net, val_nll = inference.fit_ce(mk, Etr, y_tr, Eva, y_va, lr=lr, wd=wd, seed=rep)
        logp = inference.predict_logp(net, Ete)
        nll = float(-logp[np.arange(len(true_c)), true_c].mean())
        kl = float((pc_exact * (logp_exact - logp)).sum(1).mean())
        rows.append(dict(variant=VARIANT, q=q, s=s, rep=rep, arm=name, nll=nll, kl_exact_to_arm=kl, val_ce_loss=val_nll))
        print(f"  rep={rep} {name}: nll={nll:.4f} kl={kl:.4f} ({time.time()-ta:.0f}s, total {time.time()-t0:.0f}s)", flush=True)
pd.DataFrame(rows).to_parquet(f"{D}/p3_repair_infer_{VARIANT}_{CELL_IDX}.parquet")
print("done", time.time() - t0, "s total")
