"""Phase 3: correlate the LODF compensation risk 1/|Det(a,b)| with the measured flow-addition error (exact,
no training). Reuses the 10-op sample from p3_step1_flowadd.py. Registered as an extension of registry/phase3_step1.yaml."""
import json
import numpy as np
from scipy.stats import spearmanr

from fdna.dataset import G, PAIRS
from fdna.lodf import ptdf_lodf, compensation_det
from fdna import v2data
from fdna.grid import BASE_MVA
from fdna.opgen import sample_op
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

L = G.n_branch
_, LODF = ptdf_lodf(G)
risk = {(a, b): 1 / max(abs(compensation_det(LODF, a, b)), 1e-12) for a, b in PAIRS}


def signed_flow(demand, p0, removed):
    k = np.array([i for i in range(L) if i not in removed]); w = BASE_MVA / G.x[k]
    inj = -demand.copy(); np.add.at(inj, G.gen_bus, p0)
    Lap = np.zeros((G.n_bus, G.n_bus))
    np.add.at(Lap, (G.frm[k], G.frm[k]), w); np.add.at(Lap, (G.to[k], G.to[k]), w)
    np.add.at(Lap, (G.frm[k], G.to[k]), -w); np.add.at(Lap, (G.to[k], G.frm[k]), -w)
    n, _ = connected_components(coo_matrix((np.ones(len(k)), (G.frm[k], G.to[k])), shape=(G.n_bus,) * 2), directed=False)
    if n != 1: return None
    th = np.zeros(G.n_bus); th[1:] = np.linalg.solve(Lap[1:, 1:], inj[1:])
    f = np.zeros(L); f[k] = w * (th[G.frm[k]] - th[G.to[k]]); return f


te = v2data.load_vtable("data_v2", "test")
from fdna.dataset import RATING
rows = []
for oid in te["op_ids"][:10]:
    op = sample_op(G, RATING, np.random.default_rng(int(oid)))
    f0 = signed_flow(op.demand, op.p0, ())
    f1 = {a: signed_flow(op.demand, op.p0, (a,)) for a in range(L)}
    for (a, b) in PAIRS:
        if f1[a] is None or f1[b] is None: continue
        fe = signed_flow(op.demand, op.p0, (a, b))
        if fe is None: continue
        fn = f0 + (f1[a] - f0) + (f1[b] - f0)
        alive = np.ones(L, bool); alive[[a, b]] = False
        e = (np.abs(fn - fe)[alive] / RATING[alive]).max()
        rows.append((risk[(a, b)], e))
r = np.array(rows)
finite = np.isfinite(r[:, 0])
rho, p = spearmanr(r[finite, 0], r[finite, 1])
top10 = np.quantile(r[finite, 0], 0.9)
flagged = r[:, 0] >= top10
p95_thr = np.quantile(r[:, 1], 0.95)
out = {"spearman_rho": float(rho), "p_value": float(p), "n": int(finite.sum()),
       "recall_p95_tail_at_top10pct_flag": float((r[flagged, 1] >= p95_thr).sum() / max((r[:, 1] >= p95_thr).sum(), 1)),
       "median_err_flagged": float(np.median(r[flagged, 1]) * 100), "median_err_unflagged": float(np.median(r[~flagged, 1]) * 100)}
print(out); json.dump(out, open("results/phase3/step1_lodf_corr.json", "w"), indent=1)
