"""Phase 3 Step 1c: error of naive single-outage flow addition vs exact post-N-2 DC flow (connected pairs, 10 test ops). Exploratory."""
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fdna import v2data
from fdna.dataset import G, PAIRS, RATING
from fdna.grid import BASE_MVA
from fdna.opgen import sample_op

L = G.n_branch


def signed_flow(demand, p0, removed):
    k = np.array([i for i in range(L) if i not in removed]); w = BASE_MVA / G.x[k]
    inj = -demand.copy(); np.add.at(inj, G.gen_bus, p0)
    Lap = np.zeros((G.n_bus, G.n_bus))
    np.add.at(Lap, (G.frm[k], G.frm[k]), w); np.add.at(Lap, (G.to[k], G.to[k]), w)
    np.add.at(Lap, (G.frm[k], G.to[k]), -w); np.add.at(Lap, (G.to[k], G.frm[k]), -w)
    n, _ = connected_components(coo_matrix((np.ones(len(k)), (G.frm[k], G.to[k])), shape=(G.n_bus,) * 2), directed=False)
    if n != 1:
        return None
    th = np.zeros(G.n_bus); th[1:] = np.linalg.solve(Lap[1:, 1:], inj[1:])
    f = np.zeros(L); f[k] = w * (th[G.frm[k]] - th[G.to[k]]); return f


te = v2data.load_vtable("data_v2", "test")
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
        e = np.abs(fn - fe)[alive] / RATING[alive]
        oe, on = (np.abs(fe)[alive] / RATING[alive]) > 1, (np.abs(fn)[alive] / RATING[alive]) > 1
        rows.append((e.max(), (oe != on).any(), (np.abs(fe)[alive] / RATING[alive]).max()))
r = np.array(rows, float)
out = {"n_pairs_x_ops": len(r), "max_branch_err_pct_rating": {"median": float(np.median(r[:, 0]) * 100), "p95": float(np.percentile(r[:, 0], 95) * 100), "max": float(r[:, 0].max() * 100)},
       "frac_pairs_where_overload_set_changes": float(r[:, 1].mean()), "frac_pairs_with_any_overload_exact": float((r[:, 2] > 1).mean())}
print(out); json.dump(out, open("results/phase3/step1_flowadd.json", "w"), indent=1)
