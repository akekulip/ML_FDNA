"""Phase 3 Step 1a/1b: disjoint outage classes, interaction hypotheses, naive-flow-addition error (registry/phase3_step1.yaml).
Exploratory: train table (N-0/N-1) and test table ops 200-279. Output: results/phase3/step1_topo.json"""
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fdna import spec, v2data
from fdna.dataset import G, PAIRS, RATING
from fdna.features import post_outage_flows

L = G.n_branch


def comps(removed):
    k = np.array([i for i in range(L) if i not in removed])
    adj = coo_matrix((np.ones(len(k)), (G.frm[k], G.to[k])), shape=(G.n_bus, G.n_bus))
    n, lab = connected_components(adj, directed=False)
    has_gen = np.array([(lab[G.gen_bus] == i).any() for i in range(n)])
    return n, has_gen


def klass(removed):
    n, hg = comps(removed)
    if n == 1: return "connected"
    if hg.sum() == 1: return "gen_less_only"
    if (~hg).sum() == 0: return "all_gen"
    return "mixed"


cls1 = {a: klass((a,)) for a in range(L)}
out = {"n1": {c: sum(v == c for v in cls1.values()) for c in set(cls1.values())}}
cls2 = {p: klass(p) for p in PAIRS}
new2cut = {p: (cls2[p] != "connected" and cls1[p[0]] == "connected" and cls1[p[1]] == "connected") for p in PAIRS}
out["n2"] = {c: sum(v == c for v in cls2.values()) for c in set(cls2.values())}
out["n2_total"] = len(PAIRS); out["n2_islanding"] = sum(v != "connected" for v in cls2.values())
out["n2_new_2cut"] = int(sum(new2cut.values()))
out["n2_islanding_involving_n1_islanding_member"] = int(sum(1 for p in PAIRS if cls2[p] != "connected" and not new2cut[p]))
print(out)

tr = v2data.load_vtable("data_v2", "train"); te = v2data.load_vtable("data_v2", "test")
CV = tr["CV"]
full, none = int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))
print("CV[full]", CV[full], "CV[none]", CV[none])
y0 = tr["V"][:, 0, :]  # N-0 rows (cont (-1,-1))
assert (tr["conts"][:, 0] == -1).all()
out["y0_max_over_all_controls"] = float(y0.max()); out["y0_mean_full"] = float(y0[:, full].mean())

# interactions on the test table: y_a from the same op's N-1 rows, y_ab from N-2 rows
res = {}
for name, ci in (("full_control", full), ("no_control", none)):
    rows = []
    for i in range(len(te["op_ids"])):
        conts = te["conts"][i]; V = te["V"][i, :, ci]
        n1 = {int(a): V[j] for j, (a, b) in enumerate(conts) if b < 0}
        for j, (a, b) in enumerate(conts):
            if b >= 0:
                rows.append((n1[int(a)], n1[int(b)], V[j], cls2[(int(a), int(b))], new2cut[(int(a), int(b))]))
    r = np.array([(x[0], x[1], x[2]) for x in rows]); cl = np.array([x[3] for x in rows])
    ya, yb, yab = r.T
    tol = 1e-6
    d = {"n_rows": len(r),
         "frac_y_ab_lt_sum": float((yab < ya + yb - tol).mean()), "frac_y_ab_gt_sum": float((yab > ya + yb + tol).mean()),
         "frac_y_ab_lt_max": float((yab < np.maximum(ya, yb) - tol).mean()),
         "frac_y_ab_lt_min_braess": float((yab < np.minimum(ya, yb) - tol).mean()),
         "frac_nonzero_yab": float((yab > tol).mean())}
    for c in ("connected", "gen_less_only", "all_gen", "mixed"):
        m = cl == c
        if m.any():
            d[c] = {"n": int(m.sum()), "frac_sub": float((yab[m] < ya[m] + yb[m] - tol).mean()), "frac_super": float((yab[m] > ya[m] + yb[m] + tol).mean()),
                    "frac_lt_max": float((yab[m] < np.maximum(ya[m], yb[m]) - tol).mean()), "frac_braess": float((yab[m] < np.minimum(ya[m], yb[m]) - tol).mean()),
                    "mean_yab": float(yab[m].mean()), "sev_frac": float((yab[m] > spec.SEVERE).mean())}
    res[name] = d
out["interactions_test_table"] = res

json.dump(out, open("results/phase3/step1_topo.json", "w"), indent=1, default=float)
