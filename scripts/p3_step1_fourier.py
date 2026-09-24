"""Phase 3 Step 1d: low-degree structure of the value over the 14 comm-component bits, under the actual prior (weighted LS on failure-indicator monomials) and uniform-cube WHT energy.
Exploratory: test-table (op,outage) rows with non-constant value, random sample of 150."""
import itertools, json
import numpy as np
from fdna import v2, v2data

w = v2.build_world(); te = v2data.load_vtable("data_v2", "test")
p = np.exp(w.logprior); p /= p.sum(); n = 14
D = (~w.X).astype(np.float64)                       # failure indicators
cols = {0: [np.ones(len(D))]}
for d in (1, 2, 3):
    cols[d] = [np.prod(D[:, list(s)], 1) for s in itertools.combinations(range(n), d)]
B = {d: np.column_stack(sum((cols[k] for k in range(d + 1)), [])) for d in (1, 2, 3)}
sq = np.sqrt(p)
Bw = {d: B[d] * sq[:, None] for d in B}
# Walsh (uniform cube) energy by degree
def wht(v):
    v = v.copy(); h = 1
    while h < len(v):
        for i in range(0, len(v), h * 2):
            a, b = v[i:i + h].copy(), v[i + h:i + 2 * h].copy(); v[i:i + h], v[i + h:i + 2 * h] = a + b, a - b
        h *= 2
    return v / len(v)
pc = np.array([bin(i).count("1") for i in range(2 ** n)])
rng = np.random.default_rng(0); io, jc = np.meshgrid(np.arange(len(te["op_ids"])), np.arange(te["V"].shape[1]), indexing="ij")
io, jc = io.ravel(), jc.ravel(); order = rng.permutation(len(io)); res = []; done = 0
for t in order:
    i, j = io[t], jc[t]
    y = te["V"][i, j].astype(np.float64)[w.cidx]
    mu = (p * y).sum(); var = (p * (y - mu) ** 2).sum()
    if var < 1e-10: continue
    r = {"n2": bool(te["conts"][i, j, 1] >= 0), "var": var}
    for d in (1, 2, 3):
        coef, *_ = np.linalg.lstsq(Bw[d], y * sq, rcond=None)
        r[f"R2_deg{d}"] = 1 - (p * (y - B[d] @ coef) ** 2).sum() / var
    W = wht(y); e = W ** 2; e[0] = 0; tot = e.sum()
    for d in (1, 2, 3): r[f"wht_deg<={d}"] = float(e[pc <= d].sum() / tot)
    res.append(r); done += 1
    if done >= 150: break
out = {"n_rows": len(res)}
for k in ("R2_deg1", "R2_deg2", "R2_deg3", "wht_deg<=1", "wht_deg<=2", "wht_deg<=3"):
    for grp, sel in (("all", lambda r: True), ("n1", lambda r: not r["n2"]), ("n2", lambda r: r["n2"])):
        v = np.array([r[k] for r in res if sel(r)])
        out[f"{k}_{grp}"] = {"n": len(v), "median": float(np.median(v)), "p10": float(np.percentile(v, 10)), "mean": float(v.mean())}
json.dump(out, open("results/phase3/step1_fourier.json", "w"), indent=1)
for k, v in out.items():
    if isinstance(v, dict): print(k, {a: round(b, 3) for a, b in v.items()})
