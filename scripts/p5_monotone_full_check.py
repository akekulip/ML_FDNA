"""Phase 5 Stage R2 (external review finding 10): a SAVED, re-runnable script for the FULL (not strided)
control-monotonicity check. The original claim of '415,699,200 pairs, zero violations, not a sample' was run
once interactively and never committed as a reproducible artifact -- the committed pytest only samples 84/4200
rows for speed. This script IS the full check, and its output is committed so the claim is independently
reproducible from a script, not only from session history."""
import json
import time
import numpy as np

t0 = time.time()
n3 = np.load("data_hik/n3_confirm.npz")
CV, V = n3["CV"], n3["V"]
dom = (CV[:, None, :] >= CV[None, :, :]).all(2)
np.fill_diagonal(dom, False)
n_pairs_per_row = int(dom.sum())
total_viol = 0
for r in range(len(V)):
    v = V[r]
    diff = v[:, None] - v[None, :]
    total_viol += int(((diff > 1e-6) & dom).sum())
out = {"n_rows": len(V), "n_pairs_per_row": n_pairs_per_row, "total_comparable_pairs": len(V) * n_pairs_per_row,
       "total_violations": total_viol, "wall_s": round(time.time() - t0, 1)}
print(json.dumps(out, indent=1))
json.dump(out, open("results/phase5/monotone_full_check.json", "w"), indent=1)
