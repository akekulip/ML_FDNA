"""Merge per-cell E1 outputs, write e1_params.json, print the gate table."""
import glob, json, sys
import numpy as np
import pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else "data_v2"
r = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/e1_results_*.parquet"))])
info = {}
for f in sorted(glob.glob(f"{D}/e1_params_*.json")):
    info.update(json.load(open(f)))
json.dump(info, open(f"{D}/e1_params.json", "w"))
r.to_parquet(f"{D}/e1_results.parquet")
rng = np.random.default_rng(3)
print("E1: mean per-op R-precision (test ops 200-279 exploratory, N-2 outages, K=8 draws)")
print("prevalence:", r.groupby(["q", "s"]).prev.mean().round(3).to_dict())
print(r.groupby(["q", "s", "n", "arm"]).rprec.mean().unstack("arm").round(3).to_string())
print()
for (q, s, n), g in r.groupby(["q", "s", "n"]):
    pv = g.pivot(index="op", columns="arm", values="rprec")
    best = pv[["A1_obs", "A2_logic"]].max(axis=1)
    dl = (pv["oracle"] - best).values
    ix = rng.integers(0, len(dl), (10000, len(dl)))
    lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5])
    ref = (pv["A2b_oraclefeat"] - best).values.mean()
    print(f"q={q} s={s} n={n}: oracle - best(A1,A2) = {dl.mean():+.4f} [{lo:+.4f},{hi:+.4f}] | oracle-feature GBM - best = {ref:+.4f}")
