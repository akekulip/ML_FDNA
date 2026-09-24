import glob, sys
import numpy as np
import pandas as pd
D = "data_v2"
e1 = pd.read_parquet(f"{D}/e1_results.parquet"); e1 = e1[e1.n == 100].assign(variant="v2")
parts = [pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/e1b_*.parquet"))]
r = pd.concat(parts)
rng = np.random.default_rng(5)
print("E1b inference headroom = R(exact-control GBM) - R(best obs tree of A1,A2), n=100 ops, exploratory test 200-279")
for (var, q, s), g in r.groupby(["variant", "q", "s"]):
    src = g if var != "v2" else pd.concat([g, e1[(e1.q == q) & (e1.s == s) & e1.arm.isin(["A1_obs", "A2_logic"])]])
    pv = src.pivot_table(index="op", columns="arm", values="rprec")
    best = pv[["A1_obs", "A2_logic"]].max(axis=1)
    dl = (pv["exactc"] - best).values; ix = rng.integers(0, len(dl), (10000, len(dl)))
    lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5])
    print(f"{var} q={q} s={s}: exact-c GBM {pv['exactc'].mean():.3f} | best obs tree {best.mean():.3f} | oracle {pv['oracle'].mean():.3f} | inference headroom {dl.mean():+.4f} [{lo:+.4f},{hi:+.4f}]")
