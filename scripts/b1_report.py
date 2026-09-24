import glob, sys
import numpy as np
import pandas as pd
D = "data_v2"; TEST = sys.argv[1] if len(sys.argv) > 1 else "test"
r = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/b1_{TEST}_*_*.parquet"))])
rng = np.random.default_rng(12)
print(f"B1 arms, test block '{TEST}': mean per-op R-precision")
print(r.groupby(["variant", "q", "s", "n", "arm"]).rprec.mean().unstack("arm").round(3).to_string())
print("\npaired (per-op, reps averaged) A5 minus X with 95% cluster CI")
base = ["A1_gbm_obs", "A2_gbm_logic", "A3_mlp", "A4_minmax_gnn"]
for (var, q, s, n), g in r.groupby(["variant", "q", "s", "n"]):
    pv = g.groupby(["op", "arm"]).rprec.mean().unstack("arm")
    best = pv[base].max(axis=1)
    out = []
    for name, ref in (("best(A1-A4)", best), ("A6", pv["A6_shuffled"]), ("A7", pv["A7_unconstrained"]), ("A1", pv["A1_gbm_obs"]), ("A3", pv["A3_mlp"])):
        dl = (pv["A5_fdna_belief"] - ref).values; ix = rng.integers(0, len(dl), (5000, len(dl)))
        lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5]); out.append(f"{name} {dl.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
    print(f"{var} q={q} s={s} n={n}: A5 - " + "  ".join(out))
