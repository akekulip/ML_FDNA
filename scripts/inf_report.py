import glob, sys
import numpy as np
import pandas as pd
D = "data_v2"; TEST = sys.argv[1] if len(sys.argv) > 1 else "test"
r = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/inf_{TEST}_*_*.parquet"))])
rng = np.random.default_rng(21)
print(f"Inference-only, test block '{TEST}': mean per-op R-precision using the exact value table")
print(r.groupby(["variant", "q", "s", "N", "arm"]).rprec.mean().unstack("arm").round(3).to_string())
print("\nmean KL(exact posterior || model) (lower is better)")
print(r.groupby(["variant", "q", "s", "N", "arm"]).kl_exact.mean().unstack("arm").round(3).to_string())
print("\npaired I5 minus X, 95% cluster CI")
base = ["I1_mlp", "I2_gbm_product", "I3_minmax_gnn"]
for (var, q, s, N), g in r[r.arm != "oracle_exact_posterior"].groupby(["variant", "q", "s", "N"]):
    pv = g.groupby(["op", "arm"]).rprec.mean().unstack("arm")
    out = []
    for name, ref in (("best(I1,I2,I3)", pv[base].max(axis=1)), ("I6", pv["I6_shuffled"]), ("I7", pv["I7_unconstrained"])):
        dl = (pv["I5_fdna"] - ref).values; ix = rng.integers(0, len(dl), (5000, len(dl)))
        lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5]); out.append(f"{name} {dl.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
    print(f"{var} q={q} s={s} N={N}: I5 - " + "  ".join(out))
