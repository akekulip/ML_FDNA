import sys
import numpy as np
import pandas as pd
D = "data_v2"; TEST = sys.argv[1] if len(sys.argv) > 1 else "test"
i8 = pd.concat([pd.read_parquet(f"{D}/inf8_{TEST}_{v}_3.parquet") for v in ("v2b", "v2c")])
old = pd.concat([pd.read_parquet(f"{D}/inf_{TEST}_{v}_3.parquet") for v in ("v2b", "v2c")])
r = pd.concat([i8[i8.arm != "oracle_exact_posterior"], old[old.N.isin(i8.N.unique())]])
rng = np.random.default_rng(41)
print("mean per-op R-precision (expected-shed scoring / P(severe) scoring) and KL to exact")
for col in ("rprec", "rprec_sev", "kl_exact"):
    print(col); print(r.groupby(["variant", "q", "s", "N", "arm"])[col].mean().unstack("arm").round(3).to_string())
print("\npaired I8 minus X, 95% cluster CI (per-op, reps averaged)")
gen = ["I1_mlp", "I2_gbm_product", "I3_minmax_gnn"]
for (var, q, s, N), g in r.groupby(["variant", "q", "s", "N"]):
    pv = g.groupby(["op", "arm"]).rprec.mean().unstack("arm")
    out = []
    for nm, ref in (("best-mean generic (" + str(pv[gen].mean().idxmax()) + ")", pv[pv[gen].mean().idxmax()]), ("I5_fdna", pv["I5_fdna"])):
        dl = (pv["I8_bayes_structure"] - ref).values; ix = rng.integers(0, len(dl), (5000, len(dl)))
        lo, hi = np.percentile(dl[ix].mean(1), [2.5, 97.5]); out.append(f"{nm} {dl.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
    print(f"{var} q={q} s={s} N={N}: I8 - " + "  ".join(out))
