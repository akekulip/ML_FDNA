import numpy as np
import pandas as pd
D = "data_v2"
m = pd.concat([pd.read_parquet(f"{D}/inf8mis_test_{v}_3.parquet") for v in ("v2b", "v2c")])
old = pd.concat([pd.read_parquet(f"{D}/inf_test_{v}_3.parquet") for v in ("v2b", "v2c")])
old = old[old.N.isin([1000, 5000]) & (old.arm != "oracle_exact_posterior")]
gen = old[old.arm.isin(["I1_mlp", "I2_gbm_product", "I3_minmax_gnn", "I5_fdna"])]
print("I8 with assumed wiring (rho = fraction of the 17 edges wrong), mean per-op R-precision (expected-shed / P(severe) scoring) and KL")
print(m.groupby(["variant", "N", "rho"])[["rprec", "rprec_sev", "kl_exact"]].mean().round(3).to_string())
print("\nwiring-agnostic and FDNA arms on identical draws")
print(gen.groupby(["variant", "N", "arm"])[["rprec", "rprec_sev", "kl_exact"]].mean().round(3).to_string())
rng = np.random.default_rng(51)
print("\nI8(rho) minus best-mean generic (I1,I2,I3) [95% cluster CI]")
for (v, N), g in m.groupby(["variant", "N"]):
    og = old[(old.variant == v) & (old.N == N) & old.arm.isin(["I1_mlp", "I2_gbm_product", "I3_minmax_gnn"])]
    ref_arm = og.groupby("arm").rprec.mean().idxmax()
    ref = og[og.arm == ref_arm].groupby("op").rprec.mean()
    out = []
    for rho, gg in g.groupby("rho"):
        d = (gg.groupby("op").rprec.mean() - ref).values
        ix = rng.integers(0, len(d), (5000, len(d))); lo, hi = np.percentile(d[ix].mean(1), [2.5, 97.5])
        out.append(f"rho={rho}: {d.mean():+.3f}[{lo:+.3f},{hi:+.3f}]")
    print(f"{v} N={N} (vs {ref_arm}):", "  ".join(out))
