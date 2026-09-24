import glob, sys
import numpy as np
import pandas as pd
D = "data_v2"; TEST = sys.argv[1] if len(sys.argv) > 1 else "test"
d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{D}/d_{TEST}_*_*.parquet"))])
b = None
rng = np.random.default_rng(31)
allr = pd.concat([d, b]) if b is not None else d
print(f"Decomposed composition vs end-to-end, block '{TEST}': mean per-op R-precision")
print(allr.groupby(["variant", "q", "s", "n", "arm"]).rprec.mean().unstack("arm").round(3).T.to_string())
print("\ntop-16 posterior mass covered (mean):", d.groupby("arm").top16_mass.mean().round(3).to_dict())
print("\npaired (per-op, reps averaged) with 95% cluster CI")
e2e = ["A1_gbm_obs", "A2_gbm_logic", "A8_hybrid_generic_marginals"]
dl_ = ["D_I1_mlp", "D_I2_gbm_product", "D_I3_minmax_gnn", "D_I5_fdna", "D_I6_shuffled", "D_I7_unconstrained"]
for (var, q, s, n), g in allr.groupby(["variant", "q", "s", "n"]):
    pv = g.groupby(["op", "arm"]).rprec.mean().unstack("arm")
    if not set(dl_).issubset(pv.columns):
        continue
    def ci(a, b):
        x = (a - b).values; ix = rng.integers(0, len(x), (5000, len(x))); lo, hi = np.percentile(x[ix].mean(1), [2.5, 97.5]); return f"{x.mean():+.3f}[{lo:+.3f},{hi:+.3f}]"
    best_d, best_e = pv[dl_].mean().idxmax(), pv[e2e].mean().idxmax()
    print(f"{var} q={q} s={s} n={n}: best D ({best_d}) - best end-to-end ({best_e}) = {ci(pv[best_d], pv[best_e])} | D_I5 - best(D_I1,D_I2,D_I3) = {ci(pv['D_I5_fdna'], pv[['D_I1_mlp','D_I2_gbm_product','D_I3_minmax_gnn']].max(axis=1))} | D_I5 - D_I6 {ci(pv['D_I5_fdna'], pv['D_I6_shuffled'])} | D_I5 - D_I7 {ci(pv['D_I5_fdna'], pv['D_I7_unconstrained'])} | D_exact - best e2e {ci(pv['D_exact'], pv[best_e])}")
