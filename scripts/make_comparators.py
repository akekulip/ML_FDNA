"""Fix the comparator arms from the re-run SCREEN (exploratory block) and write registry/comparators.yaml. Run once, commit BEFORE any
confirm run. Selection uses only screen data at the pre-declared primary sizes (INF: N=5000; D/DS: n=25)."""
import yaml
import pandas as pd

D = "data_v2"
out = {}
GEN = ["I1_mlp", "I2_gbm_product", "I3_minmax_gnn"]
for pk, var in (("P1", "v2b"), ("P2", "v2c")):
    i = pd.read_parquet(f"{D}/inf_test_{var}_3.parquet"); i = i[(i.N == 5000) & (i.arm != "oracle_exact_posterior")]
    m = i.groupby("arm").rprec.mean()
    out[f"INF_{pk}"] = dict(treatment="I5_fdna", comparator=max(GEN, key=lambda a: m[a]), controls=["I6_shuffled", "I7_unconstrained"], size="N=5000",
                            screen_means={k: round(float(v), 4) for k, v in m.items()})
    d = pd.read_parquet(f"{D}/d_test_{var}_3.parquet"); d = d[d.n == 25]
    m = d.groupby("arm").rprec.mean()
    dl = [a for a in m.index if a.startswith("D_I")]
    e2e = ["A1_gbm_obs", "A2_gbm_logic", "A8_hybrid_generic_marginals"]
    out[f"D_{pk}"] = dict(treatment=max(dl, key=lambda a: m[a]), comparator=max(e2e, key=lambda a: m[a]), size="n=25",
                          screen_means={k: round(float(v), 4) for k, v in m.items()})
    out[f"DS_{pk}"] = dict(treatment="D_I5_fdna", comparator=max(["D_I1_mlp", "D_I2_gbm_product", "D_I3_minmax_gnn"], key=lambda a: m[a]),
                           controls=["D_I6_shuffled", "D_I7_unconstrained"], size="n=25")
yaml.safe_dump(out, open("registry/comparators.yaml", "w"), sort_keys=False)
print(yaml.safe_dump(out, sort_keys=False))
