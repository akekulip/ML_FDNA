"""Paired cluster-bootstrap comparisons over test operating points (seeds averaged within operating point)."""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
r = pd.read_parquet(f"{DATA}/multiseed_results.parquet")
STRATA = ["n2_unseen", "n2_unseen|familiar", "n2_unseen|novel", "n2_seen", "n1_unseen", "n1_seen"]
METRICS = ["rprec", "r20", "ap", "mae"]
PAIRS = [
    ("elec+ctrl", "elec+raw_comm", "explicit service calc vs raw states (no physics)"),
    ("elec+phys+ctrl", "elec+phys+raw_comm", "explicit service calc vs raw states (with physics)"),
    ("elec+ctrl", "elec", "control info (no physics)"),
    ("elec+phys+ctrl", "elec+phys", "control info (with physics)"),
    ("elec+phys+ctrl", "elec+ctrl", "physics features (with ctrl)"),
]
rng = np.random.default_rng(11)
pd.set_option("display.width", 250)


def per_op(model, stratum, metric):
    d = r[(r.model == model) & (r.stratum == stratum)]
    return d.groupby("op")[metric].mean().sort_index()


print("seeds:", r[r.model != "zero"].seed.nunique(), "| test OPs:", r.op.nunique())
print("\n== stratum prevalence of severe (mean over OPs) and zero-baseline R-precision (should be ~prevalence) ==")
z = r[r.model == "zero"].groupby("stratum")[["prev", "rprec"]].mean().reindex(STRATA)
print(z.round(3).to_string())
print("\n== seed SD of the OP-averaged R-precision, unseen N-2 ==")
print(r[(r.stratum == "n2_unseen") & (r.model != "zero")].groupby(["model", "seed"]).rprec.mean().groupby("model").std().round(4).to_string())
rows = []
for a, b, why in PAIRS:
    for st in STRATA:
        for m in METRICS:
            dlt = (per_op(a, st, m) - per_op(b, st, m)).values
            idx = rng.integers(0, len(dlt), (10000, len(dlt)))
            lo, hi = np.percentile(np.nanmean(dlt[idx], axis=1), [2.5, 97.5])
            rows.append((why, a, b, st, m, np.nanmean(dlt), lo, hi, "excludes 0" if lo > 0 or hi < 0 else "includes 0"))
out = pd.DataFrame(rows, columns=["question", "A", "B", "stratum", "metric", "diff", "lo", "hi", "ci"])
out.to_csv(f"{DATA}/paired_table.csv", index=False)
for m in ("rprec", "r20"):
    print(f"\n== paired differences A-B in {m}: mean over OPs, 95% cluster-bootstrap CI over OPs ==")
    print(out[out.metric == m].drop(columns=["metric", "A", "B"]).round(4).to_string(index=False))
print("\n== models, per stratum (mean over OPs and seeds) ==")
tab = r[r.model != "zero"].groupby(["stratum", "model"])[["rprec", "r20", "ap", "mae", "prev"]].mean().round(4)
print(tab.loc[STRATA].to_string())
