"""Paired cluster-bootstrap comparisons over test operating points (seeds averaged within operating point)."""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
r = pd.read_parquet(f"{DATA}/multiseed_results.parquet")
METRICS = ["rprec", "ap", "mae"]
PAIRS = [
    ("elec+ctrl", "elec+raw_comm", "explicit service calc vs raw states (no physics feats)"),
    ("elec+phys+ctrl", "elec+phys+raw_comm", "explicit service calc vs raw states (with physics feats)"),
    ("elec+ctrl", "elec", "value of control info (no physics)"),
    ("elec+phys+ctrl", "elec+phys", "value of control info (with physics)"),
    ("elec+phys+ctrl", "elec+ctrl", "value of physics features (with ctrl)"),
]
rng = np.random.default_rng(11)

def per_op(model, cell, metric):
    d = r[(r.model == model) & (r.cell == cell)]
    return d.groupby("op")[metric].mean().sort_index()

print("seeds:", r.seed.nunique(), "| test OPs:", r.op.nunique())
print("\n== seed-to-seed SD of the operating-point-averaged R-precision ==")
sd = (r.groupby(["model", "cell", "seed"]).rprec.mean().groupby(["model", "cell"]).std().unstack("cell")).round(4)
print(sd.to_string())
print("\n== paired differences (A - B), mean over OPs, 95% cluster-bootstrap CI over OPs; +=A better for rprec/ap, - for mae ==")
rows = []
for a, b, why in PAIRS:
    for cell in ("n2_unseen", "n2_seen", "n1_unseen", "n1_seen"):
        for m in METRICS:
            va, vb = per_op(a, cell, m), per_op(b, cell, m)
            dlt = (va - vb).values
            idx = rng.integers(0, len(dlt), (10000, len(dlt)))
            bs = np.nanmean(dlt[idx], axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            rows.append((why, a, b, cell, m, np.nanmean(dlt), lo, hi, "excludes 0" if lo > 0 or hi < 0 else "includes 0"))
out = pd.DataFrame(rows, columns=["question", "A", "B", "cell", "metric", "diff", "lo", "hi", "ci"])
out.to_csv(f"{DATA}/paired_table.csv", index=False)
pd.set_option("display.width", 250)
for m in ("rprec",):
    print(out[out.metric == m].drop(columns=["metric"]).round(4).to_string(index=False))
print("\n== ap and mae, unseen N-2 only ==")
print(out[(out.cell == "n2_unseen") & (out.metric != "rprec")].round(5).to_string(index=False))
