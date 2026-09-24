"""Analysis of prereg/HYPOTHESIS_LC.md. Primary = fresh operating points, n2_unseen|novel, phys pair, n in {5,10,25}."""
import sys
import numpy as np
import pandas as pd

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
r = pd.read_parquet(f"{DATA}/learning_curve_results.parquet")
rng = np.random.default_rng(5)
PAIRS = [("elec+phys+ctrl", "elec+phys+raw_comm", "PRIMARY pair (physics)"), ("elec+ctrl", "elec+raw_comm", "secondary pair (no physics)")]
STRATA = ["n2_unseen|novel", "n2_unseen|familiar", "n2_unseen"]
pd.set_option("display.width", 220)

def per_op(ev, model, st, n, metric):
    d = r[(r.evalset == ev) & (r.model == model) & (r.stratum == st) & (r.n == n)]
    return d.groupby("op")[metric].mean().sort_index()

for ev in ("fresh", "exploratory"):
    print(f"\n######## evaluation set: {ev} ({r[r.evalset==ev].op.nunique()} operating points) ########")
    for st in STRATA:
        print(f"\n-- stratum {st}: mean R-precision by training operating points n (replicates and OPs averaged) --")
        print(r[(r.evalset == ev) & (r.stratum == st)].groupby(["model", "n"]).rprec.mean().unstack("n").round(3).to_string())
        for a, b, why in PAIRS:
            out = []
            for n in (5, 10, 25, 50, 100):
                dlt = (per_op(ev, a, st, n, "rprec") - per_op(ev, b, st, n, "rprec")).values
                idx = rng.integers(0, len(dlt), (10000, len(dlt)))
                lo, hi = np.percentile(np.nanmean(dlt[idx], axis=1), [2.5, 97.5])
                out.append((n, np.nanmean(dlt), lo, hi, np.nanmean(dlt) >= 0.05 and lo > 0))
            print(f"   paired (explicit - raw) {why}:", "  ".join(f"n={n}: {m:+.4f} [{lo:+.4f},{hi:+.4f}]{'*' if ok else ''}" for n, m, lo, hi, ok in out))
    if ev == "fresh":
        a, b = PAIRS[0][:2]
        ok = []
        for n in (5, 10, 25):
            dlt = (per_op(ev, a, "n2_unseen|novel", n, "rprec") - per_op(ev, b, "n2_unseen|novel", n, "rprec")).values
            idx = np.random.default_rng(99).integers(0, len(dlt), (10000, len(dlt)))
            lo = np.percentile(np.nanmean(dlt[idx], axis=1), 2.5)
            ok.append(bool(np.nanmean(dlt) >= 0.05 and lo > 0))
        print("\nH1 (all of n=5,10,25 have diff >= +0.05 and CI lower bound > 0):", "SUPPORTED" if all(ok) else "NOT SUPPORTED", ok)
        print("\nsample-size ratio (smallest n whose mean R-precision on n2_unseen|novel reaches the OTHER set's n=100 value):")
        m = r[(r.evalset == ev) & (r.stratum == "n2_unseen|novel")].groupby(["model", "n"]).rprec.mean().unstack("n")
        for a, b, why in PAIRS:
            for x, y in ((a, b), (b, a)):
                tgt = m.loc[y, 100]
                reach = [n for n in (5, 10, 25, 50, 100) if m.loc[x, n] >= tgt]
                print(f"   {x} reaches {y}@100 ({tgt:.3f}) at n =", reach[0] if reach else ">100")
