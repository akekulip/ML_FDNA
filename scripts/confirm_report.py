"""Evaluate the six declared confirmatory claims with the pre-declared rules (src/fdna/rules.py). Usage: confirm_report.py [confirm|replicate]"""
import sys
import numpy as np
import pandas as pd
import yaml
from fdna import rules

D = "data_v2"; BLOCK = sys.argv[1] if len(sys.argv) > 1 else "confirm"
cmp_ = yaml.safe_load(open("registry/comparators.yaml"))
claims = {}
for slot, spec_ in cmp_.items():
    pk = slot.split("_")[1]; var = "v2b" if pk == "P1" else "v2c"
    kind = slot.split("_")[0]
    f = f"{D}/{'inf' if kind == 'INF' else 'd'}_{BLOCK}_{var}_3.parquet"
    r = pd.read_parquet(f)
    size_col, size_val = ("N", int(spec_["size"].split("=")[1])) if kind == "INF" else ("n", int(spec_["size"].split("=")[1]))
    r = r[r[size_col] == size_val]
    pv = r.groupby(["op", "arm"]).rprec.mean().unstack("arm")
    comp = {"vs_comparator": (pv[spec_["treatment"]] - pv[spec_["comparator"]]).values}
    for c in spec_.get("controls", []):
        comp[f"vs_{c}"] = (pv[spec_["treatment"]] - pv[c]).values
    claims[slot] = rules.claim(comp, "vs_comparator")
v = rules.verdicts(claims)
for slot, x in v.items():
    print(f"{slot}: treatment {cmp_[slot]['treatment']} vs comparator {cmp_[slot]['comparator']} ({cmp_[slot]['size']}) -> "
          f"{'SUPPORTED' if x['supported'] else 'not supported'} | max-p {x['p']:.4f} Holm-adj {x['padj']:.4f} effect_ok {x['effect_ok']}")
    for k, (m, p, lo, hi) in x["stats"].items():
        print(f"    {k}: {m:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.4f}")
