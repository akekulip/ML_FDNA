"""Evaluate the H5-amended decomposition claim (registry/amendment_H5.yaml). Usage: confirm_D.py [confirm|replicate|test]
'test' = dry run on the exploratory block (NOT a confirmatory look)."""
import sys
import numpy as np
import pandas as pd
from fdna import rules

D = "data_v2"; BLOCK = sys.argv[1] if len(sys.argv) > 1 else "confirm"
TREAT, COMP, N = "D_I2_gbm_product", "A8_hybrid_generic_marginals", 100
comps, per_cell = {}, {}
for pk, var in (("P1", "v2b"), ("P2", "v2c")):
    r = pd.read_parquet(f"{D}/d_{BLOCK}_{var}_3.parquet"); r = r[r.n == N]
    a = r[r.arm == TREAT].pivot_table(index="op", columns="rep", values="rprec")
    b = r[r.arm == COMP].pivot_table(index="op", columns="rep", values="rprec")
    diff = (a - b).values
    comps[pk] = diff
    per_cell[pk] = dict(t1=rules.boot(diff, margin=rules.SESOI), t2=rules.boot(diff, margin=0.0), tost=rules.tost_negligible(diff))
    print(f"{pk}: {TREAT} - {COMP} at n={N}: mean {per_cell[pk]['t2']['mean']:+.4f}  90% interval [{per_cell[pk]['t2']['lo90']:+.4f}, {per_cell[pk]['t2']['hi90']:+.4f}]"
          f" | tier1 p={per_cell[pk]['t1']['p']:.4f} tier2 p={per_cell[pk]['t2']['p']:.4f} | negligible(+-0.02)={per_cell[pk]['tost']['negligible']}")
t1 = max(c["t1"]["p"] for c in per_cell.values()); t2 = max(c["t2"]["p"] for c in per_cell.values())
print(f"\nJOINT (intersection-union over P1,P2):")
print(f"  tier 1 (meaningful, >= 0.05): {'SUPPORTED' if t1 < 0.05 and all(c['t2']['mean'] >= rules.SESOI for c in per_cell.values()) else 'not supported'} (max p={t1:.4f})")
print(f"  tier 2 (reliable but small, > 0): {'SUPPORTED' if t2 < 0.05 else 'not supported'} (max p={t2:.4f})")
print(f"  tier 3 (negligible, within +-0.02): {'SUPPORTED' if all(c['tost']['negligible'] for c in per_cell.values()) else 'not supported'}")
if BLOCK == "test":
    print("\n(DRY RUN on the exploratory block: not a confirmatory result)")
