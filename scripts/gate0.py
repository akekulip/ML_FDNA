"""Gate 0 / generator report from labels.parquet (no ML)."""
import sys
import numpy as np
import pandas as pd
from fdna import comm, spec

d = pd.read_parquet(sys.argv[1] + "/labels.parquet")
print("spec hash", (open(sys.argv[1] + "/spec_hash.txt").read()), "| rows", len(d))
CTRL = np.array([comm.control_fraction(comm.state_of(f)) for f in spec.FAILURE_SETS])
print("\n== diversity counts (plan requirement) ==")
print("distinct communication configurations (failure sets |F|<=4):", len(spec.FAILURE_SETS))
print("distinct feasible control configurations (commandable-fraction vectors):", len(np.unique(CTRL.round(6), axis=0)))
print("failure sets per split:", {k: int((spec.FS_SPLIT == k).sum()) for k in ("train", "val", "test")})
tr_c = np.unique(CTRL[spec.fs_ids("train")].round(6), axis=0)
te_c = np.unique(CTRL[spec.fs_ids("test")].round(6), axis=0)
seen = {tuple(r) for r in tr_c}
print("control vectors in train:", len(tr_c), "| in test:", len(te_c),
      "| test control vectors never seen in train:", sum(tuple(r) not in seen for r in te_c))
fs_seen = d[d.split == "test"].groupby("fs").size().index.values
print("test failure sets whose control vector is also in train (service-combo unseen, control-set seen):",
      int(np.mean([tuple(CTRL[i].round(6)) in seen for i in spec.fs_ids("test")]) * 100), "%")

print("\n== label balance per cell (all comm states pooled) ==")
g = d.groupby(["split", "cell"]).y
print(pd.DataFrame({"n": g.size(), "shed>0": g.apply(lambda s: (s > 1e-6).mean()), "severe>1%": g.apply(lambda s: (s > spec.SEVERE).mean()),
                    "struct>0": d.groupby(["split", "cell"]).y_struct.apply(lambda s: (s > 0).mean())}).round(3))

print("\n== control sensitivity per (op, contingency) group ==")
for cell in ["n1_seen", "n2_unseen"]:
    sub = d[(d.split == "test") & (d.cell == cell)]
    grp = sub.groupby(["op", "b1", "b2"]).y
    rng_ = grp.max() - grp.min()
    nd = grp.apply(lambda s: len(np.unique(s.round(6))))
    print(cell, f"| groups {len(rng_)} | sensitive (range>{spec.TAU}) {np.mean(rng_ > spec.TAU):.2f} | >=3 distinct y {np.mean(nd >= 3):.2f}"
          f" | y_full-analogue zero {np.mean(grp.min() < 1e-6):.2f}")
