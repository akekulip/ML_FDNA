"""Reproduces the benchmark-composition diagnostics quoted in the reports (no models)."""
import sys
import numpy as np
import pandas as pd
from fdna import comm, spec

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
d = pd.read_parquet(f"{DATA}/labels.parquet", columns=["split", "cell", "fs", "y", "op"])
C = np.array([comm.control_fraction(comm.state_of(f)) for f in spec.FAILURE_SETS])
key = np.array([hash(tuple(r.round(6))) for r in C])
train_keys = list(set(key[spec.fs_ids("train")]))
size = np.array([len(f) for f in spec.FAILURE_SETS])
d["novel"] = ~np.isin(key[d.fs.values], train_keys)
d["loss"] = C[d.fs.values].sum(1) == 0
d["size"] = size[d.fs.values]
print("failure sets", len(spec.FAILURE_SETS), "| distinct control vectors", len(set(key)),
      "| train failure sets", len(spec.fs_ids("train")), "| train control vectors", len(train_keys))
t = d[d.split == "test"]
print("\ncell: complete-control-loss share | novel-control-vector share | severe prevalence | size mix")
for c, g in t.groupby("cell"):
    print(f"{c}: loss={g.loss.mean():.3f} novel={g.novel.mean():.3f} severe={(g.y > spec.SEVERE).mean():.3f}",
          {k: round(v, 2) for k, v in g["size"].value_counts(normalize=True).sort_index().items()})
u = t[t.cell == "n2_unseen"]
print("\nn2_unseen: severe prevalence by (failure size, novel control vector)")
g = u.groupby(["size", "novel"]).agg(rows=("y", "size"), severe=("y", lambda s: (s > spec.SEVERE).mean()))
g["share"] = g.rows / g.rows.sum()
print(g.round(3).to_string())
print("\nunseen-N-2 rows whose control vector occurs in train:", round(1 - u.novel.mean(), 3))
