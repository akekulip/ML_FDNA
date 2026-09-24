"""POST-HOC exploratory diagnostics of the learning-curve result (not pre-specified).
Q1: is the label a function of the explicit control vector alone (information sufficiency)?
Q2: does the raw-vs-explicit gap survive when both use identical hyper-parameters?
Q3: which part of the explicit set (fractions C, unit reachability U, total T) carries the deficit?"""
import json, sys
import lightgbm as lgb
import numpy as np
import pandas as pd
from fdna import baseline_data
from fdna.baseline_data import strata
from fdna.evalutil import op_metrics

d, f = baseline_data.load("data"), baseline_data.load("data_fresh")
info = json.load(open("data/baseline_info.json")); bag = info["bagging"]

# Q1: sufficiency on the stored labels
lab = pd.read_parquet("data/labels.parquet", columns=["op", "b1", "b2", "fs", "y"])
from fdna import comm, spec
C = np.array([comm.control_fraction(comm.state_of(x)) for x in spec.FAILURE_SETS]).round(6)
key = pd.Series([hash(tuple(r)) for r in C])[lab.fs.values].values
lab["ck"] = key
g = lab.groupby(["op", "b1", "b2", "ck"]).y.nunique()
print("Q1: (op, outage, control vector) groups:", len(g), "| groups with >1 distinct y:", int((g > 1).sum()),
      "| failure sets per control vector: median", int(pd.Series(key).value_counts().median()))

def cols(D, name):
    X = D.F["elec+phys+ctrl"]; base = X[:, :123]
    parts = {"C": X[:, 123:128], "U": X[:, 128:138], "T": X[:, 138:139]}
    raw = D.F["elec+phys+raw_comm"][:, 123:]
    if name == "raw": return np.hstack([base, raw])
    if name == "ctrl": return X
    if name == "raw+ctrl": return np.hstack([X, raw])
    return np.hstack([base] + [parts[c] for c in name])

SETS = ["raw", "ctrl", "raw+ctrl", "U", "C", "CT"]
pars = {"own": None, "ctrl_params": info["info"]["elec+phys+ctrl"]["params"], "raw_params": info["info"]["elec+phys+raw_comm"]["params"]}
mask = f.te & (f.lab.cell.values == "n2_unseen") & f.novel
ops_tr = np.unique(d.lab.op.values[d.tr]); opf = f.lab.op.values
rows = []
Xd = {s: cols(d, s) for s in SETS}; Xf = {s: cols(f, s) for s in SETS}
for n in (5, 10, 25):
    for rep in range(5):
        sub = np.random.default_rng(1000 * n + rep).choice(ops_tr, n, replace=False)
        mtr = d.tr & np.isin(d.lab.op.values, sub)
        for s in SETS:
            for pn, pv in pars.items():
                if pn == "own":
                    pv = info["info"]["elec+phys+raw_comm" if s in ("raw", "raw+ctrl") else "elec+phys+ctrl"]["params"]
                m = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=30, **bag, **pv)
                m.fit(Xd[s][mtr], d.y[mtr], eval_set=[(Xd[s][d.va], d.y[d.va])], callbacks=[lgb.early_stopping(30, verbose=False)])
                pr = np.full(len(f.y), np.nan); pr[mask] = m.predict(Xf[s][mask])
                per = [op_metrics(f.y[mask & (opf == o)], pr[mask & (opf == o)], f.key[mask & (opf == o)])["rprec"] for o in np.unique(opf[mask])]
                rows.append(dict(n=n, rep=rep, feat=s, params=pn, rprec=float(np.mean(per))))
        print("n", n, "rep", rep, flush=True)
r = pd.DataFrame(rows); r.to_parquet("data/lc_diagnostics.parquet")
print("\nQ2/Q3: mean R-precision, fresh novel N-2 (mean over 5 replicates and 80 OPs), by feature set and hyper-parameter source")
print(r.groupby(["params", "feat", "n"]).rprec.mean().unstack("n").round(3).to_string())
