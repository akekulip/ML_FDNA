"""Step 4: learning curves (pre-specified in prereg/HYPOTHESIS_LC.md). Usage: learning_curve.py [data] [data_fresh]"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data
from fdna.baseline_data import evaluate, strata

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
FRESH = sys.argv[2] if len(sys.argv) > 2 else "data_fresh"
SIZES, REPS = [5, 10, 25, 50, 100], 5
MODELS = ["elec", "elec+ctrl", "elec+raw_comm", "elec+phys+ctrl", "elec+phys+raw_comm"]
if __name__ == "__main__":
    info = json.load(open(f"{DATA}/baseline_info.json"))
    bag = info["bagging"]
    d, f = baseline_data.load(DATA), baseline_data.load(FRESH)
    assert f.te.all() and not f.tr.any()
    ops_tr = np.unique(d.lab.op.values[d.tr])
    assert len(ops_tr) == 100
    S = {"exploratory": (d, strata(d)), "fresh": (f, strata(f))}
    rows = []
    for n in SIZES:
        for rep in range(REPS):
            sub = ops_tr if n == 100 else np.random.default_rng(1000 * n + rep).choice(ops_tr, n, replace=False)
            mtr = d.tr & np.isin(d.lab.op.values, sub)
            for name in MODELS:
                p = info["info"][name]["params"]
                m = lgb.LGBMRegressor(n_estimators=400, random_state=rep, verbose=-1, n_jobs=30, **bag, **p)
                m.fit(d.F[name][mtr], d.y[mtr], eval_set=[(d.F[name][d.va], d.y[d.va])],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
                for tag, (dd, SS) in S.items():
                    pr = np.full(len(dd.y), np.nan)
                    pr[dd.te] = m.predict(dd.F[name][dd.te])
                    for r in evaluate(dd, name, pr, rep, SS):
                        r.update(n=n, rep=rep, evalset=tag)
                        rows.append(r)
            print("n", n, "rep", rep, flush=True)
    pd.DataFrame(rows).to_parquet(f"{DATA}/learning_curve_results.parquet")
    print("done")
