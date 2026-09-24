"""Tune tree baselines (one shared training config, incl. bagging), save params and per-row predictions.
Usage: run_baselines.py [data_dir] [n_trials]"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data, spec
from fdna.evalutil import op_metrics

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
N_TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
BAG = dict(bagging_fraction=0.8, bagging_freq=1)  # same in tuning and seed runs
d = baseline_data.load(DATA)
tr, va, te, y = d.tr, d.va, d.te, d.y
r = np.random.default_rng(1)
grid = [dict(num_leaves=int(a), min_child_samples=int(b), learning_rate=0.1, feature_fraction=float(c), reg_lambda=float(e))
        for a, b, c, e in zip(r.choice([15, 31, 63, 127], N_TRIALS), r.choice([20, 50, 100, 200], N_TRIALS),
                              r.choice([0.5, 0.8, 1.0], N_TRIALS), r.choice([0.0, 1.0, 10.0], N_TRIALS))]
info, preds = {}, {}
for name, X in d.F.items():
    best, t0 = None, time.time()
    for p in grid:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=0, verbose=-1, n_jobs=30, **BAG, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
        s = float(((m.predict(X[va]) - y[va]) ** 2).mean())
        if best is None or s < best[0]:
            best = (s, m, p)
    p_ = np.full(len(y), np.nan)
    t1 = time.time()
    p_[te] = best[1].predict(X[te])
    info[name] = dict(val_mse=best[0], params=best[2], n_trees=int(best[1].n_estimators_), tune_s=t1 - t0,
                      infer_us_per_scenario=(time.time() - t1) / te.sum() * 1e6, n_features=X.shape[1])
    preds[name] = p_
    print(name, info[name], flush=True)
np.savez_compressed(f"{DATA}/preds_tuned.npz", **preds)
json.dump(dict(info=info, bagging=BAG, spec=spec.spec_hash(), doc=spec.doc_hash()), open(f"{DATA}/baseline_info.json", "w"), indent=1)
print("done")
