"""Seed replication with the SAME training config as tuning (params + bagging); per-operating-point metrics
for every stratum, and saved per-row predictions. Usage: multiseed.py [data_dir] [n_seeds]"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data
from fdna.baseline_data import evaluate, strata
from fdna.evalutil import check_spec_hash

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


if __name__ == "__main__":
    check_spec_hash(DATA)
    info = json.load(open(f"{DATA}/baseline_info.json"))
    from fdna import spec
    assert info["spec"] == spec.spec_hash(), "tuned parameters were produced under a different spec"
    bag = info["bagging"]
    d = baseline_data.load(DATA)
    S = strata(d)
    rows, preds = [], {}
    for name, X in d.F.items():
        p = info["info"][name]["params"]
        for seed in range(N_SEEDS):
            m = lgb.LGBMRegressor(n_estimators=400, random_state=seed, verbose=-1, n_jobs=16, **bag, **p)
            m.fit(X[d.tr], d.y[d.tr], eval_set=[(X[d.va], d.y[d.va])], callbacks=[lgb.early_stopping(30, verbose=False)])
            pr = np.full(len(d.y), np.nan, np.float32)
            pr[d.te] = m.predict(X[d.te])
            preds[f"{name}|{seed}"] = pr
            rows += evaluate(d, name, pr, seed, S)
            print(name, seed, flush=True)
    # baselines without training
    z = np.zeros(len(d.y))
    rows += evaluate(d, "zero", z, 0, S)
    res = pd.DataFrame(rows)
    res["spec"] = spec.spec_hash()
    res.to_parquet(f"{DATA}/multiseed_results.parquet")
    np.savez_compressed(f"{DATA}/preds_seeds.npz", **preds)
    print("done", info["spec"])
