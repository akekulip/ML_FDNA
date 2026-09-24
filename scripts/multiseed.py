"""Seed replication with the SAME training config as tuning (params + bagging); per-operating-point metrics
for every stratum, and saved per-row predictions. Usage: multiseed.py [data_dir] [n_seeds]"""
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from fdna import baseline_data
from fdna.evalutil import check_spec_hash, op_metrics

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5


def strata(d):
    cell = d.lab.cell.values
    s = {c: cell == c for c in ("n1_seen", "n1_unseen", "n2_seen", "n2_unseen")}
    u = s["n2_unseen"]
    s["n2_unseen|familiar"], s["n2_unseen|novel"] = u & ~d.novel, u & d.novel
    for k in (2, 3, 4):
        s[f"n2_unseen|size{k}"] = u & (d.fs_size == k)
    for k in (2, 3, 4):
        s[f"n2_unseen|size{k}|novel"] = u & (d.fs_size == k) & d.novel
        s[f"n2_unseen|size{k}|familiar"] = u & (d.fs_size == k) & ~d.novel
    return {k: m & d.te for k, m in s.items()}


def evaluate(d, name, pred, seed, S):
    rows = []
    op = d.lab.op.values
    for sname, m in S.items():
        for o in np.unique(op[m]):
            mm = m & (op == o)
            rows.append(dict(model=name, seed=seed, stratum=sname, op=int(o), **op_metrics(d.y[mm], pred[mm], d.key[mm])))
    return rows


if __name__ == "__main__":
    check_spec_hash(DATA)
    info = json.load(open(f"{DATA}/baseline_info.json"))
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
    pd.DataFrame(rows).to_parquet(f"{DATA}/multiseed_results.parquet")
    np.savez_compressed(f"{DATA}/preds_seeds.npz", **preds)
    print("done", info["spec"])
