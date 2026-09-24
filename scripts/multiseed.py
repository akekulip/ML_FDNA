"""Milestone 1 baselines: lookup, and LightGBM on electrical / raw-comm / explicit-control / physics features."""
import sys, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from fdna import spec
from fdna.features import control_features, raw_comm_states, post_outage_flows
from fdna.grid import load_case30
from fdna.opgen import nominal_rating

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
SEED = 0
TEST_FRAC = 0.25
g = load_case30()
rating = nominal_rating(g, spec.RATING)
lab = pd.read_parquet(f"{DATA}/labels.parquet")
o = np.load(f"{DATA}/ops.npz")
op_idx = {int(k): i for i, k in enumerate(o["op_id"])}
fsC, fsU, fsT = control_features(np.arange(len(spec.FAILURE_SETS)))
fsR = raw_comm_states(np.arange(len(spec.FAILURE_SETS)))

# subsample test rows (unbiased within operating point)
rng = np.random.default_rng(SEED)
keep = (lab.split != "test") | (rng.random(len(lab)) < TEST_FRAC)
lab = lab[keep].reset_index(drop=True)

# physics features per unique (op, b1, b2)
t0 = time.time()
keys = lab[["op", "b1", "b2"]].drop_duplicates().reset_index(drop=True)
phys = np.zeros((len(keys), g.n_branch + 4))
for r, (op, b1, b2) in enumerate(keys.itertuples(index=False)):
    i = op_idx[op]
    removed = tuple(b for b in (b1, b2) if b >= 0)
    ratio, struct = post_outage_flows(g, o["demand"][i], o["p0"][i], removed, rating)
    phys[r, : g.n_branch] = ratio
    phys[r, g.n_branch :] = [ratio.max(), (ratio > 1).sum(), np.clip(ratio - 1, 0, None).sum(), struct]
t_phys = (time.time() - t0) / len(keys)
kidx = keys.reset_index().set_index(["op", "b1", "b2"])["index"]
row_key = kidx.reindex(pd.MultiIndex.from_frame(lab[["op", "b1", "b2"]])).values

n = len(lab)
opi = lab.op.map(op_idx).values
elec = np.hstack([o["demand"][opi], o["p0"][opi]]).astype(np.float32)
mh = np.zeros((n, g.n_branch), np.float32)
for c in ("b1", "b2"):
    v = lab[c].values
    m = v >= 0
    mh[np.flatnonzero(m), v[m]] = 1
elec = np.hstack([elec, mh, mh.sum(1, keepdims=True)])
fs = lab.fs.values
ctrl = np.hstack([fsC[fs], fsU[fs], fsT[fs]]).astype(np.float32)
raw = fsR[fs].astype(np.float32)
ph = phys[row_key].astype(np.float32)
F = {
    "elec": elec,
    "elec+raw_comm": np.hstack([elec, raw]),
    "elec+ctrl": np.hstack([elec, ctrl]),
    "elec+phys": np.hstack([elec, ph]),
    "elec+phys+raw_comm": np.hstack([elec, ph, raw]),
    "elec+phys+ctrl": np.hstack([elec, ph, ctrl]),
}
y = lab.y.values
sp = lab.split.values
tr, va, te = sp == "train", sp == "val", sp == "test"

def rprec(yt, yp, thr=spec.SEVERE):
    sev = yt > thr
    k = int(sev.sum())
    if k == 0:
        return np.nan
    top = np.argsort(-yp, kind="stable")[:k]
    return sev[top].mean()

def recall_at(yt, yp, frac, thr=spec.SEVERE):
    sev = yt > thr
    if sev.sum() == 0:
        return np.nan
    k = max(1, int(round(frac * len(yt))))
    return sev[np.argsort(-yp, kind="stable")[:k]].sum() / sev.sum()

def evaluate(name, pred):
    rows = []
    for cell in ("n1_seen", "n1_unseen", "n2_seen", "n2_unseen"):
        m = te & (lab.cell.values == cell)
        for op in np.unique(lab.op.values[m]):
            mm = m & (lab.op.values == op)
            yt, yp = y[mm], pred[mm]
            rows.append(dict(model=name, cell=cell, op=op, rprec=rprec(yt, yp),
                             r10=recall_at(yt, yp, .10), r20=recall_at(yt, yp, .20), r40=recall_at(yt, yp, .40),
                             mae=np.abs(yt - yp).mean(),
                             ap=average_precision_score(yt > spec.SEVERE, yp) if (yt > spec.SEVERE).any() else np.nan,
                             rho=spearmanr(yt, yp)[0] if yp.std() > 0 and yt.std() > 0 else np.nan))
    return rows


info = json.load(open(f"{DATA}/baseline_info.json"))["info"]
rows = []
models = ["elec", "elec+raw_comm", "elec+ctrl", "elec+phys", "elec+phys+raw_comm", "elec+phys+ctrl"]
for name in models:
    X = F[name]
    p = {k: (int(v) if k in ("num_leaves", "min_child_samples") else v) for k, v in info[name]["best"].items()}
    for seed in range(N_SEEDS):
        m = lgb.LGBMRegressor(n_estimators=400, random_state=seed, verbose=-1, n_jobs=16,
                              bagging_fraction=0.8, bagging_freq=1, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
        full = np.zeros(n); full[te] = m.predict(X[te])
        for r in evaluate(name, full):
            r["seed"] = seed
            rows.append(r)
        print(name, seed, flush=True)
pd.DataFrame(rows).to_parquet(f"{DATA}/multiseed_results.parquet")
print("done", spec.spec_hash())
