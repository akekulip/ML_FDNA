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

results = []
# lookups
results += evaluate("zero", np.zeros(n))
key = pd.Series([tuple(r) for r in fsC[fs].round(6)])
mean_by_ctrl = pd.Series(y[tr]).groupby(key[tr].values).mean()
results += evaluate("lookup(control vector)", key.map(mean_by_ctrl).fillna(y[tr].mean()).values)
# GBTs
grid = [dict(num_leaves=nl, min_child_samples=mc, learning_rate=0.1, feature_fraction=ff, reg_lambda=rl)
        for nl, mc, ff, rl in zip(*[np.random.default_rng(1).choice(v, 12) for v in
                                     ([15, 31, 63, 127], [20, 50, 100, 200], [0.5, 0.8, 1.0], [0.0, 1.0, 10.0])])]
info = {}
for name, X in F.items():
    best = None
    t0 = time.time()
    for p in grid:
        m = lgb.LGBMRegressor(n_estimators=400, random_state=SEED, verbose=-1, n_jobs=30, **p)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
        s = ((m.predict(X[va]) - y[va]) ** 2).mean()
        if best is None or s < best[0]:
            best = (s, m, p)
    m = best[1]
    t1 = time.time()
    pred = m.predict(X[te])
    infer_us = (time.time() - t1) / te.sum() * 1e6
    full = np.zeros(n); full[te] = pred
    results += evaluate(name, full)
    info[name] = dict(val_mse=best[0], best=best[2], tune_s=time.time() - t0, infer_us_per_scenario=infer_us,
                      n_features=X.shape[1], n_trees=m.n_estimators_)
    print(name, info[name], flush=True)
info["physics_feature_ms_per_(op,contingency)"] = t_phys * 1e3
res = pd.DataFrame(results)
res.to_parquet(f"{DATA}/baseline_results.parquet")
json.dump(dict(info=info, spec=spec.spec_hash()), open(f"{DATA}/baseline_info.json", "w"), indent=1, default=float)

# cluster bootstrap over operating points
B = np.random.default_rng(7)
def boot(v, ops):  # v per op
    idx = B.integers(0, len(v), (2000, len(v)))
    return np.nanmean(v[idx], axis=1)
print("\nseverity R-precision (mean over test OPs, 95% cluster-bootstrap CI)")
tab = []
for (model, cell), gdf in res.groupby(["model", "cell"]):
    v = gdf.sort_values("op").rprec.values
    b = boot(v, None)
    tab.append((cell, model, np.nanmean(v), np.percentile(b, 2.5), np.percentile(b, 97.5), gdf.r10.mean(), gdf.ap.mean(), gdf.rho.mean(), gdf.mae.mean()))
out = pd.DataFrame(tab, columns=["cell", "model", "rprec", "lo", "hi", "recall@10%", "AP", "spearman", "MAE"]).round(3)
print(out.sort_values(["cell", "rprec"], ascending=[True, False]).to_string(index=False))
out.to_csv(f"{DATA}/baseline_table.csv", index=False)
