"""Phase 4: CORRECTED partial-observation validation (registry/phase4_step_partialobs_v2.yaml), fixing every
issue an external review found in scripts/p4_mobius_partialobs.py (kept, not deleted, for the record):
  1. Reproducibility: canonical-key sort of every loaded array + a per-row seed derived from (op_id, outage)
     via np.random.default_rng([op_id, *outage]) -- no shared, sequentially-advancing RNG, so results no longer
     depend on multiprocessing completion order.
  2. K_DRAWS=100 stated explicitly (registered text's "200" was inside invalid YAML, never actually registered).
  3. Both squared error (where the mean-composed reference IS optimal) and MAE (secondary) reported.
  4. Mean-composed oracle relabelled honestly; a POSTERIOR-MEDIAN-composed reference added (the MAE-optimal
     point estimate, not the mean).
  5. A "no observation, average over the PRIOR" baseline replaces the weaker "assume full control" reference.
  6. R-precision (this project's standard screening metric, src/fdna/evalutil.rprec) reported alongside MAE.
"""
import json
import numpy as np

from fdna import v2, v2data
from fdna.evalutil import rprec, scenario_uniform
from fdna import spec

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = set(manifest["op_ids"])

# canonical sort: (op_id, outage tuple) -- independent of the multiprocessing completion order the file was written in
order = sorted(range(len(k3["op_ids"])), key=lambda i: (int(k3["op_ids"][i]), tuple(int(x) for x in k3["outages"][i])))
op_ids_sorted = k3["op_ids"][order]
outages_sorted = [tuple(int(x) for x in k3["outages"][i]) for i in order]
V_sorted = k3["V"][order]

tr = v2data.load_vtable("data_v2", "train")
y1 = {}
for i, op_id in enumerate(tr["op_ids"]):
    op_id = int(op_id)
    if op_id not in OP_IDS:
        continue
    for j, (a, b) in enumerate(tr["conts"][i]):
        if a >= 0 and b < 0:
            y1[(op_id, int(a))] = tr["V"][i, j].astype(np.float64)

yp = np.load("data_hik/ypair_fullgrid.npz")
yp_order = sorted(range(len(yp["op_ids"])), key=lambda i: (int(yp["op_ids"][i]), tuple(int(x) for x in yp["pairs"][i])))
y2 = {(int(yp["op_ids"][i]), tuple(int(x) for x in yp["pairs"][i])): yp["Y"][i].astype(np.float64) for i in yp_order}

w = v2.build_world()
prior_pc = np.zeros(len(w.CV)); np.add.at(prior_pc, w.cidx, np.exp(w.logprior))   # marginal prior over controls, no evidence, cell-independent

def weighted_median(vals, weights):
    o = np.argsort(vals); v, wt = vals[o], weights[o]
    c = np.cumsum(wt); return float(v[np.searchsorted(c, 0.5 * c[-1])])

K_DRAWS = 100
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
results = {}
for cell, (q, s, cov) in CELLS.items():
    ae, se = {"g1": [], "g2": [], "prior_only": [], "mean_oracle": [], "median_oracle": []}, {"g1": [], "g2": [], "mean_oracle": []}
    rprec_rows = []
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id)
        a, b, c = outage
        y_true_grid = V_sorted[i].astype(np.float64)
        ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
        g1_grid = ya + yb + yc
        g2_grid = g1_grid + sum(Ivals)

        rng = np.random.default_rng([op_id, *outage, hash(cell) % (2**31)])   # deterministic, order-independent
        st = v2.sample_states(w, rng, K_DRAWS)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]
        true_c_idx = w.cidx[st]
        y_true = y_true_grid[true_c_idx]

        pred = {
            "g1": (pc * g1_grid).sum(1), "g2": (pc * g2_grid).sum(1),
            "prior_only": np.full(K_DRAWS, (prior_pc * g2_grid).sum()),
            "mean_oracle": (pc * y_true_grid).sum(1),
            "median_oracle": np.array([weighted_median(y_true_grid, pc[k]) for k in range(K_DRAWS)]),
        }
        for k_, v_ in pred.items():
            ae[k_].append(np.abs(y_true - v_))
            if k_ in se: se[k_].append((y_true - v_) ** 2)
        key = scenario_uniform(np.full(K_DRAWS, op_id), np.full(K_DRAWS, a), np.arange(K_DRAWS) + i * 1000, np.arange(K_DRAWS), salt=9)
        rprec_rows.append((y_true, pred["g2"], pred["g1"], key))

    ae = {k_: np.concatenate(v_) for k_, v_ in ae.items()}
    se = {k_: np.concatenate(v_) for k_, v_ in se.items()}
    yt_all = np.concatenate([r[0] for r in rprec_rows]); g2_all = np.concatenate([r[1] for r in rprec_rows])
    g1_all = np.concatenate([r[2] for r in rprec_rows]); key_all = np.concatenate([r[3] for r in rprec_rows])
    results[cell] = {
        "n_rows": len(ae["g1"]),
        "mae": {k_: float(v_.mean()) for k_, v_ in ae.items()},
        "mse": {k_: float(v_.mean()) for k_, v_ in se.items()},
        "rprec_g2": rprec(yt_all, g2_all, key_all), "rprec_g1": rprec(yt_all, g1_all, key_all),
        "note": "mean_oracle is optimal under MSE, not MAE; median_oracle is the MAE-optimal reference. prior_only = no observation, average over the communication prior (stronger than 'assume full control')."
    }
    print(cell, json.dumps(results[cell], indent=1))
json.dump(results, open("results/phase4/mobius_partialobs_v2.json", "w"), indent=1)
