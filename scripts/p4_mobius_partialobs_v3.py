"""Phase 4: partial-observation validation, v3 -- fixes two more issues an external review found in v2
(kept, not deleted, for the record):
  1. v2's per-row seed used hash(cell) % (2**31); Python's built-in hash() of a string is process-randomised
     (PYTHONHASHSEED) unless fixed at interpreter startup, so v2's "deterministic" seed was NOT stable across
     processes (verified: 3 different values in 3 fresh `python3 -c` calls). Replaced with a fixed integer per
     cell (CELL_SEED below), no hash() anywhere.
  2. v2 pooled all (op, triple, draw) rows into one array before calling rprec() once -- this is NOT the
     per-operating-point ranking convention this project uses everywhere else (evalutil.op_metrics groups by
     op). Fixed: rprec computed WITHIN each operating point's own rows, then averaged across the 20 ops, with
     a paired cluster bootstrap (src/fdna/rules.boot) on the per-op (g2-g1) difference -- the project's
     standard confirmatory statistic.
  3. prior-only baseline now reported under MSE and R-precision too, not just MAE.
"""
import json
import numpy as np

from fdna import v2, v2data
from fdna.evalutil import rprec, scenario_uniform
from fdna.rules import boot

CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}   # fixed, not hash() -- stable across processes (the bug v2 had)

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = sorted(set(manifest["op_ids"]))

order = sorted(range(len(k3["op_ids"])), key=lambda i: (int(k3["op_ids"][i]), tuple(int(x) for x in k3["outages"][i])))
op_ids_sorted = k3["op_ids"][order]
outages_sorted = [tuple(int(x) for x in k3["outages"][i]) for i in order]
V_sorted = k3["V"][order]

tr = v2data.load_vtable("data_v2", "train")
y1 = {}
for i, op_id in enumerate(tr["op_ids"]):
    op_id = int(op_id)
    if op_id not in OP_IDS: continue
    for j, (a, b) in enumerate(tr["conts"][i]):
        if a >= 0 and b < 0: y1[(op_id, int(a))] = tr["V"][i, j].astype(np.float64)

yp = np.load("data_hik/ypair_fullgrid.npz")
y2 = {(int(yp["op_ids"][i]), tuple(int(x) for x in yp["pairs"][i])): yp["Y"][i].astype(np.float64) for i in range(len(yp["op_ids"]))}

w = v2.build_world()
prior_pc = np.zeros(len(w.CV)); np.add.at(prior_pc, w.cidx, np.exp(w.logprior))

def weighted_median(vals, weights):
    o = np.argsort(vals); v, wt = vals[o], weights[o]
    c = np.cumsum(wt); return float(v[np.searchsorted(c, 0.5 * c[-1])])

K_DRAWS = 100
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
results = {}
for cell, (q, s, cov) in CELLS.items():
    per_op = {op: {"g1": [], "g2": [], "prior_only": [], "mean_oracle": [], "median_oracle": [], "y_true": [], "key": []} for op in OP_IDS}
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id)
        a, b, c = outage
        y_true_grid = V_sorted[i].astype(np.float64)
        ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
        g1_grid = ya + yb + yc
        g2_grid = g1_grid + sum(Ivals)

        rng = np.random.default_rng([op_id, *outage, CELL_SEED[cell]])
        st = v2.sample_states(w, rng, K_DRAWS)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]
        true_c_idx = w.cidx[st]
        y_true = y_true_grid[true_c_idx]

        d = per_op[op_id]
        d["g1"].append((pc * g1_grid).sum(1)); d["g2"].append((pc * g2_grid).sum(1))
        d["prior_only"].append(np.full(K_DRAWS, (prior_pc * g2_grid).sum()))
        d["mean_oracle"].append((pc * y_true_grid).sum(1))
        d["median_oracle"].append(np.array([weighted_median(y_true_grid, pc[k]) for k in range(K_DRAWS)]))
        d["y_true"].append(y_true)
        d["key"].append(scenario_uniform(np.full(K_DRAWS, op_id), np.full(K_DRAWS, a), np.arange(K_DRAWS) + i * 1000, np.arange(K_DRAWS), salt=9))

    per_op_metrics = {}
    for op in OP_IDS:
        d = {k_: np.concatenate(v_) for k_, v_ in per_op[op].items()}
        yt = d["y_true"]
        mae = {k_: float(np.abs(yt - d[k_]).mean()) for k_ in ("g1", "g2", "prior_only", "mean_oracle", "median_oracle")}
        mse = {k_: float(((yt - d[k_]) ** 2).mean()) for k_ in ("g1", "g2", "prior_only", "mean_oracle")}
        rp = {k_: rprec(yt, d[k_], d["key"]) for k_ in ("g1", "g2", "prior_only")}
        per_op_metrics[op] = {"mae": mae, "mse": mse, "rprec": rp}

    ops_arr = np.array(OP_IDS)
    diff_rprec = np.array([per_op_metrics[o]["rprec"]["g2"] - per_op_metrics[o]["rprec"]["g1"] for o in OP_IDS])
    diff_mae = np.array([per_op_metrics[o]["mae"]["g1"] - per_op_metrics[o]["mae"]["g2"] for o in OP_IDS])
    boot_rprec = boot(diff_rprec, margin=0.0)
    boot_mae = boot(diff_mae, margin=0.0)
    results[cell] = {
        "n_ops": len(OP_IDS),
        "mean_over_ops": {
            "mae": {k_: float(np.mean([per_op_metrics[o]["mae"][k_] for o in OP_IDS])) for k_ in ("g1", "g2", "prior_only", "mean_oracle", "median_oracle")},
            "mse": {k_: float(np.mean([per_op_metrics[o]["mse"][k_] for o in OP_IDS])) for k_ in ("g1", "g2", "prior_only", "mean_oracle")},
            "rprec": {k_: float(np.mean([per_op_metrics[o]["rprec"][k_] for o in OP_IDS])) for k_ in ("g1", "g2", "prior_only")},
        },
        "paired_bootstrap_g2_minus_g1": {"rprec_diff": boot_rprec, "mae_improvement": boot_mae},
        "per_op": {int(o): per_op_metrics[o] for o in OP_IDS},
    }
    print(cell, json.dumps({k_: v_ for k_, v_ in results[cell].items() if k_ != "per_op"}, indent=1))
json.dump(results, open("results/phase4/mobius_partialobs_v3.json", "w"), indent=1)
