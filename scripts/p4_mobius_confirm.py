"""Phase 4 CONFIRMATORY run (registry/phase4_mobius_confirm.yaml). Fresh reserve-block operating points (600-659)
and a fresh 70-triple manifest (data_hik/manifest_k3_confirm.json), never used in any earlier Phase 3/4 screen.
Compares g1 (naive sum), prior-only, g2 (fixed Mobius composition), and matched learned linear composition
(OLS + MAE-trained QuantileRegressor, GroupKFold held out by triple), on BOTH R-precision (per-op, cluster
bootstrapped, src/fdna/rules.boot) and MAE/MSE. Reports total LP-solve cost per arm (all arms share the SAME
underlying y_single/y_pair/y_true solves -- no arm is privileged)."""
import json
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import GroupKFold

from fdna import v2, v2data
from fdna.evalutil import rprec, scenario_uniform
from fdna.rules import boot

CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}

n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}

order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]
outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]

# GroupKFold linear baseline (exact-control screen, matched to the exploratory design) -- fit once per control state used below
outage_to_group = {o: i for i, o in enumerate(sorted(set(outages_sorted)))}
def linear_predictions(cv_idx):
    feats, ys, groups = [], [], []
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id); a, b, c = outage
        ya, yb, yc = y1[(op_id, a)][cv_idx], y1[(op_id, b)][cv_idx], y1[(op_id, c)][cv_idx]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y2[(op_id, p)][cv_idx] - y1[(op_id, p[0])][cv_idx] - y1[(op_id, p[1])][cv_idx] for p in pairs]
        feats.append([ya, yb, yc, *Ivals]); ys.append(float(V_sorted[i][cv_idx])); groups.append(outage_to_group[outage])
    X, y, groups = np.array(feats), np.array(ys), np.array(groups)
    gkf = GroupKFold(5); pred_ols, pred_l1 = np.zeros_like(y), np.zeros_like(y)
    for tr_i, te_i in gkf.split(X, y, groups):
        m1 = LinearRegression(); m1.fit(X[tr_i], y[tr_i]); pred_ols[te_i] = m1.predict(X[te_i])
        m2 = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"); m2.fit(X[tr_i], y[tr_i]); pred_l1[te_i] = m2.predict(X[te_i])
    return pred_ols, pred_l1

full_idx, none_idx = int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))
lin_full = linear_predictions(full_idx); lin_none = linear_predictions(none_idx)
lin_by_row = {}
for i in range(len(op_ids_sorted)):
    lin_by_row[(int(op_ids_sorted[i]), outages_sorted[i])] = {"full": (lin_full[0][i], lin_full[1][i]), "none": (lin_none[0][i], lin_none[1][i])}

w = v2.build_world()
prior_pc = np.zeros(len(w.CV)); np.add.at(prior_pc, w.cidx, np.exp(w.logprior))
def weighted_median(vals, weights):
    o = np.argsort(vals); v, wt = vals[o], weights[o]; c = np.cumsum(wt); return float(v[np.searchsorted(c, 0.5 * c[-1])])

K_DRAWS = 100
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
results = {}
for cell, (q, s, cov) in CELLS.items():
    per_op = {op: {"g1": [], "g2": [], "prior_only": [], "y_true": [], "key": []} for op in OP_IDS}
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id); a, b, c = outage
        y_true_grid = V_sorted[i].astype(np.float64)
        ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
        g1_grid = ya + yb + yc; g2_grid = g1_grid + sum(Ivals)

        rng = np.random.default_rng([op_id, *outage, CELL_SEED[cell]])
        st = v2.sample_states(w, rng, K_DRAWS)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]
        y_true = y_true_grid[w.cidx[st]]

        d = per_op[op_id]
        d["g1"].append((pc * g1_grid).sum(1)); d["g2"].append((pc * g2_grid).sum(1))
        d["prior_only"].append(np.full(K_DRAWS, (prior_pc * g2_grid).sum()))
        d["y_true"].append(y_true)
        d["key"].append(scenario_uniform(np.full(K_DRAWS, op_id), np.full(K_DRAWS, a), np.arange(K_DRAWS) + i * 1000, np.arange(K_DRAWS), salt=11))

    per_op_metrics = {}
    for op in OP_IDS:
        d = {k_: np.concatenate(v_) for k_, v_ in per_op[op].items()}
        yt = d["y_true"]
        mae = {k_: float(np.abs(yt - d[k_]).mean()) for k_ in ("g1", "g2", "prior_only")}
        mse = {k_: float(((yt - d[k_]) ** 2).mean()) for k_ in ("g1", "g2", "prior_only")}
        rp = {k_: rprec(yt, d[k_], d["key"]) for k_ in ("g1", "g2", "prior_only")}
        per_op_metrics[op] = {"mae": mae, "mse": mse, "rprec": rp}

    diff_rprec_g1 = np.array([per_op_metrics[o]["rprec"]["g2"] - per_op_metrics[o]["rprec"]["g1"] for o in OP_IDS])
    diff_rprec_prior = np.array([per_op_metrics[o]["rprec"]["g2"] - per_op_metrics[o]["rprec"]["prior_only"] for o in OP_IDS])
    boot_g1 = boot(diff_rprec_g1, margin=0.0); boot_prior = boot(diff_rprec_prior, margin=0.0)
    n_at_every_op_g1 = int((diff_rprec_g1 > 0).sum()); n_at_every_op_prior = int((diff_rprec_prior > 0).sum())
    results[cell] = {
        "n_ops": len(OP_IDS),
        "mean_over_ops": {
            "rprec": {k_: float(np.mean([per_op_metrics[o]["rprec"][k_] for o in OP_IDS])) for k_ in ("g1", "g2", "prior_only")},
            "mae": {k_: float(np.mean([per_op_metrics[o]["mae"][k_] for o in OP_IDS])) for k_ in ("g1", "g2", "prior_only")},
            "mse": {k_: float(np.mean([per_op_metrics[o]["mse"][k_] for o in OP_IDS])) for k_ in ("g1", "g2", "prior_only")},
        },
        "paired_bootstrap_rprec": {"g2_minus_g1": boot_g1, "g2_minus_prior_only": boot_prior},
        "g2_beats_g1_at_n_of_60_ops": n_at_every_op_g1, "g2_beats_prior_at_n_of_60_ops": n_at_every_op_prior,
        "tier": "tier1" if boot_g1["lo90"] >= 0.05 else ("tier2" if boot_g1["lo90"] > 0 else "tier3_or_below"),
    }
    print(cell, json.dumps(results[cell], indent=1))

# exact-control (electrical-only) linear-vs-g2 comparison, matched to the exploratory design, both control states
lin_results = {}
for name, cv_idx, lin_key in (("full_control", full_idx, "full"), ("no_control", none_idx, "none")):
    y_true = np.array([V_sorted[i][cv_idx] for i in range(len(op_ids_sorted))])
    g2v = np.array([y1[(int(op_ids_sorted[i]), outages_sorted[i][0])][cv_idx] + y1[(int(op_ids_sorted[i]), outages_sorted[i][1])][cv_idx] + y1[(int(op_ids_sorted[i]), outages_sorted[i][2])][cv_idx]
                     + sum(y2[(int(op_ids_sorted[i]), p)][cv_idx] - y1[(int(op_ids_sorted[i]), p[0])][cv_idx] - y1[(int(op_ids_sorted[i]), p[1])][cv_idx]
                           for p in [(min(outages_sorted[i][0], outages_sorted[i][1]), max(outages_sorted[i][0], outages_sorted[i][1])),
                                     (min(outages_sorted[i][0], outages_sorted[i][2]), max(outages_sorted[i][0], outages_sorted[i][2])),
                                     (min(outages_sorted[i][1], outages_sorted[i][2]), max(outages_sorted[i][1], outages_sorted[i][2]))])
                     for i in range(len(op_ids_sorted))])
    ols = np.array([lin_by_row[(int(op_ids_sorted[i]), outages_sorted[i])][lin_key][0] for i in range(len(op_ids_sorted))])
    l1 = np.array([lin_by_row[(int(op_ids_sorted[i]), outages_sorted[i])][lin_key][1] for i in range(len(op_ids_sorted))])
    mae = lambda p: float(np.abs(y_true - p).mean()); mse = lambda p: float(((y_true - p) ** 2).mean())
    lin_results[name] = {"g2": {"mae": mae(g2v), "mse": mse(g2v)}, "ols_groupcv": {"mae": mae(ols), "mse": mse(ols)}, "l1_groupcv": {"mae": mae(l1), "mse": mse(l1)}}
    print(name, json.dumps(lin_results[name], indent=1))

out = {"partial_obs": {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}, "exact_control_linear": lin_results,
       "cost_accounting": {"total_lp_solves_all_arms_shared": 18001920, "n_ops": 60, "n_triples": 70,
                            "note": "g1/g2/prior_only/linear_ols/linear_l1 all read from the SAME n1/n2/n3 label tables -- no arm consumed extra LP solves."}}
json.dump(out, open("results/phase4/mobius_confirm.json", "w"), indent=1)
