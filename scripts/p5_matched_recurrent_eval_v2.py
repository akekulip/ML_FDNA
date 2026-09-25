"""Phase 5 Stage R3: matched recurrent/learned-correction evaluation, v2 -- uses the CORRECTED models
(scripts/p5_matched_recurrent_experiment_v2.py: fixed validation alignment, disjoint train/val/test groups,
genuinely permutation-invariant DeepSets, scaled features). Composes every arm with the SAME P1/P2 posterior,
scores R-precision on the 25 held-out TEST triples (disjoint from both train and val), all 60 operating points
(the known-operating-point / new-outage-combination axis, as documented in the training split)."""
import json, pickle
import numpy as np
import torch

from fdna import v2, v2data
from fdna.dataset import G, RATING
from fdna.evalutil import rprec, scenario_uniform
from fdna.lodf import ptdf_lodf, compensation_det
from fdna.opgen import sample_op
from fdna.physical_correction import island_floor, g2_clipped
from fdna.rules import boot
from fdna.nn.pairset import DeepSetsPairAware, ResidualNet

n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]
train_info = json.load(open("results/phase5/matched_recurrent_training_v2.json"))
TEST_OUTAGES = [tuple(o) for o in train_info["split"]["test_outages"]]
phys_mean = np.array(train_info["feature_scaling"]["phys_mean"]); phys_std = np.array(train_info["feature_scaling"]["phys_std"])

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]
row_by_outage_op = {(int(op_ids_sorted[i]), outages_sorted[i]): i for i in range(len(outages_sorted))}

demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in OP_IDS}
_, LODF = ptdf_lodf(G)


def physical_feats(op_id, outage):
    a, b, c = outage
    F = island_floor(G, demands[op_id], outage)
    pairs = [(a, b), (a, c), (b, c)]
    risks = [min(1.0 / max(abs(compensation_det(LODF, p[0], p[1])), 1e-9), 1e4) for p in pairs]
    return F, risks


PHYS = {(op, o): physical_feats(op, o) for op in OP_IDS for o in TEST_OUTAGES}


def build_features_all(op_id, outage):
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iab, Iac, Ibc = (y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs)
    F, risks = PHYS[(op_id, outage)]
    risks_scaled = [np.log1p(r) for r in risks]   # same transform as training
    phys_rep = np.tile([F, *risks_scaled], (len(CV), 1))
    phys_rep = (phys_rep - phys_mean) / phys_std    # same standardisation, TRAIN statistics
    X = np.column_stack([ya, yb, yc, Iab, Iac, Ibc, phys_rep, CV]).astype(np.float32)
    g2 = (ya + yb + yc + Iab + Iac + Ibc).astype(np.float32)
    return X, g2, F


with open("data_hik/matched_recurrent_models_v2.pkl", "rb") as f:
    models = pickle.load(f)
ridge, gbm = models["ridge"], models["gbm"]
ds_model = DeepSetsPairAware(); ds_model.load_state_dict(torch.load("data_hik/deepsets_model_v2.pt")); ds_model.eval()
res_model = ResidualNet(); res_model.load_state_dict(torch.load("data_hik/residual_model_v2.pt")); res_model.eval()

w = v2.build_world()
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
K_DRAWS = 50
CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}
ARMS = ["g2_fixed", "g2_clipped", "ridge", "gbm", "deepsets", "residual"]

results = {}
for cell, (q, s, cov) in CELLS.items():
    per_op = {op: {a: [] for a in ARMS} for op in OP_IDS}
    per_op_y = {op: [] for op in OP_IDS}
    per_op_key = {op: [] for op in OP_IDS}
    for outage in TEST_OUTAGES:
        for op_id in OP_IDS:
            i = row_by_outage_op[(op_id, outage)]
            y_true_grid = V_sorted[i].astype(np.float64)
            X, g2grid, F = build_features_all(op_id, outage)
            preds = {"g2_fixed": g2grid, "g2_clipped": np.array([g2_clipped(v, G, demands[op_id], outage) for v in g2grid])}
            preds["ridge"] = ridge.predict(X)
            preds["gbm"] = gbm.predict(X)
            with torch.no_grad():
                Xt = torch.tensor(X)
                preds["deepsets"] = ds_model(Xt).numpy()
                preds["residual"] = g2grid + res_model(Xt).numpy()

            rng = np.random.default_rng([op_id, *outage, CELL_SEED[cell]])
            st = v2.sample_states(w, rng, K_DRAWS)
            obs = v2.emit(w, st, q, s, rng, cov)
            pc = v2data.oracle_features(w, obs, s)[0]
            true_c_idx = w.cidx[st]
            y_true = y_true_grid[true_c_idx]
            key = scenario_uniform(np.full(K_DRAWS, op_id), np.full(K_DRAWS, outage[0]), np.arange(K_DRAWS) + hash(outage) % 100000, np.arange(K_DRAWS), salt=13)

            for arm in ARMS:
                per_op[op_id][arm].append((pc * preds[arm]).sum(1))
            per_op_y[op_id].append(y_true); per_op_key[op_id].append(key)

    per_op_rprec = {op: {} for op in OP_IDS}
    for op in OP_IDS:
        yt = np.concatenate(per_op_y[op]); key = np.concatenate(per_op_key[op])
        for arm in ARMS:
            pred = np.concatenate(per_op[op][arm])
            per_op_rprec[op][arm] = rprec(yt, pred, key)
    mean_rprec = {arm: float(np.mean([per_op_rprec[op][arm] for op in OP_IDS])) for arm in ARMS}
    diffs = {arm: np.array([per_op_rprec[op][arm] - per_op_rprec[op]["g2_fixed"] for op in OP_IDS]) for arm in ARMS if arm != "g2_fixed"}
    boots = {arm: boot(d) for arm, d in diffs.items()}
    results[cell] = {"mean_rprec": mean_rprec, "boot_vs_g2_fixed": boots}
    print(cell, json.dumps(results[cell], indent=1))
json.dump(results, open("results/phase5/matched_recurrent_eval_v2.json", "w"), indent=1)
