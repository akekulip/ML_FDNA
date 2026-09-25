"""Phase 5 Stage R3: matched recurrent/learned-correction evaluation, v2 -- uses the CORRECTED models
(scripts/p5_matched_recurrent_experiment_v2.py: fixed validation alignment, disjoint train/val groups,
genuinely permutation-invariant DeepSets [repair round 2: full pipeline, including the risk-feature readout
tail], scaled features). Composes every arm with the SAME P1/P2 posterior, scores R-precision on the 25
held-out TEST triples (disjoint from both train and val on the outage axis), all 60 operating points (the
known-operating-point / new-outage-combination axis, as documented in the training split -- TEST intentionally
reuses train/val operating points, never claimed otherwise)."""
import json, pickle
import numpy as np
import torch

from fdna import v2
from fdna.dataset import G, RATING
from fdna.evalutil import rprec, sample_posterior_and_truth
from fdna.lodf import ptdf_lodf
from fdna.opgen import sample_op
from fdna.physical_correction import g2_clipped
from fdna.rules import boot
from fdna.nn.pairset import DeepSetsPairAware, ResidualNet, OrderedMLP, PermAveragedModel
from fdna.nn import pairset_features as pf

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
PHYS = {(op, o): pf.physical_feats(G, LODF, demands[op], o) for op in OP_IDS for o in TEST_OUTAGES}


def build_features_all(op_id, outage):
    idx_all = np.arange(len(CV))
    y_full = V_sorted[row_by_outage_op[(op_id, outage)]]
    X, y, g2 = pf.build_row(op_id, outage, idx_all, y1=y1, y2=y2, phys=PHYS[(op_id, outage)], CV=CV, y_true=y_full)
    pf.apply_scaler(X, phys_mean, phys_std)
    return X, g2


with open("data_hik/matched_recurrent_models_v2.pkl", "rb") as f:
    models = pickle.load(f)
ridge, gbm = models["ridge"], models["gbm"]
ds_model = DeepSetsPairAware(); ds_model.load_state_dict(torch.load("data_hik/deepsets_model_v2.pt")); ds_model.eval()
# repair round 3 (external review finding 5): evaluate ALL 3 predeclared residual seeds, not only seed 0 --
# "3 seeds matching validation scores" was never 3 independent test replications when only 1 was ever scored
# on TEST. Also add residual_frozen_shrunk: the trained model's raw_correction (unscaled by its own jointly-
# trained `lam`) times a shrink value SELECTED ON VALIDATION via a prespecified grid (SHRINK_GRID in the
# training script) -- a properly separated shrinkage estimate, since `lam` itself is not independently
# identifiable (the final linear layer's own weights can absorb any rescaling of a jointly-trained scalar).
res_models = {}
for s in (0, 1, 2):
    path = "data_hik/residual_model_v2.pt" if s == 0 else f"data_hik/residual_model_v2_seed{s}.pt"
    m = ResidualNet(); m.load_state_dict(torch.load(path)); m.eval()
    res_models[s] = m
frozen_shrink = train_info["residual_diagnostics"]["frozen_shrinkage_selection"]["selected_shrink"]

# repair round 3 ablation (external review section 4): a canonical ORDERED (non-invariant) baseline, and a
# PERMUTATION-AVERAGED wrapper around it (exactly invariant by construction, 6 forward passes charged) -- both
# trained/evaluated across the same 3 seeds, isolating architecture (pooling vs. ordered+averaging) from the
# normalization/optimization changes that also happened between round 1 and round 2.
COLUMN_PERMS = [pf.column_perm_for_vertex_perm(sigma) for sigma in pf.ALL_VERTEX_PERMS]
ORDERED_SEEDS = (0, 1, 2)
ordered_models, ordered_pa_models = {}, {}
for s in ORDERED_SEEDS:
    mo = OrderedMLP(); mo.load_state_dict(torch.load(f"data_hik/ordered_model_v2_seed{s}.pt")); mo.eval()
    ordered_models[s] = mo
    ordered_pa_models[s] = PermAveragedModel(mo, COLUMN_PERMS).eval()

w = v2.build_world()
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}
# g2_fixed doubles as the residual arm's zero-correction (lambda=0) baseline: ResidualNet's final layer is
# zero-initialized (src/fdna/nn/pairset.py), so "did the trained correction actually help vs doing nothing"
# is answered directly by comparing the residual row below to this row, not by a separate arm.
ARMS = (["g2_fixed", "g2_clipped", "ridge", "gbm", "deepsets",
         "residual", "residual_seed1", "residual_seed2", "residual_frozen_shrunk"]
        + [f"ordered_seed{s}" for s in ORDERED_SEEDS]
        + [f"ordered_perm_avg_seed{s}" for s in ORDERED_SEEDS])

results = {}
for cell, (q, s, cov) in CELLS.items():
    per_op = {op: {a: [] for a in ARMS} for op in OP_IDS}
    per_op_y = {op: [] for op in OP_IDS}
    per_op_key = {op: [] for op in OP_IDS}
    for outage in TEST_OUTAGES:
        for op_id in OP_IDS:
            i = row_by_outage_op[(op_id, outage)]
            y_true_grid = V_sorted[i].astype(np.float64)
            X, g2grid = build_features_all(op_id, outage)
            preds = {"g2_fixed": g2grid, "g2_clipped": np.array([g2_clipped(v, G, demands[op_id], outage) for v in g2grid])}
            preds["ridge"] = ridge.predict(X)
            preds["gbm"] = gbm.predict(X)
            with torch.no_grad():
                Xt = torch.tensor(X)
                preds["deepsets"] = ds_model(Xt).numpy()
                preds["residual"] = g2grid + res_models[0](Xt).numpy()
                preds["residual_seed1"] = g2grid + res_models[1](Xt).numpy()
                preds["residual_seed2"] = g2grid + res_models[2](Xt).numpy()
                preds["residual_frozen_shrunk"] = g2grid + frozen_shrink * res_models[0].raw_correction(Xt).numpy()
                # NOTE: loop variable named os_seed, NOT s -- the outer `for cell, (q, s, cov) in CELLS.items():`
                # already uses `s` for the coverage-share parameter; reusing `s` here silently clobbered it
                # (Python has no block scoping), corrupting sample_posterior_and_truth's `s` argument for
                # every row after the first, identically across every arm -- caught by a suspiciously identical,
                # suspiciously low R-precision across ALL arms on the first real rerun, not assumed correct.
                for os_seed in ORDERED_SEEDS:
                    preds[f"ordered_seed{os_seed}"] = ordered_models[os_seed](Xt).numpy()
                    preds[f"ordered_perm_avg_seed{os_seed}"] = ordered_pa_models[os_seed](Xt).numpy()

            # sample the shared posterior/observation ONCE per (op,outage), compose every arm against it --
            # not once per arm (which would redundantly redraw the same deterministic seed 6x)
            pc, true_c_idx, key = sample_posterior_and_truth(w, q, s, cov, op_id, outage, seed=CELL_SEED[cell])
            y_true = y_true_grid[true_c_idx]
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
    results[cell]["ablation_summary"] = {
        "ordered_mean": float(np.mean([mean_rprec[f"ordered_seed{s}"] for s in ORDERED_SEEDS])),
        "ordered_std": float(np.std([mean_rprec[f"ordered_seed{s}"] for s in ORDERED_SEEDS])),
        "ordered_perm_avg_mean": float(np.mean([mean_rprec[f"ordered_perm_avg_seed{s}"] for s in ORDERED_SEEDS])),
        "ordered_perm_avg_std": float(np.std([mean_rprec[f"ordered_perm_avg_seed{s}"] for s in ORDERED_SEEDS])),
        "deepsets": mean_rprec["deepsets"],
        "scope": "isolates architecture (pooling vs ordered+6-way averaging) holding data/features/split/seeds "
                 "fixed; does NOT establish whether ordering is a real transferable signal on other topologies "
                 "or branch renumberings (out of scope, external review section 4, repair round 3).",
    }
    print(cell, json.dumps(results[cell], indent=1))
json.dump(results, open("results/phase5/matched_recurrent_eval_v2.json", "w"), indent=1)
