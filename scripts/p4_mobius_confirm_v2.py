"""Phase 5 Stage 0: corrects the four gaps the overnight brief found in scripts/p4_mobius_confirm.py (kept, not
deleted). No new LP solves -- reuses the cached n1/n2/n3_confirm.npz label tables from the original run.

1. Saves per-op metric arrays AND per-row canonical-keyed predictions (data_hik/confirm_predictions.npz), so an
   independent reviewer can recompute every reported number without retraining or resolving.
2. Adds a labeled 68-triple sensitivity subset, excluding (0,23,34) and (9,13,40) -- present in the earlier
   200-triple exploratory extension (data_hik/manifest_amortization_new200.json), verified by exact set
   intersection -- by MANIFEST MEMBERSHIP ONLY, never by performance. Original 70-triple observation draws and
   tie-break keys are reused unchanged for retained rows (same rng derivation as the original run).
3. Runs an EXECUTABLE intersection-union decision (src/fdna/rules.intersection_union) across all 4 comparisons
   (g2 vs g1, g2 vs prior-only) x (P1, P2), at the pre-registered SESOI margin (0.05), replacing the earlier
   per-cell eyeballed tier check.
4. Re-derives the no-control linear-model MSE point estimate (g2 vs L1) as a PAIRED per-row difference with its
   own bootstrap CI, instead of a bare point estimate.
"""
import json
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import GroupKFold

from fdna import v2, v2data
from fdna.evalutil import rprec, scenario_uniform
from fdna.rules import intersection_union, boot

CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}
OVERLAP_TRIPLES = {(0, 23, 34), (9, 13, 40)}   # verified present in data_hik/manifest_amortization_new200.json

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

w = v2.build_world()
prior_pc = np.zeros(len(w.CV)); np.add.at(prior_pc, w.cidx, np.exp(w.logprior))
def weighted_median(vals, weights):
    o = np.argsort(vals); v, wt = vals[o], weights[o]; c = np.cumsum(wt); return float(v[np.searchsorted(c, 0.5 * c[-1])])

K_DRAWS = 100
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}

# ---- per-row predictions, saved with canonical keys, for both cells (independent-reproduction artifact) ----
all_rows = {"op_id": [], "outage": [], "cell": [], "draw": [], "g1": [], "g2": [], "prior_only": [], "y_true": [], "key": []}
per_op_arrays = {cell: {op: {"g1": [], "g2": [], "prior_only": []} for op in OP_IDS} for cell in CELLS}
for cell, (q, s, cov) in CELLS.items():
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id); a, b, c = outage
        y_true_grid = V_sorted[i].astype(np.float64)
        ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
        g1_grid = ya + yb + yc; g2_grid = g1_grid + sum(Ivals)

        rng = np.random.default_rng([op_id, *outage, CELL_SEED[cell]])   # UNCHANGED from the original run: same seed derivation
        st = v2.sample_states(w, rng, K_DRAWS)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]
        y_true = y_true_grid[w.cidx[st]]
        g1v, g2v = (pc * g1_grid).sum(1), (pc * g2_grid).sum(1)
        priorv = np.full(K_DRAWS, (prior_pc * g2_grid).sum())
        key = scenario_uniform(np.full(K_DRAWS, op_id), np.full(K_DRAWS, a), np.arange(K_DRAWS) + i * 1000, np.arange(K_DRAWS), salt=11)

        all_rows["op_id"].append(np.full(K_DRAWS, op_id)); all_rows["outage"].append(np.tile(outage, (K_DRAWS, 1)))
        all_rows["cell"].append(np.full(K_DRAWS, cell)); all_rows["draw"].append(np.arange(K_DRAWS))
        all_rows["g1"].append(g1v); all_rows["g2"].append(g2v); all_rows["prior_only"].append(priorv)
        all_rows["y_true"].append(y_true); all_rows["key"].append(key)

        d = per_op_arrays[cell][op_id]
        for k_, v_ in (("g1", g1v), ("g2", g2v), ("prior_only", priorv)):
            d[k_].append((outage, y_true, v_, key))

np.savez("data_hik/confirm_predictions.npz",
         op_id=np.concatenate(all_rows["op_id"]), outage=np.concatenate(all_rows["outage"]),
         cell=np.concatenate(all_rows["cell"]), draw=np.concatenate(all_rows["draw"]),
         g1=np.concatenate(all_rows["g1"]), g2=np.concatenate(all_rows["g2"]), prior_only=np.concatenate(all_rows["prior_only"]),
         y_true=np.concatenate(all_rows["y_true"]), key=np.concatenate(all_rows["key"]))
print("saved per-row predictions:", sum(len(x) for x in all_rows["g1"]), "rows")


def per_op_rprec(cell, exclude_triples):
    out = {}
    for op in OP_IDS:
        yt_parts, g1_parts, g2_parts, prior_parts, key_parts = [], [], [], [], []
        for name, store in (("g1", g1_parts), ("g2", g2_parts), ("prior_only", prior_parts)):
            for outage, y_true, arr, key in per_op_arrays[cell][op][name]:
                if outage in exclude_triples:
                    continue
                store.append(arr)
                if name == "g1":
                    yt_parts.append(y_true); key_parts.append(key)
        yt = np.concatenate(yt_parts); key_arr = np.concatenate(key_parts)
        g1a, g2a, pa = np.concatenate(g1_parts), np.concatenate(g2_parts), np.concatenate(prior_parts)
        out[op] = {"rprec_g1": rprec(yt, g1a, key_arr), "rprec_g2": rprec(yt, g2a, key_arr), "rprec_prior": rprec(yt, pa, key_arr)}
    return out


results = {}
for label, exclude in (("70_triple_original", set()), ("68_triple_sensitivity_excl_overlap", OVERLAP_TRIPLES)):
    cell_out = {}
    for cell in CELLS:
        po = per_op_rprec(cell, exclude)
        diff_g1 = np.array([po[o]["rprec_g2"] - po[o]["rprec_g1"] for o in OP_IDS])
        diff_prior = np.array([po[o]["rprec_g2"] - po[o]["rprec_prior"] for o in OP_IDS])
        cell_out[cell] = {
            "mean_rprec": {k_: float(np.mean([po[o][f"rprec_{k_}"] for o in OP_IDS])) for k_ in ("g1", "g2", "prior")},
            "boot_g2_minus_g1": boot(diff_g1), "boot_g2_minus_prior": boot(diff_prior),
            "n_ops_g2_beats_g1": int((diff_g1 > 0).sum()), "n_ops_g2_beats_prior": int((diff_prior > 0).sum()),
        }
    results[label] = cell_out
    print(label, json.dumps(cell_out, indent=1)[:2000])

# ---- executable intersection-union decision, SESOI=0.05, on the 70-triple (primary) result ----
components = {}
for cell in CELLS:
    po = per_op_rprec(cell, set())
    components[f"{cell}_vs_g1"] = np.array([po[o]["rprec_g2"] - po[o]["rprec_g1"] for o in OP_IDS])
    components[f"{cell}_vs_prior"] = np.array([po[o]["rprec_g2"] - po[o]["rprec_prior"] for o in OP_IDS])
decision = intersection_union(components, margin_per_component={k: 0.05 for k in components})
decision["tier1_confirmed"] = bool(all(s["mean"] > 0.05 and s["lo90"] >= 0 for s in decision["stats"].values()))
# stricter reading: require the 90% lower bound itself to clear the 0.05 margin in every component
decision["tier1_confirmed_strict_lo90_ge_margin"] = bool(all(
    boot(v, margin=0.0)["lo90"] >= 0.05 for v in components.values()))
print("DECISION", json.dumps({k: v for k, v in decision.items() if k != "stats"}, indent=1))

json.dump({"sensitivity": results, "intersection_union_decision": {k: v for k, v in decision.items() if k != "stats"},
           "decision_component_stats": decision["stats"]},
          open("results/phase5/mobius_confirm_corrected.json", "w"), indent=1, default=str)

# ---- linear no-control finding, restated with paired bootstrap uncertainty (exact control, matched to prior work) ----
full_idx, none_idx = int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))
outage_to_group = {o: i for i, o in enumerate(sorted(set(outages_sorted)))}
feats, ys, groups, outs = [], [], [], []
for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
    op_id = int(op_id); a, b, c = outage
    ya, yb, yc = y1[(op_id, a)][none_idx], y1[(op_id, b)][none_idx], y1[(op_id, c)][none_idx]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Ivals = [y2[(op_id, p)][none_idx] - y1[(op_id, p[0])][none_idx] - y1[(op_id, p[1])][none_idx] for p in pairs]
    feats.append([ya, yb, yc, *Ivals]); ys.append(float(V_sorted[i][none_idx])); groups.append(outage_to_group[outage]); outs.append(outage)
X, y, groups = np.array(feats), np.array(ys), np.array(groups)
gkf = GroupKFold(5); pred_l1 = np.zeros_like(y)
for tr_i, te_i in gkf.split(X, y, groups):
    m = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"); m.fit(X[tr_i], y[tr_i]); pred_l1[te_i] = m.predict(X[te_i])
g2v = X[:, :3].sum(1) + X[:, 3:].sum(1)
sq_g2, sq_l1 = (y - g2v) ** 2, (y - pred_l1) ** 2
diff_mse_raw = sq_g2 - sq_l1  # positive = g2 worse (higher squared error); one entry per (op,triple) ROW, 4200 total
# FIX (external review, finding 9): the row array has 4200 entries (60 ops x 70 triples), NOT 70 -- the earlier
# comment "rows here are one per triple already" was wrong, and boot(diff_mse_raw) passed all 4200 rows as if
# independent, understating uncertainty. Aggregate to 70 TRIPLE-level means (via `groups`, the triple index)
# before bootstrapping, matching the claimed "paired bootstrap over 70 triples."
n_groups = groups.max() + 1
diff_mse_per_triple = np.array([diff_mse_raw[groups == g].mean() for g in range(n_groups)])
assert len(diff_mse_per_triple) == 70
boot_diff_correct = boot(diff_mse_per_triple)
boot_diff_raw_mislabeled = boot(diff_mse_raw)  # kept for comparison, explicitly labelled as the WRONG unit
lin_out = {"g2_mse": float(sq_g2.mean()), "l1_mse": float(sq_l1.mean()), "relative_gap_pct": float((sq_g2.mean() - sq_l1.mean()) / sq_g2.mean() * 100),
           "paired_bootstrap_over_70_triples_CORRECT": boot_diff_correct,
           "paired_bootstrap_over_4200_rows_MISLABELED_original": boot_diff_raw_mislabeled,
           "note": "the original committed version passed all 4200 (op,triple) rows to boot() and called it a 70-triple bootstrap; that was wrong (external review finding 9). The CORRECT field aggregates to 70 triple-level means first, matching the claimed unit; interpret lo90>0 there as g2 reliably worse."}
print("LINEAR NO-CONTROL", json.dumps(lin_out, indent=1))
json.dump(lin_out, open("results/phase5/linear_nocontrol_paired.json", "w"), indent=1)
