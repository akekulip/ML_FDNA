"""Phase 4 correction: honest measurement of the reuse/amortisation claim an external review challenged.
'Practically unaffordable' was wrong for the original 60-triple test (direct labelling was cheaper there).
The real argument is combinatorial reuse: pairs, once built, predict MANY triples. This test samples 200 NEW
triples (frozen before checking coverage) among the same 38 branches the existing 156-pair table (data_hik/
ypair_fullgrid.npz) covers, measures how many of their sub-pairs are ALREADY available (sunk cost) vs need new
solves (incremental cost), generates only the missing pairs plus the 200 new triples' TRUE labels (for scoring,
not for prediction), and reports the honest solve-count comparison and g2's accuracy on this larger unseen set."""
import json
import numpy as np
from itertools import combinations
from scipy.stats import spearmanr

from fdna import spec, v2, v2data
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]; CV_IDX = int(np.argmax(CV.sum(1)))
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; ORIGINAL_OUTAGES = set(tuple(s) for s in manifest["outage_sets"])
touched_branches = sorted({b for o in ORIGINAL_OUTAGES for b in o})
yp = np.load("data_hik/ypair_fullgrid.npz")
existing_pairs = {tuple(int(x) for x in p) for p in yp["pairs"]}

all_triples_among_touched = list(combinations(touched_branches, 3))
rng = np.random.default_rng(200)
new_candidates = [t for t in all_triples_among_touched if t not in ORIGINAL_OUTAGES]
NEW_OUTAGES = [new_candidates[i] for i in rng.choice(len(new_candidates), 200, replace=False)]
json.dump({"seed": 200, "n_new": 200, "outages": [list(o) for o in NEW_OUTAGES]}, open("data_hik/manifest_amortization_new200.json", "w"), indent=1)

needed_pairs_new = sorted({(min(o[i], o[j]), max(o[i], o[j])) for o in NEW_OUTAGES for i, j in [(0, 1), (0, 2), (1, 2)]})
missing_pairs = [p for p in needed_pairs_new if p not in existing_pairs]
print(f"200 new triples need {len(needed_pairs_new)} distinct sub-pairs; {len(existing_pairs)} already exist; {len(missing_pairs)} are genuinely NEW (incremental)")

tr = v2data.load_vtable("data_v2", "train")
y1 = {}
for i, op_id in enumerate(tr["op_ids"]):
    op_id = int(op_id)
    if op_id not in OP_IDS: continue
    for j, (a, b) in enumerate(tr["conts"][i]):
        if a >= 0 and b < 0: y1[(op_id, int(a))] = tr["V"][i, j].astype(np.float64)[CV_IDX]
y2 = {(int(o), tuple(int(x) for x in p)): Y[CV_IDX] for o, p, Y in zip(yp["op_ids"], yp["pairs"], yp["Y"])}

y2_new = {}
for op_id in OP_IDS:
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    for p in missing_pairs:
        y2_new[(op_id, p)] = ScenarioLP(G, op, p, spec.PARAMS).y(CV[CV_IDX])
n_incremental_solves = len(missing_pairs) * len(OP_IDS)
print(f"incremental solves for the missing pairs (1 control state): {n_incremental_solves}")

y_true_new, g2_new = [], []
for op_id in OP_IDS:
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    for outage in NEW_OUTAGES:
        a, b, c = outage
        y_true_new.append(ScenarioLP(G, op, outage, spec.PARAMS).y(CV[CV_IDX]))
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = []
        for p in pairs:
            yp_val = y2.get((op_id, p), y2_new.get((op_id, p)))
            Ivals.append(yp_val - y1[(op_id, p[0])] - y1[(op_id, p[1])])
        g2_new.append(y1[(op_id, a)] + y1[(op_id, b)] + y1[(op_id, c)] + sum(Ivals))
n_direct_solves_for_scoring_only = len(NEW_OUTAGES) * len(OP_IDS)  # NOT needed for prediction, only to SCORE g2 here

y_true_new, g2_new = np.array(y_true_new), np.array(g2_new)
mae = float(np.abs(y_true_new - g2_new).mean()); rho = float(spearmanr(g2_new, y_true_new)[0])
# CORRECTED (external review): "N cached pairs support ALL C(n,3) triples" was wrong -- a triple needs its
# OWN 3 sub-pairs present, not just any N pairs among the branches. Count actual coverage, not a mismatched pairing.
all_pairs_possible = set(combinations(touched_branches, 2))
cached_now = existing_pairs | set(missing_pairs)
covered_by_cached = sum(1 for t in all_triples_among_touched
                        if all(tuple(sorted((t[i], t[j]))) in cached_now for i, j in [(0, 1), (0, 2), (1, 2)]))
covered_by_all_possible = sum(1 for t in all_triples_among_touched
                              if all(tuple(sorted((t[i], t[j]))) in all_pairs_possible for i, j in [(0, 1), (0, 2), (1, 2)]))
out = {
    "n_new_unseen_triples": len(NEW_OUTAGES), "n_ops": len(OP_IDS),
    "sub_pairs_needed": len(needed_pairs_new), "sub_pairs_already_available_sunk_cost": len(needed_pairs_new) - len(missing_pairs),
    "sub_pairs_genuinely_new_incremental": len(missing_pairs),
    "incremental_solves_to_extend_coverage_to_these_200_new_triples": n_incremental_solves,
    "solves_that_WOULD_be_needed_to_label_these_200_triples_directly": n_direct_solves_for_scoring_only,
    "g2_prediction_quality_on_these_never-seen-before_triples": {"mae": mae, "rho": rho},
    "coverage_CORRECTED": {
        "n_cached_pairs": len(cached_now),
        "n_triples_with_all_3_subpairs_cached": covered_by_cached,
        "n_triples_possible_among_touched_branches": len(all_triples_among_touched),
        "n_all_possible_pairs_among_touched_branches": len(all_pairs_possible),
        "n_triples_covered_if_ALL_possible_pairs_were_cached": covered_by_all_possible,
        "note": "the original 'reuse_ratio_claim' field wrongly paired 477 (cached pairs) with 8436 (all triples) -- 477 cached pairs cover only a FRACTION of the 8436 possible triples; only the FULL 703-pair set covers all 8436."
    },
    "note": "This tests genuine reuse: the incremental solves to extend the pair table (small) vs the g2 prediction quality on triples that were NEVER part of the original 60-triple manifest. The cost-savings claim itself is separately measured, not assumed."
}
print(json.dumps(out, indent=1))
json.dump(out, open("results/phase4/amortization_test.json", "w"), indent=1)
