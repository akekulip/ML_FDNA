"""Phase 4 Stage 3 -- THE core scientific test: does a k=2-additive (Mobius/interaction) truncation, fit ENTIRELY
from N-1/N-2 information, predict the held-out EXACT k=3 labels already generated (data_hik/k3_screen.npz)?

g_2(S) = sum_i y({i}) + sum_{i<j in S} I(i,j),  I(a,b) = y({a,b}) - y({a}) - y({b})  (y(empty)=0 exactly, verified).

Correction from the first attempt: the old train/val vtables use DISJOINT operating-point ranges (train ops
0-99, val ops 100-119) and train's own table carries no N-2 rows at all -- so no existing table has the
(op, N-1/N-2) ingredients needed for the k3 manifest's own 20 ops. This version generates exactly the needed
singles and sub-pair labels FRESH, at the single fixed control CV[CV_IDX] used by the k3 screen, directly via
ScenarioLP -- a small, targeted labeling job (not the full 1024-control table), justified because only these
specific (op, branch) and (op, sub-pair) values are needed for this diagnostic.

Baselines it must beat: (a) g_1 = sum of singletons alone (ignores all interaction); (b) reported alongside,
not as a trained model here (an equal-information GBM on the same ingredients is a follow-up, not run in this
screen -- flagged as not yet done)."""
import json
import numpy as np
from scipy.stats import spearmanr

from fdna import spec, v2data
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

CV_IDX = int(np.argmax(np.load("data_hik/k3_screen.npz")["CV"].sum(1)))   # the fullest-control row, matches k3_screen's CV column order
k3 = np.load("data_hik/k3_screen.npz")
CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]

needed_singles, needed_pairs = set(), set()
for outage in OUTAGES:
    for b in outage:
        needed_singles.add(b)
    for a, b in [(outage[0], outage[1]), (outage[0], outage[2]), (outage[1], outage[2])]:
        needed_pairs.add((min(a, b), max(a, b)))
print(f"distinct branches needed: {len(needed_singles)}, distinct sub-pairs needed: {len(needed_pairs)}")

y_single, y_pair = {}, {}
for op_id in OP_IDS:
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    for b in needed_singles:
        y_single[(op_id, b)] = ScenarioLP(G, op, (b,), spec.PARAMS).y(CV[CV_IDX])
    for a, b in needed_pairs:
        y_pair[(op_id, a, b)] = ScenarioLP(G, op, (a, b), spec.PARAMS).y(CV[CV_IDX])
print(f"generated {len(y_single)} singles + {len(y_pair)} pairs = {len(y_single)+len(y_pair)} fresh LP solves")

rows = []
for op_id, outage, y_true in zip(k3["op_ids"], [tuple(int(x) for x in o) for o in k3["outages"]], k3["V"][:, CV_IDX]):
    op_id = int(op_id)
    singles = [y_single[(op_id, b)] for b in outage]
    g1 = sum(singles)
    subpairs = [(min(outage[0], outage[1]), max(outage[0], outage[1])), (min(outage[0], outage[2]), max(outage[0], outage[2])), (min(outage[1], outage[2]), max(outage[1], outage[2]))]
    interactions = [y_pair[(op_id, a, b)] - y_single[(op_id, a)] - y_single[(op_id, b)] for a, b in subpairs]
    g2 = g1 + sum(interactions)
    rows.append((op_id, outage, float(y_true), g1, g2))

y_true = np.array([r[2] for r in rows]); g1 = np.array([r[3] for r in rows]); g2 = np.array([r[4] for r in rows])
err1, err2 = np.abs(y_true - g1), np.abs(y_true - g2)
out = {
    "n_rows": len(rows), "cv_idx": int(CV_IDX), "cv_row": CV[CV_IDX].tolist(),
    "extra_lp_solves_for_ingredients": len(y_single) + len(y_pair),
    "g1_singleton_sum_only": {"mae": float(err1.mean()), "rmse": float(np.sqrt((err1**2).mean())), "spearman_vs_true": float(spearmanr(g1, y_true)[0]) if g1.std() > 0 else None},
    "g2_mobius_pairwise_truncation": {"mae": float(err2.mean()), "rmse": float(np.sqrt((err2**2).mean())), "spearman_vs_true": float(spearmanr(g2, y_true)[0]) if g2.std() > 0 else None},
    "g2_beats_g1_mae": bool(err2.mean() < err1.mean()), "g2_beats_g1_rmse": bool(np.sqrt((err2**2).mean()) < np.sqrt((err1**2).mean())),
    "mean_y_true": float(y_true.mean()), "frac_y_true_severe": float((y_true > spec.SEVERE).mean()),
    "note": "single fixed full-control state (isolates electrical/topological generalisation, no comm-inference layer, per brief 6.3); equal-information GBM comparator not yet run (flagged, not done)."
}
print(json.dumps(out, indent=1))
json.dump(out, open("results/phase4/mobius_k3_test.json", "w"), indent=1)
