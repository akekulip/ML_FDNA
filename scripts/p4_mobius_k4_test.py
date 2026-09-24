"""Phase 4 Stage 3: does the Mobius truncation extend to N-4? Tests g2 (order<=2, using only N-1/N-2 info,
naively summed over all 6 sub-pairs of the 4-set) against g3 (order<=3, ALSO using N-3 interaction terms --
now genuinely available since N-3 labels were exactly solved this session) at predicting exact N-4 labels
(data_hik/k4_screen.npz). This is the actual test of whether the truncation order needs to grow with k, or
whether order-2 alone remains sufficient -- the central open question after the N-3 result."""
import json
import numpy as np
from scipy.stats import spearmanr

from fdna import spec
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op
from itertools import combinations

manifest = json.load(open("data_hik/manifest_k4_screen.json"))
OP_IDS, OUTAGES = manifest["op_ids"], [tuple(s) for s in manifest["outage_sets"]]
k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
CV_IDX = int(np.argmax(CV.sum(1)))

needed_singles = sorted({b for o in OUTAGES for b in o})
needed_pairs = sorted({tuple(sorted(p)) for o in OUTAGES for p in combinations(o, 2)})
needed_triples = sorted({tuple(sorted(t)) for o in OUTAGES for t in combinations(o, 3)})
print(f"needed: {len(needed_singles)} singles, {len(needed_pairs)} pairs, {len(needed_triples)} triples, {len(OUTAGES)} quadruples, {len(OP_IDS)} ops")

y1, y2, y3, y4 = {}, {}, {}, {}
for op_id in OP_IDS:
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    for b in needed_singles: y1[(op_id, b)] = ScenarioLP(G, op, (b,), spec.PARAMS).y(CV[CV_IDX])
    for p in needed_pairs: y2[(op_id, p)] = ScenarioLP(G, op, p, spec.PARAMS).y(CV[CV_IDX])
    for t in needed_triples: y3[(op_id, t)] = ScenarioLP(G, op, t, spec.PARAMS).y(CV[CV_IDX])
    for S in OUTAGES: y4[(op_id, S)] = ScenarioLP(G, op, S, spec.PARAMS).y(CV[CV_IDX])
n_solves = len(y1) * len(OP_IDS) // max(len(OP_IDS), 1)  # placeholder, real count below
total_solves = len(needed_singles) * len(OP_IDS) + len(needed_pairs) * len(OP_IDS) + len(needed_triples) * len(OP_IDS) + len(OUTAGES) * len(OP_IDS)
print(f"total LP solves: {total_solves}")

rows = []
for op_id in OP_IDS:
    for S in OUTAGES:
        y_true = y4[(op_id, S)]
        ys = [y1[(op_id, b)] for b in S]
        g1 = sum(ys)
        pair_I = {p: y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in combinations(S, 2)}
        g2 = g1 + sum(pair_I.values())
        # third-order Mobius term I(T) = sum_{U subset T} (-1)^(|T|-|U|) y(U), the standard recursive definition
        triple_I = {}
        for T in combinations(S, 3):
            acc = 0.0
            for r in range(4):
                sign = (-1) ** (3 - r)
                for U in combinations(T, r):
                    yU = 0.0 if r == 0 else (y1[(op_id, U[0])] if r == 1 else (y2[(op_id, tuple(sorted(U)))] if r == 2 else y3[(op_id, tuple(sorted(U)))]))
                    acc += sign * yU
            triple_I[T] = acc
        g3 = g2 + sum(triple_I.values())
        rows.append((op_id, S, y_true, g1, g2, g3))

y_true = np.array([r[2] for r in rows]); g1 = np.array([r[3] for r in rows]); g2 = np.array([r[4] for r in rows]); g3 = np.array([r[5] for r in rows])
mae = lambda p: float(np.abs(y_true - p).mean()); rho = lambda p: float(spearmanr(p, y_true)[0]) if p.std() > 0 else None
out = {"n_rows": len(rows), "cv_idx": CV_IDX, "total_lp_solves": total_solves,
       "g1_singleton_sum": {"mae": mae(g1), "rho": rho(g1)},
       "g2_order2_truncation": {"mae": mae(g2), "rho": rho(g2)},
       "g3_order3_truncation": {"mae": mae(g3), "rho": rho(g3)},
       "g3_beats_g2": bool(mae(g3) < mae(g2)), "g2_beats_g1": bool(mae(g2) < mae(g1)),
       "mean_y_true": float(y_true.mean())}
print(json.dumps(out, indent=1))
json.dump(out, open("results/phase4/mobius_k4_test.json", "w"), indent=1)
np.savez("data_hik/k4_screen.npz", op_ids=np.array([r[0] for r in rows]),
         outages=np.array([list(r[1]) for r in rows]), y=np.array([r[2] for r in rows], np.float32), CV=CV)
