"""Phase 4 Stage 3: generate N-1 (all 41 branches) + a sampled N-2 (300 pairs) training set for the SAME 20 ops
used by the frozen k=3/k=4 zero-shot eval manifests, at the same fixed control state. Frozen sampling for N-2
(seed 100, committed before generation). This is the ONLY data any neural/tree comparator may train on; the k=3
and k=4 manifests remain zero-shot eval sets, never touched during training."""
import json
import numpy as np

from fdna import spec
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]; CV_IDX = int(np.argmax(CV.sum(1)))
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]

rng = np.random.default_rng(100)
all_pairs = [(a, b) for a in range(G.n_branch) for b in range(a + 1, G.n_branch)]
sample_idx = rng.choice(len(all_pairs), 300, replace=False)
N2_PAIRS = [all_pairs[i] for i in sample_idx]
json.dump({"seed": 100, "n_pairs": 300, "pairs": [list(p) for p in N2_PAIRS], "op_ids": OP_IDS, "cv_idx": CV_IDX},
          open("data_hik/manifest_train_n2sample.json", "w"), indent=1)

rows = []
for op_id in OP_IDS:
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    for b in range(G.n_branch):
        y = ScenarioLP(G, op, (b,), spec.PARAMS).y(CV[CV_IDX])
        rows.append((op_id, (b,), y))
    for a, b in N2_PAIRS:
        y = ScenarioLP(G, op, (a, b), spec.PARAMS).y(CV[CV_IDX])
        rows.append((op_id, (a, b), y))
op_ids = np.array([r[0] for r in rows])
outages = np.array([list(r[1]) + [-1] * (2 - len(r[1])) for r in rows])
y = np.array([r[2] for r in rows], np.float32)
np.savez("data_hik/train_n1n2.npz", op_ids=op_ids, outages=outages, y=y, cv_idx=CV_IDX)
print(len(rows), "rows (41 N-1 + 300 N-2) x", len(OP_IDS), "ops =", len(rows))
