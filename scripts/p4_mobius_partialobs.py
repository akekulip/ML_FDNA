"""Phase 4: does the g2 (order-2 Mobius truncation) advantage survive under REALISTIC partial/stale communication
observation of the control state, composed with the exact posterior p(c|obs) (src/fdna/v2.py, unchanged from
Phase 2/3)? registry/phase4_step_partialobs.yaml, committed before the new y_pair labels were generated.

Comparators, all composed with the SAME exact posterior:
  - g2_composed: sum_c p(c|obs) * g2(op,S,c)   [g2 built from y_single (free, train table) + fresh full-grid y_pair]
  - g1_composed: sum_c p(c|obs) * g1(op,S,c)   [naive singleton sum, no interaction terms]
  - oracle_full_table: sum_c p(c|obs) * V_true(op,S,c)   [upper bound: as if we had the true N-3 table -- data_hik/k3_screen.npz]
  - assume_full_control: g2(op,S, c=all-ones)  [zero-inference-cost reference: ignore observation, assume best case]
All scored against the TRUE realised shed y_true(op,S,c_true), c_true drawn from the TRUE hidden comm state.
"""
import json
import numpy as np

from fdna import v2, v2data

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]

tr = v2data.load_vtable("data_v2", "train")
y1 = {}   # (op_id, branch) -> (1024,) shed over all controls, FREE from the existing dense N-1 rows
for i, op_id in enumerate(tr["op_ids"]):
    op_id = int(op_id)
    if op_id not in OP_IDS:
        continue
    for j, (a, b) in enumerate(tr["conts"][i]):
        if a >= 0 and b < 0:
            y1[(op_id, int(a))] = tr["V"][i, j].astype(np.float64)

yp = np.load("data_hik/ypair_fullgrid.npz")
y2 = {(int(op), tuple(int(x) for x in pair)): Y.astype(np.float64) for op, pair, Y in zip(yp["op_ids"], yp["pairs"], yp["Y"])}

full_c_idx = int(np.argmax(CV.sum(1)))
w = v2.build_world()
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
K_DRAWS = 100
rng = np.random.default_rng(7)

results = {}
for cell, (q, s, cov) in CELLS.items():
    rows = []
    for i, (op_id, outage) in enumerate(zip(k3["op_ids"], [tuple(int(x) for x in o) for o in k3["outages"]])):
        op_id = int(op_id)
        if op_id not in OP_IDS:
            continue
        a, b, c = outage
        y_true_grid = k3["V"][i].astype(np.float64)   # (1024,), the TRUE N-3 shed at every control
        ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
        g1_grid = ya + yb + yc
        g2_grid = g1_grid + sum(Ivals)

        st = v2.sample_states(w, rng, K_DRAWS)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]   # (K_DRAWS, 1024) exact posterior per draw
        true_c_idx = w.cidx[st]                      # (K_DRAWS,) TRUE control index per draw
        y_true = y_true_grid[true_c_idx]              # realised true shed
        for k in range(K_DRAWS):
            rows.append((
                (pc[k] * g2_grid).sum(), (pc[k] * g1_grid).sum(), (pc[k] * y_true_grid).sum(),
                g2_grid[full_c_idx], y_true[k]))
    r = np.array(rows)
    pred_g2, pred_g1, pred_oracle, pred_assume_full, y_true_all = r.T
    mae = lambda p: float(np.abs(y_true_all - p).mean())
    results[cell] = {"n_rows": len(r),
                      "g2_composed": mae(pred_g2), "g1_composed": mae(pred_g1),
                      "oracle_full_table_composed": mae(pred_oracle), "assume_full_control_g2": mae(pred_assume_full),
                      "g2_beats_g1": bool(mae(pred_g2) < mae(pred_g1)),
                      "g2_beats_assume_full": bool(mae(pred_g2) < mae(pred_assume_full)),
                      "gap_to_oracle_g2": mae(pred_g2) - mae(pred_oracle)}
    print(cell, json.dumps(results[cell], indent=1))
json.dump(results, open("results/phase4/mobius_partialobs.json", "w"), indent=1)
