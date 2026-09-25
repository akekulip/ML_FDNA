"""Phase 5 Stage R5: evaluate the reviewer's proposed g2_capacity_clipped (clip g2_plain to
[capacity_floor(S,c), 1] instead of [island_floor(S), 1]) on the same cached confirmatory N-3 table and
(now-fixed, external review finding 7) missed-severe metric as p5_physical_correction_eval.py. capacity_floor
is gated in tests/test_capacity_floor.py before being trusted here; this script adds no new LP solves."""
import json
import numpy as np

from fdna import spec, physical_correction as pc
from fdna.dataset import G, RATING
from fdna.opgen import sample_op

n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]; full_idx, none_idx = int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]

# capacity_floor needs the full OperatingPoint (p0), not just demand -- same seeding as the confirmatory sample
# used throughout Phase 5 (op_id is the rng seed).
ops = {op: sample_op(G, RATING, np.random.default_rng(int(op))) for op in OP_IDS}

rows = []
for cv_idx, cv_name in ((full_idx, "full_control"), (none_idx, "no_control")):
    c = CV[cv_idx]
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id); a, b, c3 = outage
        y_true = float(V_sorted[i][cv_idx])
        y_singles = {b_: y1[(op_id, b_)][cv_idx] for b_ in outage}
        pairs = [(min(a, b), max(a, b)), (min(a, c3), max(a, c3)), (min(b, c3), max(b, c3))]
        y_pairs = {p: y2[(op_id, p)][cv_idx] for p in pairs}
        g2_plain = sum(y_singles.values()) + sum(y_pairs[p] - y_singles[p[0]] - y_singles[p[1]] for p in pairs)
        op = ops[op_id]
        f_isl = pc.island_floor(G, op.demand, outage)
        f_cap = pc.capacity_floor(G, op, spec.PARAMS, outage, c)
        g2_clip_isl = float(np.clip(g2_plain, f_isl, 1.0))
        g2_clip_cap = float(np.clip(g2_plain, f_cap, 1.0))
        rows.append(dict(cv=cv_name, op=op_id, outage=outage, y_true=y_true, f_island=f_isl, f_capacity=f_cap,
                          pred_plain=g2_plain, pred_clip_island=g2_clip_isl, pred_clip_capacity=g2_clip_cap))

out = {}
for cv_name in ("full_control", "no_control"):
    sub = [r for r in rows if r["cv"] == cv_name]
    yt = np.array([r["y_true"] for r in sub])
    sev = yt > spec.SEVERE
    d = {"n": len(sub), "n_severe": int(sev.sum())}
    f_isl = np.array([r["f_island"] for r in sub]); f_cap = np.array([r["f_capacity"] for r in sub])
    d["n_triples_with_capacity_gt_island_floor"] = int((f_cap > f_isl + 1e-9).sum())
    for k in ("plain", "clip_island", "clip_capacity"):
        pred = np.array([r[f"pred_{k}"] for r in sub])
        e = pred - yt
        missed = sev & (pred <= spec.SEVERE)
        d[k] = {"mae": float(np.abs(e).mean()), "missed_severe": int(missed.sum())}
    out[cv_name] = d
    print(cv_name, json.dumps(d, indent=1))
json.dump(out, open("results/phase5/capacity_floor_eval.json", "w"), indent=1)
