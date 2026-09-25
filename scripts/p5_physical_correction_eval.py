"""Phase 5 Stage 2: compare g2_plain / g2_clipped / g2_residual on the CACHED confirmatory tables (no new LP
solves). Reports uniform and posterior-weighted (per-op mean-composed) error, stratified by island class, cut
type, and residual sign. Predeclared exploratory gate: 20% relative missed-severe reduction, no material
R-precision loss."""
import json
import numpy as np

from fdna import spec, v2, v2data, physical_correction as pc
from fdna.dataset import G, RATING
from fdna.evalutil import rprec, scenario_uniform
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

demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in OP_IDS}

def classify(op_id, outage):
    f = pc.island_floor(G, demands[op_id], outage)
    from fdna.hik_diag import minimal_cut_struct
    _, n_isl = minimal_cut_struct(G, outage)
    if n_isl == 1: return "connected"
    return "gen_less_only" if f > 1e-9 else "all_gen_or_mixed"

rows = []
for cv_idx, cv_name in ((full_idx, "full_control"), (none_idx, "no_control")):
    for i, (op_id, outage) in enumerate(zip(op_ids_sorted, outages_sorted)):
        op_id = int(op_id); a, b, c = outage
        y_true = float(V_sorted[i][cv_idx])
        y_singles = {b_: y1[(op_id, b_)][cv_idx] for b_ in outage}
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        y_pairs = {p: y2[(op_id, p)][cv_idx] for p in pairs}
        g2_plain = sum(y_singles.values()) + sum(y_pairs[p] - y_singles[p[0]] - y_singles[p[1]] for p in pairs)
        g2_clip = pc.g2_clipped(g2_plain, G, demands[op_id], outage)
        g2_res = pc.g2_residual(y_singles, y_pairs, G, demands[op_id], outage)
        cls = classify(op_id, outage)
        # FIX (external review, finding 7): store predictions directly, never reconstruct pred = y - err
        # (that reconstruction was computing 2*y_true - pred, silently hiding real missed-severe cases --
        # e.g. y_true=0.05, pred=0 reconstructed to 0.10 and was not counted as missed).
        rows.append(dict(cv=cv_name, op=op_id, outage=outage, cls=cls, y_true=y_true,
                          pred_plain=g2_plain, pred_clip=g2_clip, pred_res=g2_res,
                          err_plain=g2_plain - y_true, err_clip=g2_clip - y_true, err_res=g2_res - y_true))

out = {}
for cv_name in ("full_control", "no_control"):
    sub = [r for r in rows if r["cv"] == cv_name]
    d = {"n": len(sub)}
    for cls in ("connected", "gen_less_only", "all_gen_or_mixed"):
        s2 = [r for r in sub if r["cls"] == cls]
        if not s2: continue
        d[cls] = {"n": len(s2)}
        for k in ("err_plain", "err_clip", "err_res"):
            e = np.array([r[k] for r in s2])
            d[cls][k] = {"mae": float(np.abs(e).mean()), "mean_signed": float(e.mean())}
    for k in ("err_plain", "err_clip", "err_res"):
        e = np.array([r[k] for r in sub])
        d[k + "_overall_mae"] = float(np.abs(e).mean())
    # missed-severe: severe = y_true>0.01; "missed" = predicted <= 0.01 when true is severe
    # FIX (external review, finding 7): use the stored prediction directly, no reconstruction from err.
    yt = np.array([r["y_true"] for r in sub])
    sev = yt > spec.SEVERE
    for k in ("plain", "clip", "res"):
        pred = np.array([r[f"pred_{k}"] for r in sub])
        missed = sev & (pred <= spec.SEVERE)
        d[f"missed_severe_{k}"] = int(missed.sum())
    d["n_severe"] = int(sev.sum())
    out[cv_name] = d
    print(cv_name, json.dumps(d, indent=1))
json.dump(out, open("results/phase5/physical_correction_eval.json", "w"), indent=1)
