"""Phase 5 Stage R1.2: adaptive-query experiment, v2 -- fixes four bugs an external review found in v1 (kept,
not deleted, for the record):

1. HIDDEN-STATE LEAKAGE (critical): v1's prediction indexed Lg[true_c_idx]/Ug[true_c_idx], the REALIZED hidden
   control index -- information no real policy can access. Fixed: prediction is computed ONLY from (qL,qU),
   the posterior-mass bounds over the observation-conditioned pc; the hidden state is used ONLY afterwards, to
   score correctness against true_label.
2. "Posterior MC" sampled the PRIOR (v2.sample_states draws from w.logprior unconditionally; obs_mc was
   generated but never used to condition anything). Fixed: draw K control indices directly from the ALREADY-
   COMPUTED exact posterior pc (np.random.choice(..., p=pc)) -- exact posterior Monte Carlo, no approximation.
3. The observation was redrawn per BUDGET (budget was part of the RNG seed), so different budgets scored a
   different workload, not just a different spending limit. Fixed: one frozen observation per (op,cell),
   reused identically across every budget.
4. Script/registry mismatch (60 ops / budgets {2,4,6,8,12,16} registered vs 20 ops / {2,4,8,16,32} run). Fixed:
   script now matches the (also-fixed) registry exactly. MC now uses the SAME budget as the bound policies
   (was capped at 8 regardless of budget), for a fair matched-budget comparison.

Cost accounting (external review finding 6/10): the N-3 query counts below are the ONLY thing charged against
the budget; the lower-order (singleton/pair) table each operating point's g2 guidance reads from is reported
SEPARATELY, as its true size (41 singles + 182 pairs, x1024 controls = 228,352 scalar values per op), not folded
into or confused with the target-query counts.

Repair round 2 (independent second review) adds four things the registry itself declared but v1 never
implemented, plus one requested cheap diagnostic:
5. Decision-aware stopping in `resolve()` (a binary decision certified by qL>0.5 or qU<=tau no longer keeps
   querying past that point -- see the function's own docstring).
6. Unique-query / cache accounting for the MC policy: `mc_queries_total` (draws charged, unchanged) vs
   `mc_unique_queries_total` (distinct control indices among those draws -- a cache could serve the rest).
7. Shortlist recall/precision at 10/20/40% (the registry's own declared metric, absent from v1's output): for
   each policy, pool all candidates in a cell/budget, rank by that policy's own continuous severity score
   (guided/random: (qL+qU)/2; mc: empirical severe-fraction of its draws), score against the true label via
   `fdna.evalutil.recall_at`/`precision_at`. One reasonable operationalization of the registry's field, not
   claimed to be the only one.
8. A paired-per-op bootstrap CI on (mc_acc - guided_acc) and (mc_acc - random_acc) at every budget -- v1's
   "MC beats both bound-based policies at every budget" claim had no saved uncertainty bound.
9. A cheap, CACHED-DATA-ONLY diagnostic (not a new adaptive policy) for the review's proposed posterior
   control-variate estimator `p_hat = E_p[h0] + mean(h(c_j)-h0(c_j))`, h0(c)=1[g2(c)>tau], h(c)=1[V(c)>tau]:
   reports Var_p[h-h0] vs Var_p[h] (computed exactly under the posterior, no simulation needed) -- a favorable
   ratio (<1) means the control variate is worth implementing as an actual policy; this round only measures it."""
import json
import numpy as np

from fdna import spec, v2, v2data
from fdna.adaptive_query import bounds, posterior_mass_bounds, predict_from_bounds
from fdna.evalutil import recall_at, precision_at, scenario_uniform
from fdna.physical_correction import island_floor
from fdna.dataset import G, RATING
from fdna.opgen import sample_op
from fdna.rules import boot

n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]
demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in OP_IDS}

w = v2.build_world(); n_cv = len(CV)
BUDGETS = [2, 4, 6, 8, 12, 16]
CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}
LOWER_ORDER_TABLE_SIZE_PER_OP = (41 + 182) * 1024   # 41 singles + 182 pairs, full control grid: 228,352


def g2_full_grid(op_id, outage):
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iv = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
    return ya + yb + yc + sum(Iv)


def resolve(policy, y_true_grid, floor, g2grid, pc_support, budget, guided_rng):
    """Adaptively query up to `budget` controls. Returns (n_queries_used, qL, qU) ONLY -- L/U at the true index
    are never exposed to the caller's prediction logic.

    Repair round 2 (external review): stopping used to wait until every posterior-support control state was
    individually resolved (L==U), which is strictly STRONGER than what the binary decision needs. The binary
    prediction (`predict_from_bounds`) only needs qL>0.5 or qU<=tau to be ALREADY true to be certified --
    querying further cannot change that decision. Decision-aware early exit added below (checked BEFORE the
    full-resolution check, so it fires first whenever it applies)."""
    pn = pc_support / pc_support.sum() if pc_support.sum() > 0 else pc_support
    queried_idx, queried_val = [], []
    extremes = [int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))]
    for e in extremes:
        queried_idx.append(e); queried_val.append(y_true_grid[e])
    L, U = bounds(CV, np.array(queried_idx), np.array(queried_val), floor)
    while len(queried_idx) < budget:
        qL, qU = posterior_mass_bounds(pn, L, U, spec.SEVERE)
        if qL > 0.5 or qU <= spec.SEVERE:
            break  # decision already certified -- further queries cannot change the binary prediction
        support = np.flatnonzero(pc_support)
        unresolved_support = support[(L[support] <= spec.SEVERE) & (U[support] > spec.SEVERE)]
        if len(unresolved_support) == 0:
            break
        unqueried_unresolved = np.array([i for i in unresolved_support if i not in queried_idx])
        if len(unqueried_unresolved) == 0:
            break
        if policy == "g2_guided_bounds":
            nxt = unqueried_unresolved[np.argmin(np.abs(g2grid[unqueried_unresolved] - spec.SEVERE))]
        else:
            nxt = unqueried_unresolved[guided_rng.integers(len(unqueried_unresolved))]
        queried_idx.append(int(nxt)); queried_val.append(y_true_grid[nxt])
        L, U = bounds(CV, np.array(queried_idx), np.array(queried_val), floor)
    qL, qU = posterior_mass_bounds(pn, L, U, spec.SEVERE)
    return len(queried_idx), qL, qU


CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
results = {}
N_OPS_USED = 60   # matches the registry exactly now

for cell, (q, s, cov) in CELLS.items():
    # FIX 3: draw the shared observation ONCE per op, NOT per budget -- reused identically across every budget below.
    frozen_pc, frozen_true_c_idx = {}, {}
    for op_id in OP_IDS[:N_OPS_USED]:
        rng = np.random.default_rng([op_id, CELL_SEED[cell]])
        st = v2.sample_states(w, rng, 1)
        obs = v2.emit(w, st, q, s, rng, cov)
        frozen_pc[op_id] = v2data.oracle_features(w, obs, s)[0][0]
        frozen_true_c_idx[op_id] = int(w.cidx[st][0])

    # ---- control-variate diagnostic (T5.4): CACHED-DATA ONLY, independent of any query budget -- computed
    # exactly under the full posterior pc, no simulation needed. h0(c)=1[g2(c)>tau], h(c)=1[V(c)>tau]; a
    # favorable control variate has Var_p[h-h0] < Var_p[h] (ratio < 1). Diagnostic only -- no adaptive policy
    # is built around this in this round.
    cv_ratios = []
    for op_id in OP_IDS[:N_OPS_USED]:
        pc = frozen_pc[op_id]; pn = pc / pc.sum() if pc.sum() > 0 else pc
        for i in range(len(op_ids_sorted)):
            if int(op_ids_sorted[i]) != op_id:
                continue
            outage = outages_sorted[i]; y_true_grid = V_sorted[i].astype(np.float64)
            g2grid = g2_full_grid(op_id, outage)
            h0 = (g2grid > spec.SEVERE).astype(np.float64); h = (y_true_grid > spec.SEVERE).astype(np.float64)
            diff = h - h0
            var_diff = float((pn * diff ** 2).sum() - (pn * diff).sum() ** 2)
            var_h = float((pn * h ** 2).sum() - (pn * h).sum() ** 2)
            if var_h > 1e-12:
                cv_ratios.append(var_diff / var_h)
    control_variate_diagnostic = {
        "n_candidates_measured": len(cv_ratios),
        "mean_var_ratio_diff_over_h": float(np.mean(cv_ratios)) if cv_ratios else None,
        "median_var_ratio_diff_over_h": float(np.median(cv_ratios)) if cv_ratios else None,
        "frac_candidates_favorable_ratio_lt_1": float(np.mean([r < 1.0 for r in cv_ratios])) if cv_ratios else None,
        "note": "ratio = Var_p[h(c)-h0(c)] / Var_p[h(c)] under the EXACT posterior pc, no draws; <1 means g2 as "
                "a control variate would reduce estimator variance vs plain posterior MC on h(c) alone. "
                "Diagnostic only -- no adaptive policy built around this yet (repair round 2, T5.4).",
    }
    print(cell, "control_variate_diagnostic", json.dumps(control_variate_diagnostic, indent=1), flush=True)

    for budget in BUDGETS:
        tot_q_guided = tot_q_random = tot_q_mc = tot_q_mc_unique = 0
        correct_guided = correct_random = correct_mc = n_total = 0
        # per-op correctness lists for a paired bootstrap CI (T5.8), and pooled shortlist scores (T5.3)
        per_op_correct = {op: {"guided": [], "random": [], "mc": []} for op in OP_IDS[:N_OPS_USED]}
        shortlist = {"guided": {"score": [], "label": [], "key": []}, "random": {"score": [], "label": [], "key": []},
                     "mc": {"score": [], "label": [], "key": []}}
        for op_id in OP_IDS[:N_OPS_USED]:
            rows_this_op = [(i, outages_sorted[i], V_sorted[i].astype(np.float64), island_floor(G, demands[op_id], outages_sorted[i]))
                             for i in range(len(op_ids_sorted)) if int(op_ids_sorted[i]) == op_id]
            pc = frozen_pc[op_id]; true_c_idx = frozen_true_c_idx[op_id]
            guided_rng = np.random.default_rng([op_id, budget, 7])

            for row_idx, outage, y_true_grid, floor in rows_this_op:
                g2grid = g2_full_grid(op_id, outage)
                true_label = y_true_grid[true_c_idx] > spec.SEVERE   # hidden truth, used ONLY to SCORE below
                cand_key = float(scenario_uniform(np.array([op_id]), np.array([outage[0]]), np.array([outage[1]]), np.array([outage[2]]), salt=17)[0])

                nq_g, qLg, qUg = resolve("g2_guided_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng)
                nq_r, qLr, qUr = resolve("random_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng)
                tot_q_guided += nq_g; tot_q_random += nq_r

                # FIX 1: prediction from (qL,qU) ONLY, via the shared fdna.adaptive_query.predict_from_bounds --
                # no access to true_c_idx anywhere (its signature has no such parameter to leak through).
                pred_g = predict_from_bounds(qLg, qUg)
                pred_r = predict_from_bounds(qLr, qUr)
                ok_g, ok_r = bool(pred_g) == bool(true_label), bool(pred_r) == bool(true_label)
                correct_guided += int(ok_g); correct_random += int(ok_r)
                per_op_correct[op_id]["guided"].append(ok_g); per_op_correct[op_id]["random"].append(ok_r)
                shortlist["guided"]["score"].append((qLg + qUg) / 2); shortlist["guided"]["label"].append(float(true_label)); shortlist["guided"]["key"].append(cand_key)
                shortlist["random"]["score"].append((qLr + qUr) / 2); shortlist["random"]["label"].append(float(true_label)); shortlist["random"]["key"].append(cand_key)

                # FIX 2: draw K=budget control indices DIRECTLY from the exact posterior pc -- true posterior MC.
                mc_rng = np.random.default_rng([op_id, *outage, budget, 99])
                mc_idx = mc_rng.choice(n_cv, size=budget, replace=True, p=pc / pc.sum())
                vals = y_true_grid[mc_idx]
                mc_score = (vals > spec.SEVERE).mean()
                tot_q_mc += budget; tot_q_mc_unique += len(set(mc_idx.tolist()))
                pred_mc = mc_score > 0.5
                ok_mc = bool(pred_mc) == bool(true_label)
                correct_mc += int(ok_mc)
                per_op_correct[op_id]["mc"].append(ok_mc)
                shortlist["mc"]["score"].append(float(mc_score)); shortlist["mc"]["label"].append(float(true_label)); shortlist["mc"]["key"].append(cand_key)
                n_total += 1

        # T5.3: shortlist recall/precision at the registry's declared 10/20/40% budgets, per policy, pooled
        # across all candidates in this cell/budget -- one reasonable operationalization of the registry's
        # field (rank by each policy's own continuous severity score), not claimed to be the only one.
        shortlist_metrics = {}
        for policy, d in shortlist.items():
            yt, yp, key = np.array(d["label"]), np.array(d["score"]), np.array(d["key"])
            shortlist_metrics[policy] = {
                f"{pct}pct": {"recall": recall_at(yt, yp, key, frac, thr=0.5), "precision": precision_at(yt, yp, key, frac, thr=0.5)}
                for pct, frac in ((10, 0.10), (20, 0.20), (40, 0.40))
            }

        # T5.8: paired-per-op bootstrap CI on (mc_acc - guided_acc) and (mc_acc - random_acc)
        per_op_mean = {op: {a: float(np.mean(per_op_correct[op][a])) for a in ("guided", "random", "mc")} for op in OP_IDS[:N_OPS_USED]}
        diff_mc_guided = np.array([per_op_mean[op]["mc"] - per_op_mean[op]["guided"] for op in OP_IDS[:N_OPS_USED]])
        diff_mc_random = np.array([per_op_mean[op]["mc"] - per_op_mean[op]["random"] for op in OP_IDS[:N_OPS_USED]])
        boot_ci = {"mc_minus_guided": boot(diff_mc_guided), "mc_minus_random": boot(diff_mc_random)}

        results.setdefault(cell, {})[budget] = {
            "n_candidates": n_total,
            "guided_queries_total": tot_q_guided, "random_queries_total": tot_q_random,
            "mc_queries_total": tot_q_mc, "mc_unique_queries_total": tot_q_mc_unique,
            "guided_queries_per_candidate": tot_q_guided / n_total, "random_queries_per_candidate": tot_q_random / n_total,
            "guided_acc": correct_guided / n_total, "random_acc": correct_random / n_total, "mc_acc": correct_mc / n_total,
            "shortlist_metrics": shortlist_metrics,
            "boot_ci_per_op": boot_ci,
        }
        print(cell, budget, {k: v for k, v in results[cell][budget].items() if k not in ("shortlist_metrics",)}, flush=True)
    results.setdefault(cell, {})["control_variate_diagnostic"] = control_variate_diagnostic

results["n_ops_used"] = N_OPS_USED
results["cost_accounting"] = {
    "lower_order_table_size_per_op_scalar_values": LOWER_ORDER_TABLE_SIZE_PER_OP,
    "note": "the guided/random/mc query counts above are ONLY the N-3 target queries charged against the budget. "
            "Building g2's guidance additionally requires the full lower-order table (41 singles + 182 pairs x "
            "1024 controls = 228,352 scalar values per operating point) -- an amortized, one-time-per-op cost, "
            "reported here separately and explicitly, not folded into or confused with the per-candidate query counts."
}
results["note"] = "prediction uses ONLY (qL,qU); true_c_idx is used exclusively to score correctness AFTER the prediction is frozen (external review finding 4 fix). MC draws directly from the exact posterior pc (finding 5 fix). Observation is frozen per (op,cell), reused across all budgets (finding 6 fix)."
json.dump(results, open("results/phase5/adaptive_query_experiment_v2.json", "w"), indent=1)
