"""Phase 5 repair round 3: implements the external review's proposed control-variate estimator as an ACTUAL
candidate policy (not only a diagnostic, as v2 did). Built on v2's already-fixed machinery (canonical `resolve`
in `fdna.adaptive_query`, per-op shortlist metrics, decision-aware stopping) -- v2 is kept unmodified on disk
alongside this new script, per the repo's established convention.

SCOPE (explicit, per the review's own guidance): this round is EXPLORATORY ONLY, reusing the already-open
600-659 operating-point block and the same 70-triple manifest -- no new reserve block, no fresh LP solves.
Confirmatory work on genuinely unused operating points/outages is future work, not attempted here.

For h(c)=1[V(c)>tau], h0(c)=1[g2(c)>tau], posterior p, and a FIXED coefficient beta:

    p_hat_beta = beta*E_p[h0] + mean_j(h(c_j) - beta*h0(c_j)),  c_j ~ p (the SAME K=budget draws MC already uses)

beta=0 reduces EXACTLY to plain posterior MC (asserted live below, not just claimed). beta=1 is the review's
proposed full correction. BETA_GRID = [0.0, 0.5, 1.0] is fixed and prespecified -- NOT validation-selected or
adaptive (the review's own caution: an adaptively-chosen beta needs a separated pilot-sample protocol, future
work, not attempted here). Two free companion arms: g2_zero_query (score=E_p[h0], zero N-3 queries) and
exact_posterior_ceiling (score=E_p[h] using the full truth table, tagged oracle_only -- never a deployable
policy). The control-variate diagnostic is also corrected here: EVERY candidate gets a row with a category
(zero_variance_tie / zero_variance_harmed_by_proxy / nondegenerate) -- v2's diagnostic silently dropped any
candidate with near-zero plain-MC variance before computing its ratio, hiding exactly the cases where a poor
surrogate could introduce variance where none existed (external review finding 3, confirmed with the review's
own worked example: pn=[.5,.5], h=[0,0], h0=[0,1] -> var_h=0, var_diff=0.25, a real excluded-harm case)."""
import json
import numpy as np

from fdna import spec, v2, v2data
from fdna.adaptive_query import resolve, predict_from_bounds
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
N_OPS_USED = 60
BETA_GRID = [0.0, 0.5, 1.0]
CV_POLICIES = ["guided", "random", "mc"] + [f"cv_b{b}" for b in BETA_GRID] + ["g2_zero_query", "exact_posterior_ceiling"]


def g2_full_grid(op_id, outage):
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iv = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
    return ya + yb + yc + sum(Iv)


def cv_diagnostic_row(op_id, outage, pn, y_true_grid, g2grid, tau=spec.SEVERE):
    """One row of the FULL variance panel -- every candidate gets a category, never a silent drop."""
    h = (y_true_grid > tau).astype(np.float64); h0 = (g2grid > tau).astype(np.float64)
    Eh, Eh0 = float((pn * h).sum()), float((pn * h0).sum())
    Ehh0 = float((pn * h * h0).sum())
    var_h = float((pn * h ** 2).sum() - Eh ** 2); var_h0 = float((pn * h0 ** 2).sum() - Eh0 ** 2)
    cov = Ehh0 - Eh * Eh0
    var_diff = var_h + var_h0 - 2 * cov
    disagreement_mass = float((pn * np.abs(h - h0)).sum())
    if var_h <= 1e-12:
        category = "zero_variance_harmed_by_proxy" if var_diff > 1e-12 else "zero_variance_tie"
        ratio = None
    else:
        category, ratio = "nondegenerate", var_diff / var_h
    return dict(op_id=op_id, outage=list(outage), var_h=var_h, var_h0=var_h0, var_diff=var_diff,
                cov_h_h0=cov, E_p_h=Eh, E_p_h0=Eh0, disagreement_mass=disagreement_mass,
                category=category, ratio=ratio)


CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
results = {}

for cell, (q, s, cov) in CELLS.items():
    frozen_pc, frozen_true_c_idx = {}, {}
    for op_id in OP_IDS[:N_OPS_USED]:
        rng = np.random.default_rng([op_id, CELL_SEED[cell]])
        st = v2.sample_states(w, rng, 1)
        obs = v2.emit(w, st, q, s, rng, cov)
        frozen_pc[op_id] = v2data.oracle_features(w, obs, s)[0][0]
        frozen_true_c_idx[op_id] = int(w.cidx[st][0])

    # full variance panel -- every (op,outage) candidate, no exclusion
    cv_rows = []
    for op_id in OP_IDS[:N_OPS_USED]:
        pc = frozen_pc[op_id]; pn = pc / pc.sum() if pc.sum() > 0 else pc
        for i in range(len(op_ids_sorted)):
            if int(op_ids_sorted[i]) != op_id:
                continue
            outage = outages_sorted[i]; y_true_grid = V_sorted[i].astype(np.float64)
            g2grid = g2_full_grid(op_id, outage)
            cv_rows.append(cv_diagnostic_row(op_id, outage, pn, y_true_grid, g2grid))
    by_cat = {"zero_variance_tie": 0, "zero_variance_harmed_by_proxy": 0, "nondegenerate": 0}
    for r in cv_rows:
        by_cat[r["category"]] += 1
    nondeg = [r["ratio"] for r in cv_rows if r["category"] == "nondegenerate"]
    total_var_h = sum(r["var_h"] for r in cv_rows); total_var_diff = sum(r["var_diff"] for r in cv_rows)
    control_variate_diagnostic = {
        "n_candidates_total": len(cv_rows), "n_by_category": by_cat,
        "summary_nondegenerate_only": {
            "mean_ratio": float(np.mean(nondeg)) if nondeg else None,
            "median_ratio": float(np.median(nondeg)) if nondeg else None,
            "frac_favorable_lt_1": float(np.mean([r < 1.0 for r in nondeg])) if nondeg else None,
        },
        "ratio_of_summed_variances_all_candidates": (total_var_diff / total_var_h) if total_var_h > 0 else None,
        "note": "summary_nondegenerate_only is computed ONLY over n_by_category['nondegenerate'] candidates, "
                "explicitly labeled as such (external review finding 3: v2's diagnostic silently excluded "
                "zero-variance candidates, including cases where a poor surrogate could introduce variance "
                "where none existed). ratio_of_summed_variances_all_candidates is a stabler pooled summary over "
                "EVERY candidate, avoiding the near-zero-denominator instability of the old mean-of-ratios.",
        "rows": cv_rows,
    }
    print(cell, "control_variate_diagnostic (full panel)",
          json.dumps({k: v for k, v in control_variate_diagnostic.items() if k != "rows"}, indent=1), flush=True)

    for budget in BUDGETS:
        tot_q = {p: 0 for p in CV_POLICIES}
        tot_q_unique = {"mc": 0}
        correct = {p: 0 for p in CV_POLICIES}
        n_total = 0
        per_op_correct = {op: {p: [] for p in CV_POLICIES} for op in OP_IDS[:N_OPS_USED]}
        shortlist_by_op = {p: {op: {"score": [], "label": [], "key": []} for op in OP_IDS[:N_OPS_USED]} for p in CV_POLICIES}

        for op_id in OP_IDS[:N_OPS_USED]:
            rows_this_op = [(i, outages_sorted[i], V_sorted[i].astype(np.float64), island_floor(G, demands[op_id], outages_sorted[i]))
                             for i in range(len(op_ids_sorted)) if int(op_ids_sorted[i]) == op_id]
            pc = frozen_pc[op_id]; true_c_idx = frozen_true_c_idx[op_id]
            pn = pc / pc.sum() if pc.sum() > 0 else pc

            for row_idx, outage, y_true_grid, floor in rows_this_op:
                g2grid = g2_full_grid(op_id, outage)
                true_label = y_true_grid[true_c_idx] > spec.SEVERE
                cand_key = float(scenario_uniform(np.array([op_id]), np.array([outage[0]]), np.array([outage[1]]), np.array([outage[2]]), salt=17)[0])
                guided_rng = np.random.default_rng([op_id, *outage, budget, 7])

                nq_g, qLg, qUg = resolve("g2_guided_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng, CV, spec.SEVERE)
                nq_r, qLr, qUr = resolve("random_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng, CV, spec.SEVERE)
                tot_q["guided"] += nq_g; tot_q["random"] += nq_r
                score_g, score_r = (qLg + qUg) / 2, (qLr + qUr) / 2
                ok_g = bool(predict_from_bounds(qLg, qUg)) == bool(true_label)
                ok_r = bool(predict_from_bounds(qLr, qUr)) == bool(true_label)
                correct["guided"] += int(ok_g); correct["random"] += int(ok_r)
                per_op_correct[op_id]["guided"].append(ok_g); per_op_correct[op_id]["random"].append(ok_r)
                for p, sc in (("guided", score_g), ("random", score_r)):
                    d = shortlist_by_op[p][op_id]
                    d["score"].append(sc); d["label"].append(float(true_label)); d["key"].append(cand_key)

                # SAME draws as plain MC -- with replacement, multiplicity preserved -- reused for every cv_b*
                mc_rng = np.random.default_rng([op_id, *outage, budget, 99])
                mc_idx = mc_rng.choice(n_cv, size=budget, replace=True, p=pn)
                h_at_draws = (y_true_grid[mc_idx] > spec.SEVERE).astype(np.float64)
                h0_grid = (g2grid > spec.SEVERE).astype(np.float64)
                h0_at_draws = h0_grid[mc_idx]
                E_p_h0 = float((pn * h0_grid).sum())
                mc_score = float(h_at_draws.mean())
                tot_q["mc"] += budget; tot_q_unique["mc"] += len(set(mc_idx.tolist()))
                ok_mc = bool(mc_score > 0.5) == bool(true_label)
                correct["mc"] += int(ok_mc); per_op_correct[op_id]["mc"].append(ok_mc)
                d = shortlist_by_op["mc"][op_id]; d["score"].append(mc_score); d["label"].append(float(true_label)); d["key"].append(cand_key)

                p_hat_by_beta = {}
                for beta in BETA_GRID:
                    p_hat = beta * E_p_h0 + float((h_at_draws - beta * h0_at_draws).mean())
                    p_hat_by_beta[beta] = p_hat
                assert abs(p_hat_by_beta[0.0] - mc_score) < 1e-12, "beta=0 must reduce exactly to plain posterior MC"
                for beta in BETA_GRID:
                    key_b = f"cv_b{beta}"
                    p_hat = p_hat_by_beta[beta]
                    ok_cv = bool(p_hat > 0.5) == bool(true_label)
                    correct[key_b] += int(ok_cv); per_op_correct[op_id][key_b].append(ok_cv)
                    tot_q[key_b] += budget  # reuses the SAME mc_idx draws -- zero additional oracle cost
                    d = shortlist_by_op[key_b][op_id]
                    d["score"].append(float(np.clip(p_hat, 0.0, 1.0))); d["label"].append(float(true_label)); d["key"].append(cand_key)

                # two free companion arms
                ok_zero = bool(E_p_h0 > 0.5) == bool(true_label)
                correct["g2_zero_query"] += int(ok_zero); per_op_correct[op_id]["g2_zero_query"].append(ok_zero)
                d = shortlist_by_op["g2_zero_query"][op_id]; d["score"].append(E_p_h0); d["label"].append(float(true_label)); d["key"].append(cand_key)

                E_p_h_true = float((pn * (y_true_grid > spec.SEVERE)).sum())
                ok_ceil = bool(E_p_h_true > 0.5) == bool(true_label)
                correct["exact_posterior_ceiling"] += int(ok_ceil); per_op_correct[op_id]["exact_posterior_ceiling"].append(ok_ceil)
                d = shortlist_by_op["exact_posterior_ceiling"][op_id]; d["score"].append(E_p_h_true); d["label"].append(float(true_label)); d["key"].append(cand_key)

                n_total += 1

        per_op_shortlist = {p: {f"{pct}pct": {"recall": {}, "precision": {}} for pct in (10, 20, 40)} for p in CV_POLICIES}
        for p in CV_POLICIES:
            for op in OP_IDS[:N_OPS_USED]:
                d = shortlist_by_op[p][op]
                yt, yp, key = np.array(d["label"]), np.array(d["score"]), np.array(d["key"])
                for pct, frac in ((10, 0.10), (20, 0.20), (40, 0.40)):
                    per_op_shortlist[p][f"{pct}pct"]["recall"][op] = recall_at(yt, yp, key, frac, thr=0.5)
                    per_op_shortlist[p][f"{pct}pct"]["precision"][op] = precision_at(yt, yp, key, frac, thr=0.5)
        shortlist_metrics_per_op = {
            p: {f"{pct}pct": {
                    "recall_mean_across_ops": float(np.nanmean(list(per_op_shortlist[p][f"{pct}pct"]["recall"].values()))),
                    "precision_mean_across_ops": float(np.mean(list(per_op_shortlist[p][f"{pct}pct"]["precision"].values()))),
                } for pct in (10, 20, 40)}
            for p in CV_POLICIES
        }

        per_op_mean = {op: {p: float(np.mean(per_op_correct[op][p])) for p in CV_POLICIES} for op in OP_IDS[:N_OPS_USED]}
        boot_ci, shortlist_boot_ci = {}, {}
        for beta in BETA_GRID:
            k = f"cv_b{beta}"
            boot_ci[f"{k}_minus_mc"] = boot(np.array([per_op_mean[op][k] - per_op_mean[op]["mc"] for op in OP_IDS[:N_OPS_USED]]))
            boot_ci[f"{k}_minus_guided"] = boot(np.array([per_op_mean[op][k] - per_op_mean[op]["guided"] for op in OP_IDS[:N_OPS_USED]]))
            for pct in (10, 20, 40):
                pk = f"{pct}pct"
                cv_recall = np.array([per_op_shortlist[k][pk]["recall"][op] for op in OP_IDS[:N_OPS_USED]])
                mc_recall = np.array([per_op_shortlist["mc"][pk]["recall"][op] for op in OP_IDS[:N_OPS_USED]])
                valid = ~(np.isnan(cv_recall) | np.isnan(mc_recall))
                shortlist_boot_ci[f"recall_{pk}_{k}_minus_mc"] = boot((cv_recall - mc_recall)[valid])
                cv_prec = np.array([per_op_shortlist[k][pk]["precision"][op] for op in OP_IDS[:N_OPS_USED]])
                mc_prec = np.array([per_op_shortlist["mc"][pk]["precision"][op] for op in OP_IDS[:N_OPS_USED]])
                shortlist_boot_ci[f"precision_{pk}_{k}_minus_mc"] = boot(cv_prec - mc_prec)
        # sanity: beta=0 must be an EXACT per-op tie with mc (not just close)
        assert boot_ci["cv_b0.0_minus_mc"]["mean"] == 0.0, boot_ci["cv_b0.0_minus_mc"]

        results.setdefault(cell, {})[budget] = {
            "n_candidates": n_total,
            "queries_total": tot_q, "mc_unique_queries_total": tot_q_unique["mc"],
            "acc": {p: correct[p] / n_total for p in CV_POLICIES},
            "shortlist_metrics_per_op": shortlist_metrics_per_op,
            "boot_ci_per_op": boot_ci,
            "shortlist_boot_ci_per_op": shortlist_boot_ci,
        }
        print(cell, budget, {"acc": results[cell][budget]["acc"], "boot_ci_per_op": boot_ci}, flush=True)
    results.setdefault(cell, {})["control_variate_diagnostic"] = control_variate_diagnostic

results["n_ops_used"] = N_OPS_USED
results["beta_grid"] = BETA_GRID
results["cost_accounting"] = {
    "cv_beta_note": "control-variate arms (cv_b*) reuse the IDENTICAL mc_idx draws as the mc policy at the "
                    "same budget -- zero additional oracle queries. h0(c_j)=1[g2(c_j)>tau] is a deterministic "
                    "lookup on the already-cached g2 grid, not a new LP solve. mc_queries_total/"
                    "mc_unique_queries_total apply unchanged to every cv_b* arm. exact_posterior_ceiling and "
                    "g2_zero_query use zero N-3 target queries (the ceiling uses the full cached truth table; "
                    "it is an oracle diagnostic, never a deployable policy).",
    "scope_note": "EXPLORATORY ONLY -- reuses the already-open 600-659 operating-point block and the existing "
                   "70-triple manifest, no new reserve, no fresh LP solves. Confirmatory work on genuinely "
                   "unused operating points/outages is future work, not attempted this round.",
}
json.dump(results, open("results/phase5/adaptive_query_experiment_v3.json", "w"), indent=1)
print("V3 DONE", flush=True)
