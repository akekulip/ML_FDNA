"""Phase 5 Stage 3: bounded adaptive-query experiment (registry/phase5_adaptive_query.yaml). Uses ONLY cached
n1/n2/n3_confirm.npz tables (0 new LP solves) -- a "query" is a table lookup against the exact value an LP
solve would return.

Fixed from the first attempt: stopping/resolution is now defined over the ACTUAL posterior p(c|obs) for this
operating point's shared observation (common-snapshot setting), not the whole 1024-grid uniformly -- a
candidate is "resolved" once qL(pc)==qU(pc) (posterior probability mass agrees on severe/not-severe), which is
the qL/qU quantity the brief actually specifies, not an unconditional per-control resolution.

Policies: g2_guided_bounds (next query = unqueried control nearest g2's own prediction to tau, restricted to
controls with nonzero posterior mass), random_bounds (uniformly random among posterior-supported unqueried
controls), direct_mc (K posterior draws, accumulate fraction severe, no bounds). oracle_full excluded (labeled
ceiling only, not a practical policy)."""
import json
import numpy as np

from fdna import spec, v2, v2data
from fdna.adaptive_query import bounds, posterior_mass_bounds
from fdna.physical_correction import island_floor
from fdna.dataset import G, RATING
from fdna.opgen import sample_op

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
BUDGETS = [2, 4, 8, 16, 32]
rng_global = np.random.default_rng(0)


def g2_full_grid(op_id, outage):
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)], y1[(op_id, b)], y1[(op_id, c)]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iv = [y2[(op_id, p)] - y1[(op_id, p[0])] - y1[(op_id, p[1])] for p in pairs]
    return ya + yb + yc + sum(Iv)


def resolve(policy, y_true_grid, floor, g2grid, pc_support, budget, guided_rng):
    """Adaptively query up to `budget` controls; stop once qL==qU over pc_support (the posterior-supported
    control set). Returns (n_queries_used, final_qL, final_qU, L, U)."""
    queried_idx, queried_val = [], []
    extremes = [int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))]
    for e in extremes:
        queried_idx.append(e); queried_val.append(y_true_grid[e])
    L, U = bounds(CV, np.array(queried_idx), np.array(queried_val), floor)
    while len(queried_idx) < budget:
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
    qL, qU = posterior_mass_bounds(pc_support / pc_support.sum() if pc_support.sum() > 0 else pc_support, L, U, spec.SEVERE)
    return len(queried_idx), qL, qU, L, U


CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
K_MC = 8
results = {}
N_OPS_USED = 20
for cell, (q, s, cov) in CELLS.items():
    for budget in BUDGETS:
        tot_q_guided = tot_q_random = tot_q_mc = 0
        correct_guided = correct_random = correct_mc = n_total = 0
        for op_id in OP_IDS[:N_OPS_USED]:
            rows_this_op = [(i, outages_sorted[i], V_sorted[i].astype(np.float64), island_floor(G, demands[op_id], outages_sorted[i]))
                             for i in range(len(op_ids_sorted)) if int(op_ids_sorted[i]) == op_id]
            rng = np.random.default_rng([op_id, hash(cell) % 1000 + budget if False else (1 if cell.startswith("P1") else 2), budget])
            st = v2.sample_states(w, rng, 1)
            obs = v2.emit(w, st, q, s, rng, cov)
            pc = v2data.oracle_features(w, obs, s)[0][0]
            true_c_idx = int(w.cidx[st][0])
            guided_rng = np.random.default_rng([op_id, budget, 7])

            for row_idx, outage, y_true_grid, floor in rows_this_op:
                g2grid = g2_full_grid(op_id, outage)
                true_label = y_true_grid[true_c_idx] > spec.SEVERE

                nq_g, qLg, qUg, Lg, Ug = resolve("g2_guided_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng)
                nq_r, qLr, qUr, Lr, Ur = resolve("random_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng)
                tot_q_guided += nq_g; tot_q_random += nq_r

                pred_g = Lg[true_c_idx] > spec.SEVERE if Lg[true_c_idx] > spec.SEVERE or Ug[true_c_idx] <= spec.SEVERE else (qLg + qUg) / 2 > 0.5
                pred_r = Lr[true_c_idx] > spec.SEVERE if Lr[true_c_idx] > spec.SEVERE or Ur[true_c_idx] <= spec.SEVERE else (qLr + qUr) / 2 > 0.5
                correct_guided += int(bool(pred_g) == bool(true_label)); correct_random += int(bool(pred_r) == bool(true_label))

                mc_rng = np.random.default_rng([op_id, *outage, budget, 99])
                st_mc = v2.sample_states(w, mc_rng, min(K_MC, budget))
                obs_mc = v2.emit(w, st_mc, q, s, mc_rng, cov)
                vals = y_true_grid[w.cidx[st_mc]]
                tot_q_mc += len(vals)
                correct_mc += int((vals > spec.SEVERE).mean() > 0.5) == bool(true_label)
                n_total += 1
        results.setdefault(cell, {})[budget] = {
            "n_candidates": n_total,
            "guided_queries_total": tot_q_guided, "random_queries_total": tot_q_random, "mc_queries_total": tot_q_mc,
            "guided_queries_per_candidate": tot_q_guided / n_total, "random_queries_per_candidate": tot_q_random / n_total,
            "guided_acc": correct_guided / n_total, "random_acc": correct_random / n_total, "mc_acc": correct_mc / n_total,
        }
        print(cell, budget, results[cell][budget], flush=True)

results["n_ops_used"] = N_OPS_USED
results["note"] = "resolution/stopping is defined over the posterior-supported control set (qL==qU), not the whole 1024-grid; queries_per_candidate < budget indicates genuine early stopping."
json.dump(results, open("results/phase5/adaptive_query_experiment.json", "w"), indent=1)
