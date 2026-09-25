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
into or confused with the target-query counts."""
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
    """Adaptively query up to `budget` controls; stop once qL==qU over pc_support. Returns
    (n_queries_used, qL, qU) ONLY -- L/U at the true index are never exposed to the caller's prediction logic."""
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

    for budget in BUDGETS:
        tot_q_guided = tot_q_random = tot_q_mc = 0
        correct_guided = correct_random = correct_mc = n_total = 0
        for op_id in OP_IDS[:N_OPS_USED]:
            rows_this_op = [(i, outages_sorted[i], V_sorted[i].astype(np.float64), island_floor(G, demands[op_id], outages_sorted[i]))
                             for i in range(len(op_ids_sorted)) if int(op_ids_sorted[i]) == op_id]
            pc = frozen_pc[op_id]; true_c_idx = frozen_true_c_idx[op_id]
            guided_rng = np.random.default_rng([op_id, budget, 7])

            for row_idx, outage, y_true_grid, floor in rows_this_op:
                g2grid = g2_full_grid(op_id, outage)
                true_label = y_true_grid[true_c_idx] > spec.SEVERE   # hidden truth, used ONLY to SCORE below

                nq_g, qLg, qUg = resolve("g2_guided_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng)
                nq_r, qLr, qUr = resolve("random_bounds", y_true_grid, floor, g2grid, pc, budget, guided_rng)
                tot_q_guided += nq_g; tot_q_random += nq_r

                # FIX 1: prediction from (qL,qU) ONLY -- no access to true_c_idx anywhere in this expression.
                pred_g = (qLg > 0.5) if qLg > 0.5 else ((qLg + qUg) / 2 > 0.5)
                pred_r = (qLr > 0.5) if qLr > 0.5 else ((qLr + qUr) / 2 > 0.5)
                correct_guided += int(bool(pred_g) == bool(true_label)); correct_random += int(bool(pred_r) == bool(true_label))

                # FIX 2: draw K=budget control indices DIRECTLY from the exact posterior pc -- true posterior MC.
                mc_rng = np.random.default_rng([op_id, *outage, budget, 99])
                mc_idx = mc_rng.choice(n_cv, size=budget, replace=True, p=pc / pc.sum())
                vals = y_true_grid[mc_idx]
                tot_q_mc += budget
                pred_mc = (vals > spec.SEVERE).mean() > 0.5
                correct_mc += int(bool(pred_mc) == bool(true_label))
                n_total += 1
        results.setdefault(cell, {})[budget] = {
            "n_candidates": n_total,
            "guided_queries_total": tot_q_guided, "random_queries_total": tot_q_random, "mc_queries_total": tot_q_mc,
            "guided_queries_per_candidate": tot_q_guided / n_total, "random_queries_per_candidate": tot_q_random / n_total,
            "guided_acc": correct_guided / n_total, "random_acc": correct_random / n_total, "mc_acc": correct_mc / n_total,
        }
        print(cell, budget, results[cell][budget], flush=True)

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
