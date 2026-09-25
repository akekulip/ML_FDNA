# Phase 5 Stage 3 — bounded adaptive-query screening

**This document was corrected after an external evidence review found two critical bugs** (hidden-state leakage
in the prediction logic, and a "posterior MC" baseline that actually sampled the prior) plus several execution
mismatches. Every claim below was re-verified; the original (buggy) script and its output are kept, not
deleted, at `scripts/p5_adaptive_query_experiment.py` / `results/phase5/adaptive_query_experiment.json`.

## Control-monotonicity: proven and exhaustively verified, now with a committed reproducible artifact
`V(op,outage,c)` is provably non-increasing in `c` (componentwise): `c` enters `ScenarioLP` only through the
corrective-redispatch box bounds, which weakly EXPAND as `c` increases; box-constraint relaxation on a
fixed-structure LP cannot increase its minimum. **Correction (external review finding 10):** the original claim
of "415,699,200 pairs, zero violations, the full check" was run once interactively and never committed as a
reproducible artifact -- only a strided 84/4,200-row pytest sample was actually committed. Fixed:
`scripts/p5_monotone_full_check.py` now IS the full check, committed and re-runnable, output saved
(`results/phase5/monotone_full_check.json`): **4,200 rows x 98,976 ordered pairs = 415,699,200 total
comparisons, 0 violations, 2.3 seconds.** The claim is now independently reproducible from a script, not only
from session history.

## L(c)/U(c) bounds: implemented and verified valid
`src/fdna/adaptive_query.py`: `L(c) = max(F(S), max_{c_j>=c} V(c_j))`, `U(c) = min(1, min_{c_j<=c} V(c_j))`,
vectorised. Verified: `L(c) <= V(c) <= U(c)` for every control, given any queried subset, checked against the
cached true table -- queried points are recovered exactly. This machinery itself was never implicated by the
review's bug findings and stands as before.

## Acquisition experiment: corrected, now leakage-free and posterior-conditioned

**Two critical bugs, both confirmed by independently rerunning the review's own witness script against this
repo before any fix was written, then repaired in `scripts/p5_adaptive_query_experiment_v2.py`:**

1. **Hidden-state leakage (external review finding 4).** The original prediction expression indexed
   `Lg[true_c_idx]`/`Ug[true_c_idx]` -- the REALIZED hidden control index, information no real policy can
   access. Reproduced exactly: with identical observable inputs, changing only the hidden truth flipped the
   prediction, giving up to 100% "accuracy" on a problem where an observation-only classifier caps at 50%.
   Fixed: prediction now uses ONLY `(qL, qU)`, the posterior-mass bounds; `true_c_idx` is used exclusively,
   after the prediction is frozen, to score correctness.
2. **"Posterior MC" was prior sampling (external review finding 5).** The baseline called `v2.sample_states`
   unconditionally (drawing from the PRIOR) and generated an observation it never used to condition anything.
   Reproduced exactly: for an observation with all 13 flags reported down, prior P(no control)=0.169 vs true
   posterior P(no control)=0.99998 -- completely different distributions. Fixed: MC now draws control indices
   directly from the already-computed exact posterior `pc` (`np.random.choice(..., p=pc)`) -- true posterior
   Monte Carlo, no approximation.

Two further execution-integrity fixes: the shared observation was previously redrawn per BUDGET (budget was
part of the RNG seed), so different budgets scored a different workload, not just a different spending limit --
now frozen once per (operating point, cell) and reused identically across every budget. The script previously
used 20 ops / budgets {2,4,8,16,32}, disagreeing with the registered 60 ops / {2,4,6,8,12,16} -- now aligned to
the registry exactly, and MC's sample count now matches the budget (was capped at 8 regardless of budget).

Two required acceptance tests added (`tests/test_adaptive_query_integrity.py`): a leakage guard (prediction
provably independent of hidden truth given fixed observables) and a posterior-vs-prior distribution check
reproducing the review's exact 0.169-vs-0.99998 numbers.

**Corrected result, all 60 confirmatory operating points, all 70 triples** (`results/phase5/adaptive_query_experiment_v2.json`):

| budget | guided q/candidate | random q/candidate | guided acc | random acc | **mc acc (posterior, K=budget)** |
|---|---|---|---|---|---|
| 2 | 2.00 | 2.00 | 0.872 | 0.872 | **0.932** |
| 6 | 3.70 | 3.71 | 0.915 | 0.915 | **0.944** |
| 12 | 6.06 | 6.27 | 0.923 | 0.939 | **0.951** |
| 16 | 7.56 | 7.97 | 0.929 | 0.942 | **0.951** |

(P1 cell shown; P2 follows the same pattern, both in the JSON.) **With leakage removed, the picture is
completely different from the original, invalidated report: plain posterior Monte Carlo -- the simplest
possible policy, direct samples from the exact posterior -- beats BOTH bound-based policies at every budget
tested, in both cells.** Between the two bound-based policies, g2-guided acquisition uses modestly fewer
queries than random (about 5-10% fewer, consistent with the pre-correction finding) but random acquisition
reaches equal or slightly HIGHER accuracy at most budgets -- so guided acquisition still does not clear any
reasonable promotion bar over random, and now neither bound-based policy beats simple posterior MC either.

**Cost accounting, corrected (external review finding 6/10):** the query counts above are ONLY the N-3 target
queries charged against the budget. Building g2's own guidance additionally requires the FULL lower-order table
per operating point -- 41 singletons + 182 pairs, x1024 controls = **228,352 scalar values**, not the "223
lower-order queries" the original report implied (a 1024x undercount, since each cached "query" is actually a
1024-entry control grid, not one scalar). This is reported separately and explicitly now
(`results/phase5/adaptive_query_experiment_v2.json`'s `cost_accounting` field), not folded into the per-candidate
counts.

## Verdict, corrected
Monotonicity and the bound machinery are solid, exact, and now independently reproducible from committed
scripts. The acquisition experiment's ORIGINAL conclusion ("mostly null, doesn't clear the gate") happened to
survive in DIRECTION once the bugs were fixed, but the actual finding is sharper and different: **simple
posterior Monte Carlo beats both bound-based acquisition policies outright**, not merely "guided doesn't beat
random by 25%." Neither bound-based policy is promoted. This is a genuinely corrected, leakage-free result, not
a restatement of the original -- the original's specific numbers (guided 0.951/0.936/0.957 at budgets 8/16/32,
MC "capped at K=8" showing erratic 0.81-0.89 accuracy) were artifacts of the leakage and prior-sampling bugs and
are withdrawn.

**What remains untested:** a genuine shortlist-boundary-aware acquisition rule (the brief's fuller
specification, not the simpler "closest to tau" heuristic implemented here); the residual/control-variate
fallback the brief also requested (`E[V-g2|obs]` estimated from valid posterior samples) was not attempted.

**Additional gaps confirmed by an independent second-round review (2026-09-25, repair in progress):** the
registry's own `metrics` field requires shortlist recall/precision at 10/20/40% budgets -- not computed anywhere
in this experiment, only classification accuracy and query counts. `resolve()`'s stopping rule waits until
every posterior-support control state is individually resolved, which is strictly stronger (and more expensive)
than what the binary decision needs -- `qL>0.5` or `qU<=tau` alone already certifies the prediction, so the
policy keeps querying past the point the decision was already locked in. Each candidate gets its own
independent budget cap with no shared/global allocation across the shortlist. `mc_queries_total` charges every
Monte Carlo draw as a fresh oracle solve, with no credit for a cache on repeated control-state draws (at budget
16, `4200*16=67200` "queries," most of them plausibly cache hits). The "posterior MC beats both bound-based
policies at every budget" finding above has no saved uncertainty bound -- a real point-estimate finding, not
yet a significance claim. All of this is being addressed; see `results/phase4/CLAIM_LEDGER.md` for status.
