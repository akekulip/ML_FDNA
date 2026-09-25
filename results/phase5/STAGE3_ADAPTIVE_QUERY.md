# Phase 5 Stage 3 — bounded adaptive-query screening

**This document was corrected across two independent review rounds.** Round 1 found two critical bugs
(hidden-state leakage in the prediction logic, and a "posterior MC" baseline that actually sampled the prior)
plus several execution mismatches. Round 2 found the experiment still didn't implement several things its own
registry declared (shortlist metrics, decision-aware stopping, cache accounting, an uncertainty bound on its
headline comparison) plus a cheap diagnostic worth measuring. Every claim below was re-verified both times; the
original (buggy) script and its output are kept, not deleted, at `scripts/p5_adaptive_query_experiment.py` /
`results/phase5/adaptive_query_experiment.json`.

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

**Corrected result, all 60 confirmatory operating points, all 70 triples** (`results/phase5/adaptive_query_experiment_v2.json`; P1 cell shown, P2 follows the same pattern):

| budget | guided q/candidate | random q/candidate | guided acc | random acc | **mc acc (posterior, K=budget)** |
|---|---|---|---|---|---|
| 2 | 2.00 | 2.00 | 0.872 | 0.872 | **0.932** |
| 6 | 3.49 | 3.50 | 0.915 | 0.915 | **0.944** |
| 12 | 5.48 | 5.71 | 0.923 | 0.938 | **0.951** |
| 16 | 6.72 | 7.17 | 0.929 | 0.943 | **0.951** |

(Query counts at budget 12/16 are lower than an earlier pass of this same table -- e.g. 6.72 vs a previously
reported 7.56 at budget 16 -- because of the decision-aware stopping fix added in repair round 2, see below; the
accuracy numbers are essentially unchanged, confirming the earlier stopping rule was wasting queries without
changing the eventual decision.) **With leakage removed, the picture is completely different from the original,
invalidated report: plain posterior Monte Carlo -- the simplest possible policy, direct samples from the exact
posterior -- beats guided acquisition's point estimate at every budget tested, in both cells, and (repair round
2) this is now backed by a paired-per-op bootstrap CI that excludes zero at every budget in both cells (p<0.05
throughout, mostly p<0.01) -- a real, significant effect, not just a point estimate.** MC vs. RANDOM acquisition
is a more nuanced picture: MC's point estimate is higher at every budget in both cells, and the difference is
statistically significant at low-to-mid budgets, but loses significance at the highest budgets tested (P1
budget 16: p=0.086; P2 budgets 12/16: p=0.056/0.150) -- so "MC beats random" is a solid finding at low budgets
and a real but not yet statistically confirmed one at the highest budgets. Between the two bound-based policies,
g2-guided acquisition uses fewer queries than random at every budget (now more so, thanks to decision-aware
stopping) but random acquisition reaches equal or slightly HIGHER accuracy at most budgets -- guided acquisition
still does not clear any reasonable promotion bar over random.

**Cost accounting, corrected (external review finding 6/10):** the query counts above are ONLY the N-3 target
queries charged against the budget. Building g2's own guidance additionally requires the FULL lower-order table
per operating point -- 41 singletons + 182 pairs, x1024 controls = **228,352 scalar values**, not the "223
lower-order queries" the original report implied (a 1024x undercount, since each cached "query" is actually a
1024-entry control grid, not one scalar). This is reported separately and explicitly now
(`results/phase5/adaptive_query_experiment_v2.json`'s `cost_accounting` field), not folded into the per-candidate
counts.

## Verdict, corrected (repair round 2: now with real significance, not just point estimates)
Monotonicity and the bound machinery are solid, exact, and now independently reproducible from committed
scripts. The acquisition experiment's ORIGINAL conclusion ("mostly null, doesn't clear the gate") happened to
survive in DIRECTION once the bugs were fixed, but the actual finding is sharper and different, and now
statistically backed rather than a bare point estimate: **plain posterior Monte Carlo significantly beats
g2-guided acquisition at every budget tested, in both cells (paired-per-op bootstrap, p<0.05 throughout, mostly
p<0.01)**; it also beats random acquisition's point estimate at every budget, but that specific comparison is
only statistically significant at low-to-mid budgets, not at the highest ones tested. Neither bound-based policy
is promoted. This is a genuinely corrected, leakage-free result, not a restatement of the original -- the
original's specific numbers (guided 0.951/0.936/0.957 at budgets 8/16/32, MC "capped at K=8" showing erratic
0.81-0.89 accuracy) were artifacts of the leakage and prior-sampling bugs and are withdrawn. Repair round 2 also
added decision-aware stopping (genuinely saves queries at the same accuracy), unique-query/cache accounting,
the registry's declared shortlist recall/precision metrics, and a favorable-looking (but not yet implemented)
control-variate diagnostic -- see the section below for all five.

**What remained untested at this point:** a genuine shortlist-boundary-aware acquisition rule (the brief's
fuller specification, not the simpler "closest to tau" heuristic implemented here); the residual/control-variate
fallback the brief also requested was DIAGNOSED but not implemented as a policy -- see repair round 2 below.

## Repair round 2: the registered-contract gaps a second independent review found, now addressed
The review found four things the registry itself declared but the experiment never implemented, plus one
requested cheap diagnostic. All five addressed in `scripts/p5_adaptive_query_experiment_v2.py`:

1. **Decision-aware stopping.** `resolve()` used to wait until every posterior-support control state was
   individually resolved (`L(c)==U(c)`), strictly stronger than the binary decision needs. Fixed: an early exit
   checks `qL>0.5` or `qU<=tau` after every query and stops the moment the decision is already certified. Effect
   confirmed in the corrected table above: fewer queries used at the same budget cap (6.72 vs 7.56 at budget
   16), same accuracy -- the old rule was provably wasting queries without changing outcomes. **Correction (repair
   round 3): this fix itself had a threshold bug** (compared the probability bound `qU` against `spec.SEVERE`,
   the load-shed threshold, instead of the probability-decision threshold 0.5) -- overly conservative, never an
   invalid certificate. Fixed and moved into `fdna.adaptive_query.resolve` with a named
   `PROB_DECISION_THRESHOLD` constant; queries dropped further, to **4.72** at budget 16 (P1) -- accuracy again
   unchanged, confirming the query counts, not the decisions, were what the bug affected.
2. **Unique-query / cache accounting.** MC's `mc_queries_total` (draws charged) is now reported alongside
   `mc_unique_queries_total` (distinct control indices among those draws, which a real cache could serve
   without a fresh solve). At budget 16, P1: 67,200 draws charged, only 40,187 unique (60%) -- a real cache
   would cut MC's effective cost by roughly 40% at this budget. At budget 2, the ratio is much closer to 1
   (7,699/8,400, 92%) -- caching matters more as the budget (and therefore the redundancy) grows.
3. **Shortlist recall/precision at 10/20/40% budgets** (the registry's own declared metric, absent before):
   implemented via `fdna.evalutil.recall_at`/`precision_at`, ranking pooled candidates by each policy's own
   continuous severity score. For the guided policy at budget 16 (P1): recall@10/20/40% = 0.240/0.481/0.906,
   precision@10/20% = 1.00 (every candidate in the top 10-20% by guided score really is severe), precision@40% =
   0.943. One reasonable operationalization of the registry's underspecified field, not claimed to be the only
   valid one -- full numbers for every policy/budget/cell in the JSON's `shortlist_metrics`.
4. **A paired-per-op bootstrap CI**, added as described above -- MC vs guided is significant at every budget in
   both cells; MC vs random is significant at low-to-mid budgets only.
5. **A cheap, cached-data-only diagnostic for the review's proposed control-variate estimator**
   (`p_hat = E_p[h0] + mean(h(c_j)-h0(c_j))`, `h0(c)=1[g2(c)>tau]`, `h(c)=1[V(c)>tau]`): computed exactly under
   the posterior (`Var_p[h-h0]` vs `Var_p[h]`, no simulation needed). **Correction (2026-09-25, independent
   third-round review): the "97% favorable" figure below was of an already-filtered subset, not of all 4,200
   candidates, and the original phrasing didn't say so.** The diagnostic silently drops any candidate where
   plain-MC variance is already ~zero (`var_h <= 1e-12`) before computing the ratio -- confirmed:
   `n_candidates_measured` is 2,292/4,200 (P1, 54.6%) and 2,053/4,200 (P2, 48.9%). So "97% favorable" describes
   roughly half the pool; the other half was never assessed, including cases where a poor surrogate could
   introduce variance where none existed (worked counterexample from the review, confirmed algebraically:
   posterior weights [0.5,0.5], true indicators [0,0], proxy indicators [0,1] -- plain-MC variance is exactly
   zero while the corrected estimator's variance is 0.25, a real excluded-harm case). Read honestly (revised):
   among the ~half of candidates where plain MC already has meaningful variance, g2 as a control variate looks
   favorable for the great majority of them (median ratio 0.0, 97% of that subset <1) -- promising, but the
   excluded half needs its own accounting before claiming a population-wide result. Repair round 3 replaces this
   diagnostic with a full panel (every candidate categorized, none silently dropped) and implements the
   estimator as an actual candidate policy rather than only measuring it further.

**Additional caveats confirmed by the third-round review, not previously stated:**
- **MC's accuracy advantage is not blanket shortlist-screening superiority -- and pooling versus per-op framing
  changes the picture materially.** Pooled across all 4,200 candidates, guided's precision@10%/20% is 1.00/1.00
  versus MC's 0.962/0.967 at P1 budget 2, despite MC's higher overall classification accuracy. Recomputed the
  operational way (per-operating-point, then averaged -- see below), the direction reverses at the fine-grained
  end: at recall@10%/20%, GUIDED is modestly but statistically significantly BETTER than MC at several
  budget/cell combinations (e.g. P1 budget 2: mc-guided mean -0.013/-0.012, 90% CI entirely negative both
  percentiles; P2 shows the same sign (guided ahead) at nearly every budget, small magnitude, but only about
  40% of the recall/precision-by-budget combinations checked for P2 reach 90% significance -- **correction
  (independent qa-verifier pass, repair round 3): an earlier version of this sentence said "at most of them,"
  which overstated the fraction; recomputed directly from `shortlist_boot_ci_per_op`, it is roughly 10/24
  (~42%), not most.** MC's clear, consistent advantage shows up specifically at the coarser recall@40% threshold
  (mc-guided
  CI excludes zero, positive, at every single budget in both cells) and at raw classification accuracy. Read
  honestly: this is NOT "MC dominates shortlist screening" -- it is "different policies are better at different
  points on the recall curve," a genuinely more interesting and more cautionary finding than either the original
  pooled comparison or a first-pass per-op summary suggested.
- **The pooled shortlist metrics answer a different question from per-operating-point screening.** A
  constructed two-group example reproduces the artifact exactly: pooled recall@20% can be 0.0 while the mean
  PER-OPERATING-POINT recall@20% on the identical scores is 0.5. Repair round 3 adds genuine per-operating-point
  shortlist metrics (`shortlist_metrics_per_op` in the JSON, with a paired-per-op bootstrap CI on every
  policy-vs-policy comparison); the pooled numbers are kept (`shortlist_metrics_pooled`) but should be read as
  an "offline pooled-selection task" answer, not the operational snapshot-screening answer.
- **"MC beats guided" at matched nominal budget doesn't mean matched real cost.** At P1 budget 16 (post-threshold-
  fix), guided uses about 4.72 target queries/candidate while MC's DISTINCT (unique) target queries per candidate
  is about 9.57 -- MC's accuracy advantage is real but bought with roughly double the actual oracle work the
  nominal budget cap suggests, once repeated draws are accounted for honestly (`mc_unique_queries_total`).

## Repair round 3: the control-variate estimator implemented as an actual policy, exploratory
Per the review's own recommendation ("implement and test the actual control-variate estimator... measure
whether it improves shortlist quality at matched total cost"), not just diagnose it further. New script
`scripts/p5_adaptive_query_experiment_v3.py` (v2 kept unmodified on disk). Scope, stated explicitly: EXPLORATORY
only, reusing the already-open 600-659 block and the existing 70-triple manifest -- no new reserve, no fresh LP
solves; confirmatory work on genuinely unused operating points is future work.

**The full variance panel (correcting the diagnostic itself first).** Every one of the 4,200 candidates now gets
a category, none silently dropped: P1 -- 1,881 zero-variance ties, 27 zero-variance-but-harmed-by-proxy, 2,292
nondegenerate; P2 -- 2,111/36/2,053. The stabler pooled statistic across ALL candidates,
`ratio_of_summed_variances_all_candidates`, is **0.106 (P1) / 0.098 (P2)** -- genuinely favorable (well below 1)
even accounting for every zero-variance case the old diagnostic silently excluded, not just the nondegenerate
subset. This is a materially stronger and more complete confirmation than the withdrawn "97% of candidates"
figure.

**The estimator, for fixed beta in {0, 0.5, 1.0} (0=plain MC, 1=the full correction), reusing the IDENTICAL MC
draws at zero additional oracle cost** (P1 cell, accuracy):

| budget | guided | mc (=cv, beta=0) | cv, beta=0.5 | cv, beta=1.0 | g2_zero_query | exact_posterior_ceiling |
|---|---|---|---|---|---|---|
| 2 | 0.872 | 0.932 | 0.943 | **0.953** | 0.947 | 0.956 |
| 8 | 0.917 | 0.946 | 0.952 | **0.955** | 0.947 | 0.956 |
| 16 | 0.929 | 0.951 | 0.955 | **0.956** | 0.947 | 0.956 |

(P2 follows the same pattern, full numbers in the JSON.) **At beta=1.0, the estimator closes almost the entire
remaining gap to the theoretical exact-posterior ceiling using only the SAME K=budget oracle queries plain MC
already spends** -- at budget 16, cv(beta=1.0)=0.9564 versus the ceiling's 0.9562 (a labeled oracle quantity,
never a deployable policy). The improvement over plain MC (`cv_b1.0_minus_mc`) is statistically significant
(p<0.01) at 11 of the 12 budget/cell combinations tested; the one exception is P1 at budget 12 (p=0.068,
still the right sign, mean +0.0045) -- a real, overwhelmingly significant, practically meaningful effect, not a
diagnostic percentage. The shortlist
metrics move the same direction: cv(beta=1.0)'s per-op recall@40% (0.869 at P1 budget 16) beats both guided
(0.847) and plain mc (0.868). One caveat worth stating plainly: `exact_posterior_ceiling`'s decision rule
(`E_p[h_true]>0.5`) is Bayes-optimal in EXPECTATION over the posterior, not a strict per-candidate accuracy
ceiling against the one frozen realized true state each candidate uses here -- so a lower-variance finite-sample
estimator can occasionally edge past it on this particular frozen sample without that meaning it beat the truth.

**Verdict: the control variate looks like a genuine, low-cost improvement over plain posterior MC, worth a
confirmatory run.** Still not attempted (explicitly future work): validation-selected/adaptive beta (needs a
separated pilot-sample protocol per the review's own caution); a genuine shortlist-boundary-aware acquisition
rule; a global budget allocation shared across the whole shortlist rather than an independent per-candidate cap;
and a confirmatory run on genuinely unused operating points/outages.
