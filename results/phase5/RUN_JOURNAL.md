# Phase 5 (overnight PI execution brief) — run journal

Start: 2026-09-25T02:44:09Z (UTC) / 2026-09-24 22:44:09 EDT. Deadline (8h ceiling): 2026-09-25T10:44:09Z / 06:44 EDT.
HEAD at start: 7b7cb3ea33ff207177d2bb5679e77477960ccff8 (matches the brief's audited checkpoint). Branch: main. Dirty: only the new brief file (untracked).
Hardware: RTX 2070 8GB (274 MiB used, 0% util — idle), 32 cores, 31Gi RAM (26Gi available). No ML_FDNA jobs running (only unrelated MCP servers for other projects).

## Stage 0 — first-hour corrections (6 items, all verified before this run started)
1. Triple overlap (0,23,34) / (9,13,40) confirmed present in both manifest_k3_confirm.json and manifest_amortization_new200.json — verified exactly, zero overlap with the original 60-triple screen.
2. Cost breakdown verified exact: N-1=2,519,040; N-2=11,182,080; N-3=4,300,800; N-1+N-2=13,701,120; total 18,001,920.
3. Decision-rule gap confirmed: p4_mobius_confirm.py never ran an executable intersection_union check.
4. Amortization coverage (2,698/8,436) already correct from earlier session, not regressed.
Work log below.

## Stage 0 — completed (all 6 items)
1. 68-triple sensitivity (excludes (0,23,34), (9,13,40), manifest-membership only): P1 +0.254 (was +0.250), P2 +0.273 (was +0.270) -- nearly identical, overlap triples had negligible effect on the conclusion.
2. Executable intersection_union decision (src/fdna/rules.intersection_union, SESOI=0.05 margin on all 4 components): tier1_confirmed_strict_lo90_ge_margin = TRUE, all_positive = TRUE. Formal, code-based confirmation, not eyeballed prose.
3. No-control linear finding restated with paired bootstrap over the 70 triples: g2 MSE 7.691e-5 vs L1 7.331e-5 (4.68% gap), mean diff 3.60e-6, 90% CI [1.62e-6, 5.65e-6] -- a small but real, statistically confirmed effect (lo90>0), not a bare point estimate. Note: this is the OPPOSITE direction from OLS in the earlier exploratory screen (OLS beat g2 by 17.9% there); in this confirmatory run g2 beats OLS outright and only L1 keeps a small edge.
4. Per-row predictions with canonical keys saved: data_hik/confirm_predictions.npz (87MB, op_id/outage/cell/draw/g1/g2/prior_only/y_true/key), enabling independent recomputation without retraining or resolving.
5. Cost attribution already recorded exactly (verified this session): N-1=2,519,040, N-2=11,182,080, N-3=4,300,800, total 18,001,920.
6. Audited: op_ids 660-699 confirmed untouched by any manifest in the project (grep across all data_hik/*.json). Genuinely fresh, available for any future look.

Scripts: scripts/p4_mobius_confirm_v2.py (kept p4_mobius_confirm.py, not deleted). Output: results/phase5/mobius_confirm_corrected.json, results/phase5/linear_nocontrol_paired.json.
No new LP solves needed -- reused cached n1/n2/n3_confirm.npz label tables from the original confirmatory run.

## Stage 2 — physical (island-floor) correction: DONE
V(empty)=0 proven from the generator's rating calibration + dispatch-balance acceptance criterion, verified
exactly across 150 (op,control) checks. Three composition variants implemented, correctness-gated (5/5 tests).
Evaluated on cached confirmatory data: g2_residual collapses to g2_plain (zero genuinely-new 3-cuts in the
70-triple sample, verified exhaustively); g2_clipped gives a small MAE improvement in one class but misses the
predeclared missed-severe gate. Hand-checkable N-3 counterexample (9,26,33) added as a permanent test, revealing
a second emergent mechanism (congestion/trip-rule shed, zero structural floor) neither variant addresses.
See results/phase5/STAGE2_PHYSICAL_CORRECTION.md.

## Stage 3 — bounded adaptive-query screening (the brief's "main candidate"): DONE
Control-monotonicity proven and verified exhaustively (415.7M pairs, zero violations; caught+fixed a sign bug
in the first check). L/U bound machinery implemented, vectorised, verified valid. Acquisition experiment: honest
mostly-null result -- g2-guided next-query selection saves only ~5-10% of queries vs random at matched-or-worse
accuracy, missing the predeclared >=25% gate (caught+fixed a stopping-rule bug in the first attempt where
neither policy ever stopped early). Bound machinery is solid reusable infrastructure; this acquisition heuristic
is not promoted. See results/phase5/STAGE3_ADAPTIVE_QUERY.md.

## Stopping point (context/time budget)
Stages 1 (full query-access-wrapper contract), 4 (matched recurrent experiment), 5 (creative idea cards / lit
review), 6 (one extension), 7 (independent-reviewer pass) NOT reached this run -- recorded honestly, not
silently dropped. See the final status report delivered to Philip for the explicit recommendation on what to
run next. 95 commits total this session, none pushed without explicit request. Tests: 77 pass.

## Stage 4 — matched recurrent/learned-correction experiment: DONE
Identifiability check stated first and respected by design (residual arm trained only on N-3, never on the
trivially-zero N-1/N-2 residual). 45/25 triple split, verified disjoint. All six arms (g2_fixed, g2_clipped,
ridge, gbm, deepsets [commutative pair-aware set model], residual-correction net) given identical 15-dim
features, composed with the SAME P1/P2 posterior, scored on R-precision, 25 held-out test triples.
RESULT: g2_fixed wins decisively in both cells; every learned arm's 90% CI vs g2_fixed lies below zero;
deepsets (closest to a "recurrent structured correction") is worst of all. No learned arm promoted. See
results/phase5/STAGE4_MATCHED_RECURRENT.md.

## Stopping point (this stretch)
Stages 0, 2, 3, 4 (the four core scientific stages) complete with decisive or honestly-null results. Stage 1's
full query-access-wrapper CLASS was not built as separate infrastructure (canonical keys and cost accounting
were applied ad hoc within Stages 3/4 instead). Stage 5 (fresh 6-card idea/lit pass) not run this stretch --
Phase 4's existing results/phase4/IDEA_CARDS.md (6 cards, 3 fields, verified sources) is the closest prior
artifact and was not superseded. Stage 6 (one extension) and Stage 7 (independent-reviewer pass) not reached.
99 commits total this session, none pushed without explicit request. Tests: 77 pass throughout.

## Repair round (2026-09-25): external evidence review of commit d962880, Stages R0-R6
Philip supplied `ML_FDNA_d962880_Evidence_Review.md`, an external audit of this run's commit d962880, with a
runnable witness script and 10 numbered findings. **Every finding was independently reproduced against the live
repo before any fix was written** (all 10 reproduced exactly, using the review's own witness code against this
repo's actual committed functions), then a repair plan (Stages R0-R6) was approved and executed in full.

**R0 -- registry YAML.** `phase5_adaptive_query.yaml`'s flow-mapping brace error fixed (the third occurrence of
this exact mistake this session); `phase3_step1.yaml`/`phase3_step2.yaml` also found and fixed during a full
sweep. `phase4_step_partialobs.yaml` and three older files (`registry.yaml`, `addendum_H4.yaml`,
`amendment_H5.yaml`) remain invalid YAML but are pre-existing, documented-superseded/unloaded-by-any-code
prose files -- confirmed genuinely out of scope by the R4 verification pass (grep-confirmed no script ever
`yaml.safe_load()`s them).

**R1 -- the six code bugs, each independently reproduced then fixed with its own commit:**
1. Physical-correction missed-severe sign error (`p5_physical_correction_eval.py`): predictions were
   reconstructed as `truth-err` (=2*truth-pred) instead of used directly; silently hid real missed-severe cases.
   Corrected counts: full_control 47/47/47 (was 1/1/1), no_control 38/38/38 (was 12/12/12).
2. `is_new_cut` used as the (wrong) necessary condition for `g2_residual==g2_plain` in STAGE2's reasoning;
   corrected to the actual test, `F(S)-Mobius_order_2[F](S)`, verified exactly zero on all 4200 real rows.
3. Adaptive-query hidden-state leakage: prediction indexed the REALIZED hidden control index directly, giving
   up to 100% "accuracy" on a problem an observation-only policy caps at 50%. Fixed: prediction now uses only
   the posterior-mass bounds `(qL,qU)`.
4. "Posterior MC" baseline actually sampled the PRIOR (0.169 vs true posterior 0.99998 for an informative
   observation). Fixed: draws directly from the exact posterior `pc`.
5. Matched-recurrent validation targets were misaligned (4,796/4,800 mismatched control indices) and validation
   overlapped training (612/4,800 shared rows). Fixed: `(key,X,y,g2)` built together in one pass with
   `r+g2==y` asserted; TRAIN and VAL are genuinely disjoint on both the outage and operating-point axes
   (corrected 2026-09-25: an earlier version of this line said "three genuinely disjoint train/val/test groups
   on both axes," which is false -- TEST deliberately reuses all 60 operating points against 25 held-out
   outages, the known-operating-point/new-outage-combination axis, as `STAGE4_MATCHED_RECURRENT.md` itself
   always correctly stated; only this summary line overstated it, caught by an independent second-round review).
6. The "commutative" DeepSets model was not actually permutation-invariant (0.10 vs 0.11 under relabeling with
   a valid weight assignment). Fixed with a symmetric per-edge encoder (`src/fdna/nn/pairset.py`), verified
   invariant on the actual trained model (later found still incomplete -- see the repair-round-2 entry below).
   Plus: an uncontrolled bootstrap unit (`p4_mobius_confirm_v2.py` passed 4,200 raw rows to `boot()`, mislabeled
   as "70 triples"); corrected CI now includes zero (p=0.36, was p=0.001 mislabeled).

**R2/R3 -- acceptance tests woven into each R1 fix, and all four affected cached-data evaluations rerun** (no
new LP solves anywhere): see `results/phase5/{STAGE2_PHYSICAL_CORRECTION,STAGE3_ADAPTIVE_QUERY,
STAGE4_MATCHED_RECURRENT}.md` and `results/phase4/CLAIM_LEDGER.md` rows 10-12 for the corrected numbers and an
honest account of what changed. Headlines: adaptive query -- plain posterior Monte Carlo now beats BOTH
bound-based acquisition policies at every budget in both cells, reversing the original (leaking) report's
ranking; matched-recurrent -- g2_fixed still wins both cells (headline unchanged), but the properly-invariant
DeepSets model's gap shrank from -0.113/-0.122 to -0.011/-0.007, roughly on par with GBM rather than "worst of
five by a wide margin" (largely an artifact of the invariance bug); linear no-control -- the 4.68% MSE point
estimate stands but its claimed statistical significance is withdrawn (properly clustered CI includes zero).

**R4 -- independent verification, run by a qa-verifier subagent that authored none of the R1-R3 fixes**, exactly
as both the original brief and the evidence review required before any corrected conclusion could be treated as
final. Verdict: **PASS on all 10 findings**, each checked against fresh, independent executions of the real
code (not comment/docstring inspection) -- a full pytest run, fresh reruns of every affected script, an
independent from-scratch rebuild of the permutation-invariance and third-order-Mobius checks, and several
reruns reproducing committed artifacts bit-for-bit. It flagged two non-blocking test-quality nits (a vacuous
leakage-guard test that re-derived its formula inline instead of calling the real prediction logic, and
mislabeled permutation index tuples in the DeepSets invariance test) -- both fixed immediately: the prediction
rule was factored into `fdna.adaptive_query.predict_from_bounds` (a pure function of `(qL,qU)` with a
structurally-asserted signature that cannot carry a hidden-state parameter) shared by the script and the test,
and the permutation tuples were corrected by direct vertex-consistent enumeration. No regressions from either
fix (script output byte-identical after the refactor; full suite went from 84 to 85 passed).

**R5 -- the reviewer's proposed `capacity_floor` mechanism** implemented (`src/fdna/physical_correction.py`) and
gated (`tests/test_capacity_floor.py`, reproduces the review's own toy LP exactly: 0%/20%/20% agreement).
Evaluated on the real 70-triple sample: a genuine, mathematically-guaranteed tighter lower bound than
`island_floor`, active on 2.9% of no-control rows, giving a small MAE improvement but no change to the
missed-severe count on this specific sample -- real but not decisive here (`CLAIM_LEDGER.md` row 13).

**Verdict on the whole repair round.** Every one of the review's 10 findings had a real bug, a real fix, a
regression test, and independent confirmation from an agent that did not write the fix. Two of the corrected
findings (adaptive query, matched-recurrent) materially changed the practical conclusion, not just the
supporting numbers; two others (physical-correction missed-severe, linear-bootstrap significance) changed the
evidentiary status of an existing claim without changing its qualitative direction. Nothing in this round
overturns g2's own confirmatory result (row 5c) or the earlier Phase 4 findings not implicated by any of the 10
findings. Tests: 85 pass. Commits: R0 through this stage, each its own commit, none pushed without explicit
request.
