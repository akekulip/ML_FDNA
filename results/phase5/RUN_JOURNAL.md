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
