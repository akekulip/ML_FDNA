# WORKING_NOTES — ML_FDNA
Repo: ~/Projects/ML_FDNA (github.com/akekulip/ML_FDNA private; **pushed head 943e09a** (refreshed 2026-09-24, was stale at 31aff42); local-only commits since then, no push without asking).
Read first: results/SUMMARY.md, results/PHASE2.md, results/LEDGER.md, results/phase3/{STEP1,STEP1_EXT}.md, results/phase4/STATUS_AUDIT.md, registry/*.yaml (locked hypotheses, addenda, amendment, slots, locks).

## State (2026-09-24, Phase 4 start)
Phase 1/2 (overnight programme) and Phase 3 (diagnostics: LODF compensation-determinant mechanism, Jensen-gap mechanism, FDNA repair module, A9 equal-information
audit) complete -- see results/phase3/STEP1.md and STEP1_EXT.md. No registered FDNA claim survived; one small tier-2 non-FDNA decomposition effect (+0.03, not tier 1,
two blocks share training draws so not an independent replication). Phase 4 (external PI execution brief, `ML_FDNA_Claude_Code_PI_Prompt.md`) now in progress:
Stage 0/1 (state audit + 4 corrections) done. Stage 2 (literature verification + 2 independent blind brainstorming agents) done -- results/phase4/IDEA_CARDS.md.
Stage 3 headline finding: a zero-training k=2-additive Mobius/interaction truncation (built from N-1/N-2 values only) beats a naive singleton-sum baseline AND
an equal-information GBM at predicting EXACT N-3 labels, at 4/4 tested control states (results/phase4/RESULTS.md, results/phase4/mobius_k3_robust.json).
Extended to N-4: order-2 truncation still beats naive baseline (2.8x); adding order-3 information improves further (3.1x). Staged comparator set (tuned tree,
DeepSets, GRU), trained on 20 ops only, all underperform even the naive baseline -- a data-scale limitation, not evidence against recurrence in principle.
Repaired-FDNA inference-only screen COMPLETE both cells: combined repair (OR-aware+hard-min+joint+CC-prior) ties the generic MLP and is worse than plain
dependency logic in both P1 and P2 -- a stable, twice-replicated negative finding (registry/phase3_step2.yaml's attribution rule fails).
Partial-observation validation of the Mobius result went through three rounds of external review (each independently verified before acting): reproducibility
bugs fixed twice (hash() instability, pooled-vs-per-op R-precision), a p-value floor caught, an amortisation arithmetic error caught, a linear-baseline
overfitting claim withdrawn and replaced with a metric/regime-dependent finding. See results/phase4/RESULTS.md for the full corrections trail.
**CONFIRMED (tier1, both cells): reserve block 600-659 opened (registry/phase4_mobius_confirm.yaml, MOBIUS_CONF_P1/P2 slots, lock output sha256 80308ac7...),
fresh 70-triple manifest, 18,001,920 fresh LP solves. R-precision improvement g2-g1: +0.250 (P1, 90% CI [0.237,0.264]) / +0.270 (P2, 90% CI [0.255,0.285]),
g2 beats both baselines at 60/60 ops in both cells. First confirmatory (not exploratory-screen) positive result in the whole Phase 1-4 programme.** The
exploratory "learning helps under no control" nuance did NOT replicate at this larger sample -- reported as a genuine correction, not hidden.
See results/phase4/{RESULTS,CLAIM_LEDGER,IDEA_CARDS,STATUS_AUDIT,REPRODUCE}.md for full detail.
Confirmatory looks so far: four block openings for the H5 claim (confirm 400-479, replicate 500-579); two new openings for this Phase 4 confirmatory result
(reserve 600-659, MOBIUS_CONF_P1/P2). Reserve 660-699 and all other slots remain unopened. Tests: 70 pass.

## Data (git-ignored; regenerate with scripts/value_table.py, gen_dataset.py)
data/ (v1 labels + ops), data_fresh/, data_v2/ (value tables, screens, results parquet), data_v2_confirm/ (label tables for blocks 400-479, 500-579), data_v2_fs/.

## Phase 5 repair round (2026-09-25): external evidence review of commit d962880, all 10 findings fixed
Philip supplied `ML_FDNA_d962880_Evidence_Review.md`, a rigorous external audit of Phase 5 (commit d962880) with
a runnable witness script. All 10 findings were independently reproduced against the live repo before any fix
was written, then repaired one at a time (Stage R0-R5 of the approved repair plan, `git log` shows each as its
own commit): registry YAML syntax (R0); residual validation misalignment + train/val overlap + non-invariant
DeepSets + unscaled feature in the matched-recurrent experiment (R1, 4 bugs, `p5_matched_recurrent_experiment_v2.py`);
adaptive-query hidden-state leakage + prior-not-posterior MC baseline (R1, `p5_adaptive_query_experiment_v2.py`);
physical-correction missed-severe sign error (R1, `p5_physical_correction_eval.py`); `is_new_cut` reasoning error
in STAGE2 (R1, corrected to the actual `F(S)-Mobius_order_2[F](S)` test); linear-bootstrap clustering (R1,
`p4_mobius_confirm_v2.py`). Acceptance tests for each bug added inline (R2). All four affected cached-data
evaluations rerun with the fixes (R3) -- see results/phase5/{STAGE2_PHYSICAL_CORRECTION,STAGE3_ADAPTIVE_QUERY,
STAGE4_MATCHED_RECURRENT}.md and results/phase4/CLAIM_LEDGER.md rows 10-12 for the corrected numbers and what
changed (adaptive query: plain posterior MC now beats both bound-based policies, reversing the original ranking;
matched-recurrent: DeepSets' gap to g2_fixed shrank from -0.11/-0.12 to -0.01, largely an artifact of the
invariance bug, headline "no arm promoted" unchanged; linear no-control: point estimate stands, significance
withdrawn). Stage R5 done: the reviewer's proposed `capacity_floor` mechanism implemented and gated
(`src/fdna/physical_correction.py`, `tests/test_capacity_floor.py`), evaluated on the real 70-triple sample --
real, gated, mathematically-tighter bound; small no-control MAE improvement, no missed-severe change on this
sample (CLAIM_LEDGER row 13). Old/buggy scripts and outputs kept, not deleted, throughout.
**Stage R4 (independent verification of R1-R3, mandatory per the review's own closing instruction) is DONE and
PASSED.** A qa-verifier subagent that authored none of the fixes independently checked all 10 findings against
fresh, from-scratch executions of the real code (not comment/docstring trust) -- full pytest rerun, fresh reruns
of every affected script, an independent rebuild of the permutation-invariance and third-order-Mobius checks,
several reproducing committed artifacts bit-for-bit. Verdict: PASS on all 10, no regressions. It flagged two
non-blocking test-quality nits (a vacuous leakage-guard test, mislabeled DeepSets permutation tuples); both
fixed immediately and verified (full suite 84->85 passed, script output byte-identical after the refactor).
Stage R5 (capacity_floor) and R6 (this journal update) are also done -- see results/phase5/RUN_JOURNAL.md for
the full consolidated account. The whole repair round (R0-R6) is complete.

## Phase 5 repair round 2 (2026-09-25): independent SECOND review of the repair round -- COMPLETE, pushed
The Stage R0-R6 repair round above (commit 842e6a9) was pushed, then independently reviewed a SECOND time
(`ML_FDNA_842e6a9_Repair_Review.md`, committed in the repo root). Verdict: the repairs were real, but one fix
(DeepSets permutation invariance) was still incomplete, plus several completeness/test-quality/documentation
gaps. Every claim independently reproduced against the live repo before any fix was written (same discipline as
round 1). Fixed in two commits (4a2c630 doc-only; 87b4943 code + retrain):
- **T1 (doc fixes):** RUN_JOURNAL/CLAIM_LEDGER's "three disjoint train/val/test groups on both axes" corrected
  (TEST intentionally reuses all 60 ops); capacity-floor's "~1.5%" MAE improvement corrected to 0.551% (the
  actual capacity-vs-island comparison; 1.543% was capacity-vs-plain, a different comparison); missed-severe
  "38/38/38 both control states" corrected to 47/47/47 full-control, 38/38/38 no-control.
- **T2:** two vacuous regression guards (a literal self-comparison, and a tautological check given its own
  definition) in the matched-recurrent script replaced with genuine independent reconstructions from canonical
  (no-split-prefix) keys.
- **T3 (the substantive one):** DeepSets' readout was STILL concatenating raw, unpermuted risk features after
  pooling -- the dedicated test used equal risk values and couldn't detect it. Fixed (risk folded into its own
  edge token; one shared scaler across all 3 risk slots, new `src/fdna/nn/pairset_features.py`), retrained,
  verified genuinely invariant end-to-end on the actual checkpoint. **Result: g2_fixed's lead over DeepSets
  WIDENED (round 1's -0.011/-0.007 -> -0.0386/-0.0380) -- round 1's improvement was itself partly an artifact of
  the remaining non-invariance; GBM is now the real second-closest arm.**
- **T4:** residual-model diagnostic protocol added (zero-init + free lambda scalar, dual value-MSE/R-precision
  checkpoint selection, 3 predeclared seeds, a tiny-batch overfit check -- caught and fixed two of my OWN bugs
  in this new check's row-selection logic before trusting it). Confirms no optimization bug; residual arm's
  failure looks like genuinely no transferable signal at this label budget.
- **T5:** adaptive-query benchmark completeness -- decision-aware stopping (genuinely saves queries), unique-
  query/cache accounting, the registry's declared shortlist recall/precision (previously absent), a paired-per-op
  bootstrap CI (MC significantly beats guided at every budget, p<0.05; MC vs random significant at low-mid
  budgets only), and a cached-data-only control-variate diagnostic (favorable for 97% of candidates, promising
  future work, not yet a policy).
Full suite: 86 passed both commits. See `results/phase5/{STAGE3_ADAPTIVE_QUERY,STAGE4_MATCHED_RECURRENT}.md` and
`results/phase4/CLAIM_LEDGER.md` rows 11-13 for full detail.
**T6 (independent re-verification) is DONE and PASSED.** A qa-verifier subagent that authored none of these
fixes independently verified all 6 items: it reran the FULL training script itself (not cached), reran the eval
and adaptive-query scripts (byte-identical output), wrote its own witness scripts (unequal risks, nonzero
control vector, run against the actual retrained checkpoint) to re-derive the DeepSets invariance claim rather
than trusting the pytest suite alone, hand-recomputed the control-variate diagnostic on 3 real candidates, and
independently recomputed the capacity-floor percentages and one bootstrap CI from raw JSON. Verdict: PASS on all
6, no regressions (86/86 both before and after its reruns), no blockers. Repair round 2 (T1-T6) is complete.

## Phase 5 repair round 3 (2026-09-25): independent THIRD review, of round 2 (commit 10fb6da)
Round 2 (T1-T6 above, commit 10fb6da) was pushed, then independently reviewed a THIRD time
(`ML_FDNA_10fb6da_Evidence_Review.md`, committed in repo root). Verdict: the DeepSets invariance repair and
canonical-key checks were sound, but a real stopping-threshold bug remained, and several round-2 conclusions
overclaimed what the evidence supported. Every claim independently verified before any fix. Fixed in three
commits (672c7e1 doc-only T1; 4db9fb9 code T2/T3/T6; 0c3f0b9 code T4/T5):
- **T1 (doc):** withdrew "leaking non-invariant information" (pair-risk features are legitimate, available to
  every arm -- the old model violated its OWN claimed invariance, not privileged information); corrected an
  overstated "3 seeds converging to 0.8691" residual-diagnostic claim (was a too-coarse, 2-value diagnostic).
- **T2:** `resolve()`'s decision-aware stopping compared `qU` against the LOAD-SHED threshold (spec.SEVERE=0.01)
  instead of the PROBABILITY-decision threshold (0.5) -- overly conservative, never invalid. Fixed and moved
  into `fdna.adaptive_query` with a named `PROB_DECISION_THRESHOLD` constant. Also fixed a shared, sequentially-
  advancing RNG across outages (found while fixing the threshold) -- now seeded per (op,outage,budget).
- **T3:** shortlist metrics were pooled across all 60 ops (a different question from per-op screening, confirmed
  with a real pooling-artifact witness). Added genuine per-op metrics + bootstrap CI: reveals guided is
  significantly BETTER than MC at recall@10/20% on several budgets, MC better at recall@40%/accuracy -- a
  nuanced pattern, not blanket MC superiority.
- **T4:** residual diagnostic was too coarse (12-pair sample, P1-only, pooled) -- fixed to the full 80-pair
  cross-product, per-op, both cells. Seed 1's selector now correctly reverts to the untrained zero-correction
  state (never found an improving epoch). Added a frozen-shrinkage-grid estimate (properly separated from
  `lam`'s joint training) -- ties g2_fixed, no promotion-worthy margin.
- **T5 (the big one):** DeepSets ablation -- `OrderedMLP` (plain, non-invariant) beats DeepSets by a wide margin;
  permutation-averaging it (exactly invariant by construction) closes most of the gap to g2_fixed, statistically
  tying it for one of three seeds in each cell. Points at the POOLING architecture, not invariance itself, as
  DeepSets' likely weakness. Caught and fixed 3 of my own bugs before trusting this: a small-sample (8-row)
  spread check gave a false "near-invariant" reading, corrected with a larger 64-row/all-3-seed check; a
  device-mismatch crash; and a variable-shadowing bug (`for s in ORDERED_SEEDS` clobbering an outer cell-loop
  `s`) that silently corrupted every arm's posterior sampling identically -- caught because ALL arms, including
  untouched g2_fixed, came back with an implausible, identical R-precision on the first real rerun.
- **T6:** implements the review's proposed control-variate estimator as an ACTUAL policy (new
  `scripts/p5_adaptive_query_experiment_v3.py`), not just a diagnostic -- beta=1.0 closes nearly the entire gap
  to the theoretical posterior ceiling, significantly beats plain MC at 11/12 budget/cell combinations, zero
  additional oracle cost. Exploratory only (reuses the 600-659 block); confirmatory run is future work.
Full suite: 96 passed. **T7 (independent re-verification) is DONE and PASSED.** A qa-verifier subagent instructed
to be more adversarial than the prior two rounds (given this is the third review of the same code) verified all
6 items: reran both adaptive-query scripts fully (byte-identical output), wrote its own witness scripts with
different seeds/rows than the shipped tests, recomputed the control-variate significance count and the
ablation's headline numbers directly from raw JSON, and traced the residual-seed-1 tie to actual code logic.
Verdict: PASS on all 6, no regressions (96/96). It flagged one minor rhetorical overclaim in STAGE3.md ("at most
of them" for P2's fine-grained shortlist comparison -- actually ~42%, not "most") -- corrected immediately.
Repair round 3 (T1-T7) is complete; see `results/phase5/RUN_JOURNAL.md` for the full consolidated account.

## Open / next (needs Philip)
1. Push the local commits? (asked before every push) -- round 2 (T1-T6, up to 10fb6da) was pushed; the
   repair-round-3 commits (672c7e1, 4db9fb9, 0c3f0b9) have NOT been pushed yet.
2. T7's independent verification report, once it arrives: resolve anything it flags before treating round 3's
   corrected conclusions (especially the DeepSets/pooling-architecture finding) as final.
3. Direction (unchanged, still open): (a) benchmark + protocol paper (workshop/IEEE Access tier); (b) pre-registered scale experiment (large dependency graph, approximate-posterior oracle, headroom gate first);
   (c) N-1->N-2 label-efficiency method; (d) second topology (IEEE-118).
4. Commits authored by Philip only, no attribution lines.

<!-- AUTO-HANDOFF (PreCompact/auto) 2026-09-24T10:57:15Z -->
### Compaction handoff — 2026-09-24T10:57:15Z
- Git: branch `main`, 0 uncommitted file(s): 
- Last verification run recorded: 2026-09-24T10:57:14Z	python3 - <<'EOF' def edit(p, pairs): s=open(p).read() for a,b in pairs: assert a in s, (p, a[:50]) s=s.replace(a,b) ope
- RESUME: re-read the Task/Status/Next-action sections above; trust this file over recollection.

<!-- AUTO-HANDOFF (PreCompact/auto) 2026-09-25T15:09:23Z -->
### Compaction handoff — 2026-09-25T15:09:23Z
- Git: branch `main`, 6 uncommitted file(s): data_hik/deepsets_model.pt data_hik/deepsets_model_v2.pt data_hik/matched_recurrent_models.pkl data_hik/matched_recurrent_models_v2.pkl data_hik/residual_model.pt data_hik/residual_model_v2.pt 
- Last verification run recorded: 2026-09-25T15:08:56Z	git commit -q -m "Phase 5 Stage R1 (3/6, 4/6): fix all four matched-recurrent-experiment bugs and the linear-bootstrap c
- RESUME: re-read the Task/Status/Next-action sections above; trust this file over recollection.
