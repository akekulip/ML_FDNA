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

## Open / next (needs Philip)
1. Push the local commits? (asked before every push)
2. Direction: (a) benchmark + protocol paper (workshop/IEEE Access tier); (b) pre-registered scale experiment (large dependency graph, approximate-posterior oracle, headroom gate first);
   (c) N-1->N-2 label-efficiency method; (d) second topology (IEEE-118).
3. Commits authored by Philip only, no attribution lines.

<!-- AUTO-HANDOFF (PreCompact/auto) 2026-09-24T10:57:15Z -->
### Compaction handoff — 2026-09-24T10:57:15Z
- Git: branch `main`, 0 uncommitted file(s): 
- Last verification run recorded: 2026-09-24T10:57:14Z	python3 - <<'EOF' def edit(p, pairs): s=open(p).read() for a,b in pairs: assert a in s, (p, a[:50]) s=s.replace(a,b) ope
- RESUME: re-read the Task/Status/Next-action sections above; trust this file over recollection.
