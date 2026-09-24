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
Exploratory screen only: N-4 untested, partial observation not reintroduced, no cluster bootstrap, no confirmatory block. Repaired-FDNA inference-only screen
(NLL/KL, single GPU process) running in background at /tmp/infonly_v2b.log, cell P1 in progress.
Confirmatory looks so far: four block openings, all for the single H5 claim (confirm 400-479, replicate 500-579, cells P1/P2); reserve 600-699 and all other slots
unopened (registry/slots.yaml has no Phase-3/4 entry yet -- add one before opening any new block). Tests: 58 pass.

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
