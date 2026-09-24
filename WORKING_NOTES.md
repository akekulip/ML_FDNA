# WORKING_NOTES — ML_FDNA
Repo: ~/Projects/ML_FDNA (github.com/akekulip/ML_FDNA private; pushed head 31aff42; everything since is LOCAL commits in Philip's name, no push yet).
Read first: results/SUMMARY.md, results/PHASE2.md, results/LEDGER.md, registry/*.yaml (locked hypotheses, addenda, amendment, slots, locks).

## State (2026-09-24 ~07:00)
Phase 1 (v1 benchmark) and Phase 2 (overnight programme, ~8 h) complete. Outcome: no registered FDNA claim survived; one small confirmed and replicated non-FDNA
effect (decomposition, +0.025..+0.036 tier 2); structure is a ceiling/fragile; dominant error is unseen N-2 generalisation (label coverage), not dependency modelling.
Confirmatory looks: four block openings, all for the single H5 claim (confirm 400-479 and replicate 500-579, cells P1/P2); reserve 600-699 and all other slots unopened.
Independent reviews (QA + code) at H4 and at close; all findings fixed or disclosed. Tests: 48 pass.

## Data (git-ignored; regenerate with scripts/value_table.py, gen_dataset.py)
data/ (v1 labels + ops), data_fresh/, data_v2/ (value tables, screens, results parquet), data_v2_confirm/ (label tables for blocks 400-479, 500-579), data_v2_fs/.

## Open / next (needs Philip)
1. Push the local commits? (asked before every push)
2. Direction: (a) benchmark + protocol paper (workshop/IEEE Access tier); (b) pre-registered scale experiment (large dependency graph, approximate-posterior oracle, headroom gate first);
   (c) N-1->N-2 label-efficiency method; (d) second topology (IEEE-118).
3. Commits authored by Philip only, no attribution lines.
