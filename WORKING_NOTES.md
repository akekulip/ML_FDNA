# WORKING_NOTES — ML_FDNA
Plan: ~/.claude/plans/using-brainstorming-research-ideas-and-d-witty-stonebraker.md (revision 3, approved).
Status (2026-09-23): Milestone 1 and 1b done locally. Steps 1-4 complete: reproducibility fixes, physical-assumption pilot, report
(results/MILESTONE1.md), learning-curve experiment (results/LEARNING_CURVE.md): H1 NOT SUPPORTED (explicit control features worse
than raw flags at n<=25). Decision rule: stop FDNA architecture claim on this benchmark.
Reviews: qa-verifier PASS (6/6); code-reviewer found 4 major issues, all fixed (hash scope, cache key, tracked results, report wording).
Repo: 21 tests pass. Label spec hash: scripts/spec_hash.py. data/ and data_fresh/ are gitignored (regenerate: scripts/gen_dataset.py).
Local commits ahead of origin (pushed head: 31aff42). Push only when Philip says so. Commits authored by Philip, no attribution lines.
Open: decide next question with Philip (uncertain/stale dependency info; changed wiring); Roy & Hylviu notes in results/LITERATURE_NOTES.md.


## PHASE 2 (overnight programme) — started 2026-09-23T23:00:52-04:00; hard wall-clock cap 10 h (new experiments stop at H9.5).
Plan: ~/.claude/plans/using-brainstorming-research-ideas-and-d-witty-stonebraker.md. Local commits only; installs only; no web/literature calls.

### Phase 2 log (times are wall-clock; started 23:00:52)
- 23:00 torch 2.5.1+cu121 + optuna installed; GPU visible. 23:05 registry + v2 spec frozen and committed (870f487) BEFORE any experiment.
- 23:04 value-table generation started (train/val done; test split in progress) -> E1 (oracle headroom) runs after.
- 23:25 Branch 3 first run INVALID (NN targets ~1e-2 starved gradients; diagnosis scripts/nn_diagnose.py, validation only). Fixed trainer (Y_SCALE=100), variant B3v2 registered before rerun.
- 23:38 B3v2 result: monotone NN 0.06-0.13 below monotone GBM at n<=25; monotone GBM ~0.025 below free GBM -> B3 killed as hypothesis (registry outcomes).
- 23:38 Branch 2 (corrupted wiring, rho=0.2) running: /tmp/b2.log -> data/b2_results.parquet.
Next: E1 when data_v2/vtable_test.npz exists; B2 result; then decide freeze of v2 (only after E1 passes).
