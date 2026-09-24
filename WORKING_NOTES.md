# WORKING_NOTES — ML_FDNA
Plan: ~/.claude/plans/using-brainstorming-research-ideas-and-d-witty-stonebraker.md (revision 3, approved).
Status (2026-09-23): Milestone 1 and 1b done locally. Steps 1-4 complete: reproducibility fixes, physical-assumption pilot, report
(results/MILESTONE1.md), learning-curve experiment (results/LEARNING_CURVE.md): H1 NOT SUPPORTED (explicit control features worse
than raw flags at n<=25). Decision rule: stop FDNA architecture claim on this benchmark.
Reviews: qa-verifier PASS (6/6); code-reviewer found 4 major issues, all fixed (hash scope, cache key, tracked results, report wording).
Repo: 21 tests pass. Label spec hash: scripts/spec_hash.py. data/ and data_fresh/ are gitignored (regenerate: scripts/gen_dataset.py).
Local commits ahead of origin (pushed head: 31aff42). Push only when Philip says so. Commits authored by Philip, no attribution lines.
Open: decide next question with Philip (uncertain/stale dependency info; changed wiring); Roy & Hylviu notes in results/LITERATURE_NOTES.md.
