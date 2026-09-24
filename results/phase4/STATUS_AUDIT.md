# Phase 4 — State audit (2026-09-24, start of Phase 4)

## Hardware / processes (measured this session)
GPU: RTX 2070, 8192 MiB, 274 MiB used, 0% util at audit time. CPU: 32 cores, load average 0.73/0.88/1.30 (idle). RAM: 31 GiB total, 26 GiB available. No ML_FDNA python/training processes running (checked `ps aux`; only unrelated MCP server processes for other projects). torch 2.5.1+cu121, `torch.cuda.is_available()` True. `uv run pytest -q`: **57 passed**, 6 warnings, 30.3s.

## Git
Branch `main`. HEAD = pushed = `943e09a81733c2538ec2289991f4b63a6f52584b` (`git status -sb` shows `main...origin/main` with zero ahead/behind at audit time, aside from the brief file and this audit itself, both untracked/new). Recent history: correction/lock commits from this session (7053232, 2b06bc2, 943e09a) plus the Phase-2 close (15c71cd, a1af8ad).

## Registry / locks
`registry/slots.yaml` has 6 slots, all from the H5 decomposition claim (INF_P1/P2, D_P1/P2, INF8_P1/P2) — **no Phase-3 or Phase-4 slot exists yet**. `registry/locks/` holds exactly 4 lock files (confirm/replicate x D_P1/D_P2) plus `access.log` — matches the H5 confirm+replicate history; no other block was ever opened. Reserve block 600-699 is unopened and, per `slots.yaml`'s own comment, "never allowed" under any current slot.

## Per-item status (brief's required classification)

| Item | Status | Evidence |
|---|---|---|
| Phase 1/2 benchmark, baselines, no-FDNA-advantage finding | DONE AND VERIFIED HERE | `results/SUMMARY.md`, `results/PHASE2.md`, 57 tests pass |
| H5 decomposition effect (+0.03, tier 2, not tier 1, blocks shared training draws) | DONE AND VERIFIED HERE (with the stated caveat) | `registry/amendment_H5.yaml`, `results/LEDGER.md` row 13 |
| Phase 3 Step 1 diagnostics (topology classes, interactions, flow-addition error, comm-bit structure, FDNA motifs, A9 equal-info) | DONE AND VERIFIED HERE | `results/phase3/STEP1.md`, `registry/phase3_step1.yaml`, scripts `p3_step1_*.py` |
| LODF compensation-determinant mechanism (Det=0 <=> new 2-cut; correlation with flow-addition error) | DONE AND VERIFIED HERE, **with a corrected number** | `src/fdna/lodf.py`, `tests/test_lodf.py` (3/3 pass); the "8.4%" figure in `STEP1_EXT.md` is WRONG — corrected to ~10.1% in Stage 1 below |
| Jensen-gap mechanism / direction check | REPORTED BUT INVALID AS COMPUTED | `scripts/p3_step1_jensen.py` evaluates `V(round(E[C]))`, not `V(E[C])`; direction (P1>P2, always non-negative) may still hold but the numbers (0.0042/0.0027) are not the claimed quantity — being fixed in Stage 1 |
| FDNA repair module (OR-aware aggregation, hard-min, joint/common-cause head) | IMPLEMENTED, PARTIALLY EVALUATED | `src/fdna/nn/repair.py`, 5/5 known-answer tests pass (representability: current flat layer cannot fit the OR motif at any tau, confirmed); **but** the tau ablation as coded conflates OR-semantics with temperature — being fixed in Stage 1 |
| Repaired-FDNA training screen (R-precision comparison) | STOPPED, NEITHER POSITIVE NOR NEGATIVE EVIDENCE | Killed at ~44 min CPU with 0 completed replicates (`registry/phase3_step2.yaml`'s own noise-floor analysis already showed it could not resolve its predicted effect even if it had finished) |
| Step 3 island-decomposition confirmatory design | PLANNED, NOT EVALUATED | `registry/phase3_step3.yaml`; endpoint has a degeneracy risk (all-severe subset) flagged by the brief, being fixed in Stage 1 |
| Recurrent / higher-k (N-3/N-4) generalisation, any model | NOT STARTED | No such code exists in the repo prior to this session; `dataset.py`'s `PAIRS` and `value_table.py`'s `conts` are pair-only |
| Second topology (IEEE-118) | NOT STARTED | `grid.py` loads only `case_ieee30` |
| WORKING_NOTES.md | INVALID / STALE | Says "pushed head 31aff42" and "48 tests"; actual pushed head is 943e09a3, 57 tests. Refreshed below. |

## WORKING_NOTES.md refresh
Superseded lines corrected: pushed head is now `943e09a`; local test count is 57 (was 48 at the point that note was written, before Phase 3). Phase 3 (diagnostics, LODF mechanism, repair module, Jensen check) is complete through the point above; Phase 4 (this brief) begins now. Full detail lives in this file and `results/phase3/STEP1_EXT.md`, not restated in WORKING_NOTES.md itself.
