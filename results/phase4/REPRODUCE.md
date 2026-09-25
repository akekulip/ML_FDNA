# Phase 4 — Reproduction

For the subsequent all-seed evaluation and total-cost screening experiment, use
the [Phase 5 development report](../phase5/cost_quality_dev/REPORT.md) and
[matched-model report](../phase5/STAGE4_MATCHED_RECURRENT.md). Test counts below
describe historical commits, not the current checkout.

Environment: `uv` project at repo root, Python via `.venv` (torch 2.5.1+cu121, CUDA available, RTX 2070 8GB).
`uv run pytest -q` should show 70 passed at commit 784b099 (this file's commit; check `git log -1` for the
current head, which may have moved).

## Corrections (Stage 1)
- `uv run python scripts/p3_step1_jensen.py` -> `results/phase3/step1_jensen.json`
- `uv run pytest -q tests/test_repair.py` (6/6; the fractional-input test specifically checks the OR/tau fix)

## Higher-k schema and diagnostics (Stage 3)
- `uv run pytest -q tests/test_hik.py tests/test_hik_diag.py` (12/12; includes the backward-compat gate against
  the old N-0/N-1/N-2 oracle, exact match atol 1e-6, and the two correctness gates for `generalized_det` /
  `minimal_cut_struct`)
- Manifests (frozen, seeded, committed before any label): `data_hik/manifest_k3_screen.json` (seed 42, k=3, 60
  sets, 20 ops), `data_hik/manifest_k4_screen.json` (seed 43, k=4, 25 sets, same 20 ops),
  `data_hik/manifest_train_n2sample.json` (seed 100, 300 sampled N-2 pairs, same 20 ops)
- Label generation (bulk data git-ignored, regenerate with):
  - `uv run python scripts/hik_label_k3.py` -> `data_hik/k3_screen.npz` (1,200 rows, 0 infeasible; ~135s on 24
    cores)
  - `uv run python scripts/p4_mobius_k4_test.py` -> also writes `data_hik/k4_screen.npz` (500 rows; the N-4
    Mobius test and the label generation are the same script)
  - `uv run python scripts/hik_gen_train_data.py` -> `data_hik/train_n1n2.npz` (6,820 rows)

## Core results (Stage 3)
- `uv run python scripts/p4_mobius_k3_test.py` -> `results/phase4/mobius_k3_test.json`
- `uv run python scripts/p4_mobius_k3_robust.py` -> `results/phase4/mobius_k3_robust.json` (4 control states)
- `uv run python scripts/p4_mobius_k3_richbaseline.py` -> `results/phase4/mobius_k3_richbaseline.json`
- `uv run python scripts/p4_mobius_k4_test.py` -> `results/phase4/mobius_k4_test.json`
- `uv run python scripts/p4_setmodels.py` -> `results/phase4/setmodels_k3k4.json` (~2 min on CPU with
  `torch.set_num_threads(4)`; do not run alongside another GPU/CPU-heavy job, per this session's own concurrency
  lesson)

## Repaired-FDNA inference-quality screen (registry/phase3_step2.yaml correction)
- Single process only (two concurrent copies contend for the one GPU and stall — measured this session):
  `VARIANT=v2b REPS=3 OMP_NUM_THREADS=4 uv run python scripts/p3_repair_infer_only.py` (cell P1, ~2,135s /
  ~36 min), then the same with `VARIANT=v2c` for cell P2. Output: `data_v2/p3_repair_infer_{variant}_3.parquet`
  (git-ignored; regenerate).

## What is NOT yet reproducible / not run
- No confirmatory block opened (registry/slots.yaml has no Phase-3/4 entry). Everything in Stage 2/3 is an
  exploratory screen.
- Second topology (IEEE-118) not attempted.
- Partial-observation (communication-uncertainty) layer not reintroduced into the higher-k schema.
- A cluster bootstrap over operating points/triples has not been run on any Stage-3 number.

## Manifest hashes (for provenance; recompute with `sha256sum`)
Run `sha256sum data_hik/manifest_k3_screen.json data_hik/manifest_k4_screen.json data_hik/manifest_train_n2sample.json`
after checkout to verify the frozen manifests are byte-identical to what this session generated (they are
tracked in git, not regenerated, so this should always match at a given commit).
