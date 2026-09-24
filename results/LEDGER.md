# Forking-paths ledger (exploratory block 200-279; written 2026-09-24 from the statistician audit and the commit history)
The exploratory block was inspected in Phase 1 and reused by every Phase-2 screen. Its intervals are descriptive, not confirmatory.
"Pre-reg" = committed to `registry/` before the run started. "Steered" = the outcome changed later design.

| # | Branch / variant | Pre-registered before running? | Outcome | Steered later design? |
|---|---|---|---|---|
| 0 | Phase 1 (v1 benchmark, H1 learning curves, diagnostics) | H1 yes; diagnostics post hoc | explicit control features do not beat raw flags | yes (defined Phase 2) |
| 1 | B3 monotone value model: run 1, then B3v2 (target x100) | v2 registered after diagnosing run 1 on validation data | killed | yes (-> B2) |
| 2 | B2 corrupted wiring | yes | killed vs trees; side finding (A5 > generic MLP) post hoc | yes |
| 3 | E1 oracle headroom (v2 mild) | yes | passed as registered, INVALID measure | yes (-> E1b) |
| 4 | E1b gate on v2 | written after seeing E1 | failed (+0.027; +0.032 under a different aggregation) | yes (-> v2b/v2c) |
| 5 | v2b, v2c worlds (harsher telemetry, sparse coverage) | registered before their gate ran, after the v2 gate failed | passed E1b (later shown to measure irreducible information loss) | yes: **worlds were made harsher until a gate opened** |
| 6 | B1 end-to-end arms A1-A7, then B1v2 rerun after the softmin fix | yes | killed (rerun pending at time of writing) | yes |
| 7 | Inference-only branch, first run (invalid), rerun | registered before coding | null after the fix | yes (-> I8) |
| 8 | Decomposed composition, first run (invalid), rerun (top-64, +A8) | registered before coding; top-16 -> 64 and A8 added after the first run | +0.015..+0.039 over the strongest end-to-end arm | yes |
| 9 | I8 exact differentiable Bayesian layer | registered after seeing the inference KL results, before coding | null in R-precision, near-exact posterior | yes |
| 10 | Wiring-misspecification sweep for I8 | registered before coding | (see PHASE2) | - |
| 11 | Value ladder (feature artifact?) | registered before coding | not a feature artifact | - |
| 12 | Extra composition cells | screens of the remaining v2b/v2c cells, exploratory | (see PHASE2) | - |
| 13 | H5 amended confirmatory claim: decomposition D_I2 vs A8, n=100, cells P1 and P2 | amendment committed before the confirm block was opened; **decided after seeing screen effects** | pending | - |
| post hoc | raw_edge (capacity dependence of the raw-flag edge), oracle_check (P(severe) oracle), P(severe) secondary scoring, mean-field/KL analyses | no | exploratory | some |

Facts for readers: about 15 gate or kill decisions, at least 6 reruns of the same 80 operating points on the same hidden-state draws; primary sizes
for D (n=25) and INF (N=5000) were declared after the first runs; two of my own gates (E1, E1b) measured the wrong quantity and were found by
inspecting reference arms; an independent review found and I fixed a softmin bias, block-seed reuse and an advisory lock; the screens used weaker
tuning (12 GBM trials / 4-point NN grid) than the registry's original protocol text (nested 40-trial TPE), and confirm-level tuning is 40 trials / 9 points.
Confirmatory looks consumed: two block openings, both for the single H5 claim (P1 and P2), as one intersection-union test.
