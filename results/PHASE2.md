# Phase 2 — overnight programme: results and honest assessment
Run 2026-09-23 23:00 to 2026-09-24 ~09:00 (autonomous). Governing files: `registry/registry.yaml`, `registry/addendum_H4.yaml`,
`registry/amendment_H5.yaml`, `registry/slots.yaml`, `results/LEDGER.md` (forking-paths ledger), `prereg/SPEC_V2.md`. Every screen ran on the
inspected exploratory block (operating points 200-279), so its intervals are descriptive. Confirmatory looks: **one claim, opened once per cell on
the confirm block (400-479) and, after it was confirmed, once on the replicate block (500-579)**; all other declared slots were never opened.

## Bottom line
1. **No FDNA-specific advantage was found anywhere.** Every FDNA contrast is null or negative against the best appropriate baseline (table below).
2. **A small, confirmed, non-FDNA effect exists:** learning the value model on full-information simulation rows and the inference module on cheap
   (obs, c) pairs, then composing, beats the strongest end-to-end baseline that also receives the cheap pairs by **+0.036 (P1) and +0.025 (P2)**
   R-precision on the fresh confirm block (tier 2: reliable but below the pre-declared 0.05 threshold). REPLICATE_RESULT_PLACEHOLDER
3. **Structure is a ceiling, not a lever:** an exact differentiable Bayesian layer over the known dependency graph (17 learnable parameters) reaches
   KL 0.02 to the exact posterior at N=1,000 samples, yet gains at most ~0.01 R-precision over a strong per-generator GBM; with 10% of the wiring
   wrong (2 of 17 edges) it falls 0.07-0.13 below the wiring-agnostic learner. Exact structure helps only if it is exactly right.
4. **The dominant error is elsewhere:** unseen N-2 generalisation under the train-on-N-1 protocol (0.93-0.97 on N-1 rows vs 0.79-0.89 on N-2 pairs
   with the exact control vector known). Better features do not close it; 10 labelled N-2 pairs per operating point close 25-40% of it.
5. **Methodological lessons that changed conclusions** (each found by inspecting reference arms or by independent review, and fixed): two of my own
   headroom gates measured the wrong quantity; neural baselines needed a target-scale sanity check; an un-normalised soft-min capped FDNA
   operability at ~0.945; the confirm block had reused the exploratory block's hidden-state draws; the fresh-block lock was advisory; the
   posterior-mean predictor is not the R-precision-optimal ranking (exact P(severe|obs) scores 0.916/0.943 vs 0.865/0.912).

## Screens (exploratory block; P1 = v2b q=0.7 s=0.3, P2 = v2c q=0.3 s=0.2; R-precision, higher is better)
| Branch | Question | Result | Status |
|---|---|---|---|
| B3 monotone value model | hard monotone-in-control prior vs monotone GBM | NN 0.06-0.13 below | killed |
| B2 corrupted wiring | learned dependency layer vs tuned trees at 20% wrong edges | 0.10-0.16 below trees; +0.014..+0.092 over generic MLP | killed |
| E1 / E1b gates | oracle / information-loss headroom | both measured the wrong quantity | corrected |
| B1 end-to-end (rerun, fixed layer) | FDNA belief net A5 vs best of A1-A4 | -0.102/-0.074 (P1), -0.062/-0.074 (P2); vs unconstrained A7 -0.011..+0.026 | killed |
| Inference-only | FDNA mean-field I5 vs best generic at N=5000 | -0.006 (P1), -0.021 (P2); vs shuffled +0.04; KL to exact 1.0-2.3 (generic 0.03-0.6) | null |
| I8 exact Bayes structure | structure-known posterior vs best generic at N=1000 | -0.019 (P1), +0.002 (P2); KL 0.01-0.02 | null (ceiling) |
| Misspecification sweep | I8 with rho of edges wrong vs best generic | rho=0.1: -0.115/-0.073; 0.2: -0.19/-0.20; 0.3: -0.21/-0.18 (N=1000, P1/P2) | structure fragile |
| Decomposed composition | best D arm vs strongest end-to-end (A8) | +0.015..+0.039 (below 0.05) | carried to confirm (H5 amendment) |
| Value ladder | is the value gap a feature artifact? | R1/R2 close -0.28..+0.13 of the gap | not an artifact |
| Few-shot N-2 | labelled N-2 pairs per operating point | gap closed 25%/29%/39% (k=10/30/100, 1% threshold) | label-limited |

## Confirmatory (single amended claim; `registry/amendment_H5.yaml`, `scripts/confirm_D.py`)
Treatment D_I2 (per-generator GBM inference, composed over the top-64 posterior vectors) vs comparator A8 (GBM on observations plus marginals from a generic
inference net trained on the same cheap pairs), n=100 training operating points, 5 replicates, confirm-level tuning (40 GBM trials, 9-point NN grid):
| cell | mean difference | 90% interval |
|---|---|---|
| P1 (confirm) | +0.0357 | [+0.0294, +0.0424] |
| P2 (confirm) | +0.0249 | [+0.0194, +0.0309] |
Tier 1 (>= 0.05) not supported; tier 2 (reliable, > 0) supported in both cells; tier 3 (negligible) not supported. REPLICATE_TABLE_PLACEHOLDER
Caveat: the treatment uses simulator-side privileged information (true control vectors during value training; cheap pairs from a known generative model);
this is a claim about privileged-information training, not about deployable screening and not about FDNA.

## What is and is not supportable
Supportable: the benchmark and evaluation protocol; the headroom decomposition and the gate pitfalls; the ceiling-and-fragility characterisation of
structured inference; the small confirmed privileged-information effect; the N-1 to N-2 label-coverage finding. **Not supportable:** any FDNA advantage;
generalisation beyond a 14-component, enumerable, DC-LP, synthetic-wiring benchmark; describing the run as fully pre-registered (it is a registered
exploratory protocol with an amended confirmatory claim).

## Reproduce
`uv sync`; value tables: `scripts/value_table.py`; tests: `uv run pytest -q`; results regenerate from the scripts named in each registry entry; data dirs are
git-ignored. Independent reviews (QA + code review at H4) and their fixes: `registry/addendum_H4.yaml`.
