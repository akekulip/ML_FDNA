# Phase 2 — overnight programme log and results (living document)
Started 2026-09-23 23:00:52; hard wall-clock cap 10 h. Registry: `registry/registry.yaml` + `registry/addendum_H4.yaml` (locked before
the experiments they govern). Screens use the inspected exploratory block 200-279. **The confirm (400-479) and replicate (500-579) blocks
have not been opened** (no `registry/locks/` entries; only their label tables exist). Independent QA/code review at the H4 gate found
real problems (below); everything affected is being re-run.

## Chronology of the programme (each pivot was forced by a diagnosed result)
| Time | Step | Outcome |
|---|---|---|
| 23:25-23:38 | Branch 3 (monotone value model) | first run invalid (neural targets ~1e-2 starved gradients; diagnosed on validation data, fixed); rerun: monotone NN 0.06-0.13 below monotone GBM. Killed. |
| 23:38-23:56 | Branch 2 (learned dependency layer, 20% wrong wiring) | 0.10-0.16 below tuned trees; but +0.014/+0.047/+0.092 over a generic MLP. Killed vs trees; side finding. |
| 00:00-01:00 | E1 (oracle headroom on v2 partial observation) | passed as registered (+0.12..+0.16) but INVALID: the oracle receives the exact LP value table; trees given exact posterior marginals gain nothing. |
| 00:59-01:27 | E1b (exact-control GBM minus best observation tree) | mild telemetry: max +0.027 (fail); pre-registered harsher variants v2b/v2c pass (+0.03..+0.10). Later found to measure irreducible information loss, not reducible headroom. |
| 01:27 | v2 family frozen (tag `v2-freeze`), primary cells P1 (v2b q=0.7,s=0.3) and P2 (v2c q=0.3,s=0.2) | |
| 01:40-02:50 | B1 end-to-end arms A1-A7 | A5 (FDNA belief net) 0.04-0.08 below the best arm (A2, logic-feature GBM) in both cells; beats shuffled/unconstrained controls only in P2. Killed (subject to re-run after the softmin fix). |
| 02:16- | Inference-only branch; decomposed-composition branch | registered before coding; first runs invalidated by the softmin bug; re-running. |

## Key diagnoses
1. Neural baselines must be sanity-checked against trees first (target scale x100 fixed a 0.3 R-precision gap).
2. With complete observation, trees do not need dependency structure; structure helps neural nets relative to generic MLPs only.
3. **Headroom decomposition (main methodological result so far):** the gap between a learner given the exact control vector and one given observations mixes irreducible information loss with reducible inference headroom. The reducible part (GBM given exact posterior marginals minus best observation tree) is only +0.007..+0.021 in P1/P2, so better inference cannot help a tree value model by more than ~0.02. Two of my own registered gates (E1, E1b) measured the wrong quantity and were corrected by inspecting reference arms.
4. The posterior-mean-of-shed predictor is not R-precision-optimal: exact P(severe|obs) scores 0.916 (P1) / 0.943 (P2) versus 0.865 / 0.912.
5. **Reduced-effort control needed:** `raw_edge` (exploratory): the ~0.02 raw-flag edge over explicit control features on novel control vectors depends on tree capacity (+0.02 for small trees, ~0 for 63 leaves) and reverses sign on familiar vectors; not a claim.

## H4 independent review (QA verifier + code reviewer) — findings and actions
Blockers/majors found and fixed before any confirm run: confirm block reused the exploratory block's hidden states (same seed); fresh-block lock was advisory; FDNA softmin capped operability at ~0.945 (invalidates FDNA arms in all screens, hence the re-runs); shuffled control used one mask; I2 trained on different observations; decision rules existed only on paper (now code: `src/fdna/rules.py`); tuning protocol differed from the registry text (now disclosed; screen vs confirm tuning levels declared); registry deltas quoted with an unstated aggregation (both aggregations now reported). Details: `registry/addendum_H4.yaml`.

## Pending (in progress)
Re-runs with the fixed FDNA layer: inference-only screen (P1, P2), decomposed composition (P1, P2), B1v2_softmin_fix (P1, P2). Then comparators fixed from those screens, and only claims with a positive screen go to the confirm block.
