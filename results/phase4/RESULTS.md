# Phase 4 — Results (exploratory/screen level; no confirmatory block opened yet)

**Status: exploratory.** Everything below is a screen: one sampled k=3 manifest (60 triples x 20 operating
points, seed 42, frozen and committed before any label was generated — `data_hik/manifest_k3_screen.json`),
single fixed control states (not the full partial-observation pipeline), no cluster bootstrap yet, no fresh
confirmatory block opened. Reported honestly as a strong screen result, not a confirmed claim.

## Headline finding: low-degree Möbius/interaction truncation generalises from N-1/N-2 to exact N-3 labels

**Setup** (`scripts/p4_mobius_k3_test.py`, `scripts/p4_mobius_k3_robust.py`). For each of 60 frozen outage
triples `{a,b,c}` across 20 operating points, at a fixed control state `c*`:
- `g_1(S) = y({a}) + y({b}) + y({c})` — naive singleton sum, ignoring all interaction (a "recurrence with no
  interaction term" model).
- `g_2(S) = g_1(S) + I(a,b) + I(a,c) + I(b,c)`, `I(x,y) = y({x,y}) - y({x}) - y({y})` (`y(empty set)=0` exactly,
  independently verified this session) — the k=2-additive Möbius/interaction truncation, using ONLY N-1 and N-2
  information, fit with **zero training** (pure closed-form arithmetic).
- `gbm_equal_info`: a LightGBM regressor given the identical 6 numbers (`y_a, y_b, y_c, I(ab), I(ac), I(bc)`),
  5-fold cross-validated to avoid overfitting-inflated comparison on 1,200 rows — the required equal-information
  baseline, not just an information-blind naive model.
- All three are scored against the TRUE, exactly-solved N-3 label (`data_hik/k3_screen.npz`, 1,200 rows, 0
  infeasible solves).

**Result, at four different control states (full control, no control, and two intermediate points):**

| control state | g1 MAE | g2 (Mobius) MAE | GBM equal-info MAE | g2 vs g1 | g2 vs GBM | g2 rho | GBM rho |
|---|---|---|---|---|---|---|---|
| full (sum(c)=5.0) | 0.00800 | **0.00087** | 0.00267 | 9.2x better | 3.1x better | 0.959 | 0.829 |
| none (sum(c)=0.0) | 0.01509 | **0.00374** | 0.00509 | 4.0x better | 1.4x better | 0.982 | 0.973 |
| mid (sum(c)=0.6) | 0.01538 | **0.00360** | 0.00506 | 4.3x better | 1.4x better | 0.981 | 0.973 |
| low (sum(c)=1.4) | 0.01282 | **0.00127** | 0.00431 | 10.1x better | 3.4x better | 0.956 | 0.896 |

**g2 beats both baselines on MAE in all 4 conditions**, and matches or beats them on Spearman correlation with
the true value. This holds using ONLY N-1/N-2 information — the model never sees a single N-3 label during
fitting (g2 has no fitting step at all).

**What this means, stated carefully:**
- This directly and empirically engages the brief's stated impossibility (identical singleton values can
  differ arbitrarily on pairs/triples in the worst case) — and on THIS LP structure, the low-degree assumption
  holds well enough that an explicit degree-2 truncation captures the dominant part of the N-3 interaction.
- The GBM comparison shows this is not merely "any information beats none" — a flexible learner given the
  identical 6 numbers does worse than the closed-form structural model in every condition tested. That is
  evidence FOR the specific additive/interaction structure, not just for having N-2 information available.
- **What is NOT yet established:** (a) whether this holds at N-4 (not tested); (b) whether it holds under
  partial observation / communication uncertainty (this screen is electrical-only, exact control, per the
  brief's own staged design in section 6.3 — "first use exact controls to isolate electrical generalization");
  (c) whether a richer equal-information baseline (raw electrical/topological features, not just the 6 Möbius
  numbers — the brief's actual "tuned tree on full descriptors" comparator) would close more of the gap; (d)
  statistical significance with proper clustering (60 distinct triples x 20 ops is not 1,200 independent draws;
  a cluster bootstrap over triples and/or operating points has not yet been run); (e) whether this is FDNA in any
  sense — it is not; it is a generic property of the LP's value function, consistent with the earlier finding
  that FDNA's algebra is dominated on this benchmark.
- This finding is exploratory and screens on a small (60-triple) manifest. It has not been confirmed on a
  fresh block, and per this project's own discipline it cannot be reported as more than a strong, reproducible
  screen result until it is.

## Repaired-FDNA inference-only screen (registry/phase3_step2.yaml correction, brief 4.3)
Running as a single GPU process (not the earlier contended two-process run). Partial results at time of writing
(cell P1/v2b): I5_ref (flat, mean-field) NLL 4.80, I5_tau0 (hard-min) NLL 4.59, I5_OR (OR-aware, corrected
semantics) NLL 4.78, I5_OR_hard NLL 4.54, J_plain (joint head, independent prior) NLL 3.86 — the joint head's
lower NLL is expected (it models the full 16,384-state posterior exactly, mean-field arms cannot). Full results,
including the common-cause-prior and logic-head arms and cell P2, pending completion; this is an inference-quality
check only (per the research-scientist's power analysis earlier this session), not an R-precision claim.

## Diagnostics generalized and correctness-gated (not yet used in a predictive model)
`src/fdna/hik_diag.py`: `generalized_det` (k-way LODF compensation determinant) and `minimal_cut_struct`
(k-way structural shed floor), both exact and LP-solve-free, both gated against the already-verified k=2 facts
(`tests/test_hik_diag.py`, 5/5 pass; catching and fixing a real bug — `minimal_cut_struct` initially used the
grid's nominal load instead of the operating point's actual demand).

## Next steps (not yet done)
1. Add a richer equal-information baseline (raw electrical/topological features per operating point, matching
   the brief's staged comparator set) to properly test card 1 against the strongest baseline, not just a 6-number GBM.
2. Extend the Möbius test to N-4 using the same frozen-manifest discipline.
3. Reintroduce partial observation (communication uncertainty) and test whether the effect survives.
4. Cluster-aware statistics (bootstrap over triples and operating points) before any tier claim.
5. Complete the repaired-FDNA inference-only screen (cell P2, remaining arms) and report a representability
   verdict (not an R-precision claim, per the noise-floor analysis already on record).
