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

## N-4 extension: does the truncation order need to grow with k?

Per the brief's own staging rule ("extend to N-4 only if the N-3 screen passes"), and since it passed
decisively above, a frozen k=4 manifest (25 quadruples x 20 ops, seed 43, `data_hik/manifest_k4_screen.json`,
committed before any label was generated) was exactly solved (6,040 LP solves: singles, sub-pairs, sub-triples,
and the true quadruple label, all at one fixed control state). `scripts/p4_mobius_k4_test.py`:

| model | information used | MAE | Spearman rho vs true N-4 |
|---|---|---|---|
| g1 (naive singleton sum) | N-1 only | 0.01098 | 0.668 |
| g2 (order-2 Mobius truncation) | N-1 + N-2 | 0.00391 | 0.935 |
| g3 (order-3 Mobius truncation) | N-1 + N-2 + N-3 | **0.00126** | **0.982** |

**Reading:** order-2 truncation (the same model that worked for N-3) still beats the naive baseline by 2.8x at
N-4, so low-order structure has NOT collapsed. But g3 (adding the now-genuinely-available N-3 interaction terms)
improves further, by another 3.1x, showing the truncation order needs to grow somewhat as k grows — the
approximation does not stay fixed at order 2 forever. Even so, g3's error is small relative to the mean true
value (0.0013 vs 0.020, about 6% relative error) using information three orders below the target (N-1/2/3
predicting N-4), which is a genuinely cheap and informative result: **each additional truncation order costs a
combinatorially SMALLER label budget than the target order** (100 triples here vs 25 quadruples needed, but the
triples generalize across many quadruples), and captures most of the signal. This is the clearest evidence so
far that a "recurrent" model processing an outage set element-by-element, if it can represent something like
this order-truncated composition internally, has a real physical/mathematical reason to generalize from small
to large k on this benchmark — not a hope, a measured property of the LP's value function.

**Caveats:** single fixed control state, one topology, small manifest (25-100 sets depending on order), no
cluster bootstrap, no confirmatory block. This is a strong screen, not a confirmed claim.

## Partial-observation validation: does the Mobius result survive realistic communication uncertainty?

**This section has been corrected TWICE by external review**, each round independently verified against the
code/data before anything was accepted or changed. Round 1 found invalid registry YAML, an undisclosed
draws/metric deviation, an overstated "achievable gap" framing, an unfair neural comparator, an unmeasured
cost-savings claim, and a reproducibility bug (fixed in `scripts/p4_mobius_partialobs_v2.py`,
`registry/phase4_step_partialobs_v2.yaml`). Round 2 found that round 1's own "fix" for the reproducibility bug
was itself unstable (`hash()` of a Python string is process-randomised, so the "deterministic" seed changed
across runs — verified: three different values from three fresh `python3 -c` calls), that the reported
R-precision was computed by pooling all 20 operating points into one ranking instead of the project's
established per-operating-point convention, and that the linear-baseline and amortisation follow-ups had their
own bugs (below). All prior versions (`_v2` scripts/registries) are kept, not deleted.

**Setup, unchanged in substance across all three versions:** `y_pair` across all 1024 control vectors for the
156 needed sub-pairs x 20 ops (3,194,880 LP solves, 350s on 24 cores); `y_single` free from the existing dense
N-1 rows; composed with the exact communication-state posterior p(c|obs) (`src/fdna/v2.py`, unchanged),
K_DRAWS=100 stated explicitly. Final, corrected script: `scripts/p4_mobius_partialobs_v3.py` — a fixed integer
per cell (no `hash()` anywhere), and R-precision computed WITHIN each operating point then averaged across the
20 ops, with a proper paired cluster bootstrap (`src/fdna/rules.boot`, this project's standard confirmatory
statistic) on the per-op (g2 − g1) difference.

| comparator | P1 (v2b) MAE | P2 (v2c) MAE | P1 MSE | P2 MSE | P1 R-prec | P2 R-prec |
|---|---|---|---|---|---|---|
| g1-composed (naive sum + exact posterior) | 0.01557 | 0.01428 | 0.000819 | 0.000725 | 0.686 | 0.697 |
| prior-only (no observation, average over the prior) | 0.01558 | 0.01551 | 0.000774 | 0.000764 | 0.717 | 0.716 |
| **g2-composed (order-2 truncation + exact posterior)** | **0.00878** | **0.00693** | **0.000369** | **0.000264** | **0.873** | **0.902** |
| mean-composed reference (true table; optimal under MSE) | 0.00765 | 0.00569 | 0.000336 | 0.000229 | — | — |
| median-composed reference (true table; MAE-optimal) | 0.00592 | 0.00433 | — | — | — | — |

All values are the mean of a per-operating-point statistic over the 20 ops (not a pooled ranking).
**Paired cluster bootstrap on (g2 − g1), 20,000 resamples over operating points:**
R-precision improvement: **+0.188 (P1), 90% CI [0.172, 0.204]**; **+0.205 (P2), 90% CI [0.188, 0.222]**. MAE
improvement: **+0.0068 (P1), 90% CI [0.0064, 0.0072]**; **+0.0074 (P2), 90% CI [0.0069, 0.0078]**. The reported
p (≈5×10⁻⁵) is exactly the bootstrap's floor, 1/20,001 — verified: none of the 20,000 resamples fell at or
below zero, so this is "p below the floor this procedure can resolve," not a precisely measured tail
probability; it should not be quoted as a specific number. The effect sizes and intervals above, and the
following independently-verified consistency check, are the load-bearing evidence: **g2 beats both the
naive-sum and prior-only comparators on R-precision at every one of the 20 operating points, in both cells** —
the gain is not driven by one exceptional operating point.

**This still strengthens the existing exploratory screen; it is not independent confirmation.** The intervals
quantify variation across operating points conditional on the reused 60-triple manifest, not sampling from a
fresh one.

**Corrected reading of the gap-closure numbers.** Under MSE (where the mean-composed reference is genuinely
optimal): g2 closes ~93% of the naive-to-oracle gap in both cells. Under MAE against the metric-correct median
reference: ~70-74%. The **prior-only baseline is nearly as bad as g1** on MAE/MSE but is notably closer to g2 on
R-precision (0.717/0.716 vs g1's 0.686/0.697) — averaging over the prior alone recovers some ranking value even
with zero observation, but g2-with-observation still adds a further ~0.16-0.19 R-precision on top.

**Linear-composition baseline, corrected, now with MSE reported too.** `scripts/p4_linear_baseline_v2.py` fixes
two issues: (1) the original split rows by (op, triple) individually, letting a triple's OTHER operating points
leak into training; now uses `GroupKFold` grouped by the 60 distinct triples, so an entire triple is held out
together. (2) The original fit only squared error (OLS) but reported only MAE. Added an L1/MAE-trained
`QuantileRegressor` alongside OLS, and now reports held-out MSE for both, not only MAE.

| control state | g2 (fixed) MAE / MSE | OLS (learned, held-out) MAE / MSE | L1 (learned, held-out) MAE / MSE |
|---|---|---|---|
| full control | 0.000868 / 2.015e-5 | 0.001834 / 2.305e-5 | 0.000900 / 2.021e-5 |
| no control | 0.003745 / 1.158e-4 | 0.005731 / **9.503e-5 (−17.9%)** | 0.003786 / **1.054e-4 (−9.0%)** |

The coefficients quoted below are from an **in-sample fit on the full 60-triple dataset** (for interpretability
only); the MAE/MSE values above are the actual held-out, group-cross-validated performance and are the numbers
that matter for the comparison. At full control, an in-sample L1 fit gives coefficients of exactly [1.0, 1.0,
1.0, 1.0, 1.0, 1.0]; at no control the six coefficients are [1.0, 0.999, 1.001, 0.999, 1.0, 0.860] — one clear
outlier, not "every coefficient near 1" as an earlier draft of this section claimed (a real error, now
corrected: the earlier OLS coefficient of 0.6432 at no-control directly contradicted that claim).

**The held-out comparison is genuinely metric- and regime-dependent, not a clean win for g2 everywhere.** At
full control, g2 beats both learned models on both MAE and MSE. **At no control, learned OLS beats g2 on MSE by
17.9%, and the L1 model beats it by 9.0%** — a real, previously unreported case where a fitted model outperforms
the fixed-coefficient composition. g2 still wins on MAE in both regimes. The correct interpretation, following
the review: since the model's inputs already contain the engineered pairwise-interaction terms, this result
supports the usefulness of that interaction representation on this exploratory dataset — it is evidence for the
FEATURES (singleton + pairwise Möbius terms), not proof that g2's specific fixed-coefficient-of-1 form is
"independently validated as correct." That earlier phrasing overstated what a single in-sample coefficient fit
can show, and has been withdrawn.

**Amortisation/cost claim, corrected again.** `scripts/p4_amortization_test.py` had a genuine arithmetic error:
its `reuse_ratio_claim` field said "8,436 triples constructible from 477 pairs," which is wrong — a triple
needs its OWN 3 sub-pairs present, not merely membership in a pool of some pairs. Independently recomputed:
**477 cached pairs cover only 2,698 of the 8,436 possible triples** (verified exactly, matching the external
review's independent count); the full 703-pair set covers all 8,436, as it must by definition. The rest of the
finding is unaffected: at the tested 60-triple and 200-new-triple scales, direct labelling was cheaper than
extending the pair table (confirmed both times); the asymptotic full-closure argument (703 pairs → all 8,436
triples, ~12x cheaper than exhaustive direct labelling) remains valid as an exact count, not a measured
crossover — the crossover point itself is still not characterised.

**What remains, honestly, going into a confirmatory design:** still the same 60-triple/20-op manifest reused
across every test in this stage (not independent data — the paired bootstrap above quantifies sampling
uncertainty WITHIN this manifest, it does not substitute for a fresh one); N-4 under partial observation not
tested; the 200-triple extension (amortisation test) reused the same operating points and full control only,
so it strengthens unseen-triple evidence but does not independently confirm partial-observation performance;
MSE was registered alongside MAE from the start, but treating it as the PRIMARY endpoint is a post-hoc emphasis
that should be stated as an amendment in any confirmatory pre-registration, not adopted retroactively here; this
is not FDNA in any sense. The RNN/DeepSets question remains open until they receive matched pairwise information
and query budget (not yet done).

## Staged comparator set: does a trained recurrent/set model beat the closed-form composition?

**Correction (external review):** the comparison below gives g2/g3 STRICTLY MORE information than any trained
model tested — g2/g3 receive both singleton (N-1) AND pairwise-interaction (N-2) values; the "equal-information"
DeepSets/GRU variant received only the singleton value, never the pairwise interaction terms; the aggregate
tree received neither. This is a real information-budget asymmetry, not just a data-scale one, and the
conclusion below should be read as "generic learners given LESS structural information underperform," not as a
clean "recurrence loses to composition" result. A genuinely fair rerun (pairwise-aware DeepSets/GRU, matched
information) is listed as follow-up work, not yet done. The linear-baseline result above (same information as
g2, fitted vs fixed coefficients) is the properly matched comparison and is more trustworthy for the "does
recurrence/learning help" question than the table below.

Per the brief's own preferred direction (section 6.1/6.3), `scripts/p4_setmodels.py` trains a tuned tree
(permutation-invariant sum/max-aggregate electrical descriptors), a DeepSets model, and a GRU — all ONLY on N-1
(41 branches) + a sampled 300 N-2 pairs (`data_hik/train_n1n2.npz`, seed 100, frozen before generation, disjoint
from every eval manifest), evaluated ZERO-SHOT on the frozen N-3 and N-4 manifests. Two variants: static
electrical features only, then an equal-information variant also given the known N-1 singleton value per branch
(the same first-order information g1/g2/g3 use).

| model (equal-info variant) | N-3 MAE | N-3 rho | N-4 MAE | N-4 rho |
|---|---|---|---|---|
| g1 (naive sum, for reference) | 0.00800 | 0.679 | 0.01098 | 0.668 |
| tuned tree, aggregate descriptors | 0.01479 | 0.364 | 0.01822 | 0.391 |
| DeepSets | 0.01553 | 0.463 | 0.02326 | 0.563 |
| GRU (sorted order) | 0.01898 | 0.431 | 0.02483 | 0.609 |
| g2 / g3 (for reference, from above) | 0.00087 | 0.959 | 0.00126 | 0.982 |

**All three trained models perform WORSE than the trivial g1 naive sum**, and dramatically worse than g2/g3.
This holds even in the equal-information variant (branch-level known N-1 values as an explicit feature) — the
models are not merely missing information, they are failing to reliably learn even the additive structure from
only 20 distinct training operating points. This is best read as a **data-scale limitation of these small
models at this label budget**, not evidence against recurrence or set-processing in principle (per the brief's
own gating language, section 6.5: "failure of a generic recurrent model does not logically rule out a
structured recurrent model"). The clean, decisive finding at THIS budget is that **the zero-training closed-form
composition (g2/g3) is both cheaper and far more accurate than any trained comparator tested**, at every k
tested (3 and 4).

**GRU permutation sensitivity** (brief 6.5, required check): mean |prediction difference| between 5 random
input orderings and the canonical sorted order was 0.0054-0.0093 (small relative to the MAE values above, but
nonzero — the GRU is not exactly order-invariant, as expected for a plain GRU with no invariance-enforcing
mechanism). Canonical sorting was NOT treated as proof of invariance, per the brief's explicit warning; this
sensitivity is reported, not assumed away.

**What would change this conclusion:** more training operating points (the frozen k=3/k=4 manifests reuse the
same 20 ops as training deliberately, so this budget is small by design — a larger training pool, held
genuinely disjoint from any eval manifest, is the natural next step before concluding recurrence "does not
help" more broadly).

## Repaired-FDNA inference-only screen, cell P1 complete (registry/phase3_step2.yaml correction, brief 4.3)
Single GPU process (not the earlier contended two-process run that stalled). 3 reps, N_INF=25000, cell P1 (v2b,
q=0.7 s=0.3). NLL/KL only — no R-precision claim (per the earlier research-scientist power analysis, that
endpoint cannot resolve an effect this small at this scale; NLL on thousands of rows can).

| arm | NLL (mean +- std) | KL to exact posterior (mean +- std) |
|---|---|---|
| I8_bayes (exact structure, learnable priors — ceiling) | 3.2310 +- 0.0001 | 0.0067 +- 0.0001 |
| L_CC (plain logic, common-cause prior) | 3.3198 +- 0.0055 | 0.0985 +- 0.0059 |
| L_indep (plain logic, independent prior) | 3.3228 +- 0.0040 | 0.1025 +- 0.0045 |
| I1_mlp (generic MLP) | 3.3985 +- 0.0027 | 0.1754 +- 0.0012 |
| **J_OR_hard_CC (combined repaired FDNA: OR-aware + hard-min + joint head + CC prior)** | **3.3958 +- 0.0076** | **0.1742 +- 0.0079** |
| J_plain_CC (joint head, CC prior, flat FDNA mask) | 3.8199 +- 0.0037 | 0.5995 +- 0.0034 |
| J_plain (joint head, independent prior, flat FDNA mask) | 3.8646 +- 0.0041 | 0.6452 +- 0.0039 |
| I5_OR_hard (mean-field, OR-aware, hard-min) | 4.5479 +- 0.0130 | 1.3263 +- 0.0114 |
| I5_tau0 (mean-field, flat, hard-min) | 4.5845 +- 0.0235 | 1.3634 +- 0.0250 |
| I5_OR (mean-field, OR-aware, tau 0.05) | 4.7756 +- 0.0035 | 1.5539 +- 0.0040 |
| I5_ref (mean-field, flat, tau 0.05 — the ORIGINAL unrepaired arm) | 4.7941 +- 0.0070 | 1.5730 +- 0.0070 |

**Reading this against `registry/phase3_step2.yaml`'s own pre-stated attribution rule** ("any gain is attributed
to FDNA's strength/criticality form only if J_OR_hard_CC beats BOTH L_CC and I2/I8"): **it does not.** The
combined, fully repaired FDNA arm (J_OR_hard_CC) essentially ties the generic MLP (I1_mlp) and is measurably
WORSE than both plain-logic comparators (L_indep, L_CC) — a ~0.07 KL gap against a ~0.005 standard deviation,
i.e. not a noise artifact. The repairs (OR-awareness, hard-min, joint/common-cause head) DID close most of the
gap to the exact-structure ceiling (I8_bayes) relative to the original flat mean-field arm (KL 0.174 vs 1.573,
a real and large improvement) — but that improvement is fully explained by fixing the joint-head/common-cause
representation (J_plain_CC alone already gets to KL 0.60, and L_CC with NO FDNA algebra at all gets to 0.099).
**FDNA's specific strength/criticality parametrisation adds nothing measurable beyond ordinary dependency logic,
even after every representational defect flagged by the external review was fixed.** This directly and cleanly
answers the brief's question 4 ("does FDNA add anything beyond recurrence, dependency logic and physical
features?") for the inference-quality endpoint: no.

**Cell P2 (v2c, q=0.3 s=0.2) replicates this exactly.** J_OR_hard_CC: NLL 3.211+-0.010, KL 0.169+-0.010, vs
L_CC 0.087+-0.012, L_indep 0.097+-0.010, I1_mlp 0.092+-0.002, I8_bayes (ceiling) 0.0005+-0.0001. Same ordering,
same conclusion, in both cells: the repaired FDNA arm ties the generic MLP and is clearly worse than plain
logic with an identical head. This is now a stable, well-powered, twice-replicated negative finding, consistent
with every FDNA result across Phase 1-4.

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

## CONFIRMATORY RESULT: fresh operating points, fresh outage sets (registry/phase4_mobius_confirm.yaml)

Per the pre-registered protocol (block opened via `src/fdna/blocks.open_block`, reserve block ids 600-659, 70
fresh outage-set triples sampled with `block_seed('reserve')`=613, none of this data used in any earlier
Phase 3/4 screen, locks `reserve__MOBIUS_CONF_P1.lock` / `reserve__MOBIUS_CONF_P2.lock` committed with output
sha256 `80308ac7...`). 18,001,920 LP solves (32.5 min on 24 cores, 0 infeasible) generated N-1 (all 41 branches),
N-2 (182 needed sub-pairs), and N-3 (the 70 triples) labels across the full 1024-control grid for 60 fresh
operating points, `scripts/hik_label_confirm.py` / `scripts/p4_mobius_confirm.py`.

| | P1 (v2b) | P2 (v2c) |
|---|---|---|
| R-precision: g1 / prior-only / **g2** | 0.624 / 0.734 / **0.875** | 0.642 / 0.735 / **0.912** |
| Paired bootstrap, g2 − g1 (90% CI) | **+0.250 [0.237, 0.264]** | **+0.270 [0.255, 0.285]** |
| Paired bootstrap, g2 − prior-only (90% CI) | +0.141 [0.134, 0.148] | +0.177 [0.168, 0.185] |
| g2 beats g1 / prior-only, ops out of 60 | 60/60 | 60/60 |
| **Tier reached (pre-registered: tier1 ≥0.05)** | **tier1** | **tier1** |
| MAE: g1 / prior-only / g2 | 0.0160 / 0.0155 / 0.0087 | 0.0147 / 0.0155 / 0.0067 |
| MSE: g1 / prior-only / g2 | 0.000948 / 0.000872 / 0.000409 | 0.000864 / 0.000876 / 0.000293 |

**Both pre-registered primary claims (g2 vs g1, g2 vs prior-only, intersection-union across both cells) reach
tier1 — the strongest pre-declared tier, at an effect size roughly 5x the tier1 threshold.** This is on data no
earlier Phase 3/4 test ever touched: fresh operating points, a freshly-sampled outage-set manifest, frozen
before any label was generated. This is the first genuinely confirmatory (not exploratory-screen) positive
result in the whole Phase 1-4 programme.

**Exact-control linear-composition comparison, replicated at 3.5x the exploratory sample (70 vs 60 triples, 60
vs 20 ops).** At full control, g2 beats both OLS and the MAE-trained linear model on MAE and MSE. At no control,
g2 beats OLS on BOTH MAE and MSE this time (unlike the exploratory screen, where OLS beat g2's MSE by 17.9%);
the L1 model edges g2 on MSE only slightly (7.33e-5 vs 7.69e-5, ~4.8%, far smaller than the exploratory
screen's 9%). **The exploratory screen's "learning helps under no control" nuance mostly did not replicate on
independent data** — read honestly, that specific finding looks like it was partly a small-sample artifact of
the 60-triple/20-op exploratory manifest, not a robust regime-dependent effect. This is reported as a genuine
correction from replication, exactly what a confirmatory design is supposed to catch.

**What this does and doesn't establish:** this confirms the g2 composition beats the naive and prior-only
baselines on R-precision, with a large, tightly-bounded effect, on fresh IEEE-30 data at N-3 with exact control
of the electrical outage set and partial observation of the communication state. It does NOT establish: N-4
generalisation under partial observation (untested here); performance on a second topology; a fair RNN/DeepSets
comparison with matched pairwise information (still open); or that this is FDNA (it is not — a generic property
of the LP's value function, unchanged from every earlier caveat in this document).
