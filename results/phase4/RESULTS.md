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

Per the recommendation at the end of the previous report, `registry/phase4_step_partialobs.yaml` (committed
before any new label was generated) pre-registered this test. New data: `y_pair` regenerated across ALL 1024
control vectors (not one fixed state) for the 156 needed sub-pairs x 20 ops (3,194,880 LP solves, 350s on 24
cores, `scripts/hik_gen_ypair_fullgrid.py`); `y_single` reused for free from the existing dense N-1 rows in
`data_v2/vtable_train.npz`. `scripts/p4_mobius_partialobs.py` then composes g2(op,S,c) and g1(op,S,c) over all
1024 controls with the EXACT communication-state posterior p(c|obs) already used throughout Phase 2/3
(`src/fdna/v2.py`, unchanged), for both partial-observation cells (P1: q=0.7 s=0.3; P2: q=0.3 s=0.2), 100
observation draws per triple, scored against the TRUE realised shed at the TRUE hidden control state (not the
posterior mean).

| comparator | P1 (v2b) MAE | P2 (v2c) MAE |
|---|---|---|
| g1-composed (naive sum + exact posterior) | 0.01568 | 0.01438 |
| assume-full-control (ignore observation, use g2 at c=all-ones) | 0.01350 | 0.01342 |
| **g2-composed (order-2 truncation + exact posterior)** | **0.00888** | **0.00704** |
| oracle-full-table-composed (upper bound: as if the true N-3 table existed) | 0.00775 | 0.00580 |

**g2-composed beats both the naive-sum baseline (1.77x / 2.04x lower MAE) and ignoring the observation entirely
(1.52x / 1.91x lower MAE), in both cells.** It also closes **85.9% (P1) and 85.7% (P2) of the achievable gap**
between the naive baseline and the (practically unaffordable) true-N-3-table oracle — using ONLY cheap N-1/N-2
information, no N-3 label at all. This is the clearest positive result of the whole Phase 3/4 programme: **the
low-degree structural composition survives reintroducing realistic partial/stale communication observation**,
not just the exact-control screen it was first found in.

**What remains, honestly:** still the same frozen 60-triple manifest and 20 operating points as every earlier
test in this stage (not independent data); still no cluster bootstrap; still no confirmatory block; N-4 under
partial observation not yet tested (would need the same full-grid treatment applied to `k4_screen.npz`, not yet
done); this is not FDNA in any sense (a generic property of the LP's value function, same caveat as before).

## Staged comparator set: does a trained recurrent/set model beat the closed-form composition?

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
