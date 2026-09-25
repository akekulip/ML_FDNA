# Phase 5 Stage 4 — matched recurrent/learned-correction experiment

**This document was corrected twice, after two independent evidence reviews of successive repair attempts.**
Every finding from both reviews was independently reproduced against this repo's own committed code before
being accepted, then fixed in `scripts/p5_matched_recurrent_experiment_v2.py` / `p5_matched_recurrent_eval_v2.py`
and `src/fdna/nn/pairset.py` / `pairset_features.py`. The original (buggy) scripts and their outputs are kept,
not deleted, at every stage.

## Identifiability check (unchanged, still respected)
g2 is exact for any set of size <=2 by construction, so the residual-correction arm is trained ONLY on charged
N-3 labels, never on N-1/N-2 rows where the residual is trivially zero.

## Superseding current result: all declared DeepSets seeds evaluated on TEST

**Status (2026-09-25):** `results/phase5/matched_recurrent_eval_v2.json` now supersedes the earlier seed-0-only
DeepSets TEST rows below. The evaluation uses the existing checkpoints only -- no retraining, no fresh holdout
-- over the same 60 operating points x 25 held-out TEST outages x 50 posterior draws in both P1/P2 cells. It
adds canonical `deepsets_seed0`, `deepsets_seed1`, and `deepsets_seed2` arms while preserving legacy `deepsets`
as the seed-0 alias. The JSON contains per-op R-precision rows, per-seed bootstrap CIs, seed-family mean/std,
and paired matrix bootstraps over operating points x seeds against `g2_fixed`, `gbm`, and
`ordered_perm_avg`.

| arm / family | P1 R-precision | P1 vs g2_fixed (90% CI) | P2 R-precision | P2 vs g2_fixed (90% CI) |
|---|---:|---:|---:|---:|
| **g2_fixed** | **0.8921** | -- | **0.9221** | -- |
| gbm | 0.8895 | -0.0026 [-0.0046, -0.0006] | 0.9185 | -0.0036 [-0.0059, -0.0013] |
| ridge | 0.8663 | -0.0259 [-0.0310, -0.0209] | 0.9065 | -0.0156 [-0.0186, -0.0126] |
| **DeepSets mean across seeds 0/1/2** | **0.8606 (std 0.0196)** | **-0.0315 [-0.0495, -0.0139]** | **0.8901 (std 0.0208)** | **-0.0320 [-0.0514, -0.0134]** |
| ordered mean across seeds 0/1/2 | 0.8796 (std 0.0030) | -0.0125 [-0.0162, -0.0092] | 0.9097 (std 0.0053) | -0.0123 [-0.0181, -0.0075] |
| ordered, permutation-averaged mean across seeds 0/1/2 | 0.8883 (std 0.0023) | -0.0038 [-0.0062, -0.0014] | 0.9165 (std 0.0045) | -0.0056 [-0.0101, -0.0011] |
| residual seed 0 | 0.8380 | -0.0541 [-0.0594, -0.0490] | 0.8736 | -0.0485 [-0.0541, -0.0430] |
| residual seed 1 (zero-correction checkpoint) | 0.8921 | 0.0000 [0.0000, 0.0000] | 0.9221 | 0.0000 [0.0000, 0.0000] |
| residual seed 2 | 0.8359 | -0.0562 [-0.0611, -0.0514] | 0.8658 | -0.0563 [-0.0611, -0.0515] |
| residual frozen-shrunk | 0.8922 | +0.0001 [-0.0004, +0.0006] | 0.9224 | +0.0003 [+0.0000, +0.0006] |

DeepSets per-seed TEST rows:

| arm | P1 R-precision | P1 vs g2_fixed (90% CI) | P2 R-precision | P2 vs g2_fixed (90% CI) |
|---|---:|---:|---:|---:|
| deepsets_seed0 / legacy `deepsets` | 0.8535 | -0.0386 [-0.0439, -0.0333] | 0.8840 | -0.0380 [-0.0436, -0.0324] |
| deepsets_seed1 | 0.8874 | -0.0047 [-0.0070, -0.0024] | 0.9180 | -0.0041 [-0.0061, -0.0020] |
| deepsets_seed2 | 0.8409 | -0.0512 [-0.0567, -0.0457] | 0.8682 | -0.0539 [-0.0598, -0.0480] |

Seed-family paired matrix bootstraps (operating points x seeds) show the DeepSets family below all three
comparators: P1 vs GBM = -0.0289 [-0.0472, -0.0110], vs ordered permutation-average = -0.0277 [-0.0458,
-0.0098]; P2 vs GBM = -0.0285 [-0.0478, -0.0097], vs ordered permutation-average = -0.0264 [-0.0471,
-0.0059]. This is a matched-condition comparison under the same data, features, split, and seeds; it does not
identify pooling as a causal mechanism, and it does not establish whether ordered branch-position information
would transfer to other topologies or branch renumberings.

Provenance is embedded in `results/phase5/matched_recurrent_eval_v2.json`: HEAD
`a7ec4d0a6adade844dd9d553b9d749deca10bf3b`, torch threads = 4, SHA256 for all used checkpoints/input caches,
the training config/scaler JSON, and the relevant source files. The final artifact is not from a later clean
fresh rerun: a full all-seed scoring run completed both cells, then failed only while serializing old NumPy
coverage provenance. Those completed cell results were recovered, the scope text and provenance were regenerated
with JSON-safe coverage, and the final JSON was written by atomic replace after
`json.dumps(..., allow_nan=False)` completed and strict JSON validation passed.

**Hardware validation update.** The evaluator now supports explicit `--device cpu|cuda` and `--output` options.
After that source change, the default CPU artifact was rerun end-to-end and a separate CUDA artifact was run on
the host GPU (`results/phase5/matched_recurrent_eval_v2_cuda.json`). The comparison artifact
`results/phase5/matched_recurrent_eval_v2_device_compare.json` records source/input/output hashes, versions,
hardware metadata, and elapsed times. CPU and CUDA results match exactly across all compared cell result fields:
maximum absolute difference = 0.0. CPU elapsed = 690.0s; CUDA elapsed = 547.1s. The CUDA provenance reports
`NVIDIA GeForce RTX 2070`, driver `535.183.01`, 8192 MiB, torch `2.5.1+cu121`.

## The four bugs, each confirmed by reproduction before being fixed

1. **Validation target misalignment (critical, external review finding 1).** The original script sampled
   `Xva`/`yva` with one RNG draw, then advanced the SAME RNG to sample `g2va` at DIFFERENT control indices, so
   `rva=yva-g2va` compared mismatched states. Reproduced exactly: 4,796/4,800 mismatched indices; the reported
   `7.474e-4` residual validation MSE was not a meaningful model-quality number. Fixed: `(key, X, y, g2)` are
   now built together in one pass for both training and validation; `r+g2==y` is asserted by construction.
2. **Validation overlapped training (external review finding 2).** Validation reused the first 10 TRAINING
   outages and the first 15 (of 60) TRAINING operating points. Reproduced: 612/4,800 validation rows also
   present in training; the remainder still shared both axes with training (interpolation, not held-out
   generalisation). Fixed: outages AND operating points are now split into three genuinely DISJOINT groups
   (35 train / 10 val outages; 45 train / 8 val ops), verified disjoint by direct set intersection. The final
   test still uses all 60 ops against 25 held-out outages -- explicitly labelled as the known-operating-point /
   new-outage-combination axis, per the review's own guidance that this is a legitimate experiment if stated
   precisely.
3. **The "commutative" model was not actually permutation-invariant (external review finding 3).** The original
   tokens `(ya,Iab),(yb,Iac),(yc,Ibc)` paired a singleton with the WRONG interaction term under relabeling.
   Reproduced exactly: relabeling a<->b changed the forward output from 0.10 to 0.11 with a valid weight
   assignment executed against the committed `forward` method. Fixed (`src/fdna/nn/pairset.py`): each of the 3
   EDGES `{a,b},{a,c},{b,c}` is now encoded as `(min(y_i,y_j), max(y_i,y_j), I_ij)` -- symmetric in its own two
   endpoints -- pooled by sum over the 3 edges, which are the same 3 edges regardless of relabeling. Verified
   genuinely invariant on the ACTUAL TRAINED model across all 6 relabelings (not just architecturally), both in
   a dedicated pytest (`tests/test_deepsets_invariance.py`) and inline in the training script.
4. **Unscaled feature.** The reciprocal-determinant (LODF risk) feature was capped at 10,000 next to features
   roughly in [0,1]. Fixed: log1p-transformed, then standardised using TRAINING-set statistics only.

## A second, independent review found bug #3's fix was still incomplete
The edge/interaction pairing itself was genuinely fixed, but the readout concatenated the raw, per-edge
`risk_ab/risk_ac/risk_bc` features AFTER pooling, in a fixed slot order -- a genuine unordered-edge quantity
handled asymmetrically, one layer downstream of the bug already fixed. The dedicated test used EQUAL risk values
(1.0,1.0,1.0) and only permuted columns 0-5, so it was structurally blind to this: permuting three identical
numbers changes nothing. Confirmed by reading the code directly (no witness execution needed -- the defect is
visible in the tensor wiring). **Fixed** (`src/fdna/nn/pairset.py`, `src/fdna/nn/pairset_features.py`):
`risk_ij` now travels INSIDE its own edge token as a 4th component (`edge_enc` input dim 3->4, `readout` tail
reduced from `[F,risk_ab,risk_ac,risk_bc,c1..c5]` to just `[F,c1..c5]`, both genuinely label-independent);
preprocessing now fits ONE shared scaler across all 3 risk slots (`fit_shared_risk_scaler`), not 3 separate
per-slot ones -- a fixed architecture alone does not fix a per-slot-dependent normalisation. Verified genuinely
invariant end-to-end (raw features -> scaler -> model, UNEQUAL risk values, all 6 relabelings generated
programmatically via `pairset_features.column_perm_for_vertex_perm`, not hand-written tuples) on the actual
RETRAINED checkpoint: max spread 1.49e-8 against a tolerance of 1e-6 calibrated from measured fp32-vs-fp64
rounding on the same input (`tests/test_deepsets_invariance.py`, plus an inline full-pipeline check in the
training script).

## Corrected result (repair round 2): g2_fixed's lead is LARGER than round 1's partially-fixed model suggested

| arm | P1 R-precision | P1 vs g2_fixed (90% CI) | P2 R-precision | P2 vs g2_fixed (90% CI) |
|---|---|---|---|---|
| **g2_fixed** | **0.8921** | -- | **0.9221** | -- |
| g2_clipped | 0.8921 | 0.000 (exact tie) | 0.9221 | 0.000 (exact tie) |
| gbm (tuned) | 0.8895 | -0.0026 [-0.0046, -0.0006] | 0.9185 | -0.0036 [-0.0059, -0.0013] |
| ridge | 0.8663 | -0.0259 [-0.0310, -0.0209] | 0.9065 | -0.0156 [-0.0186, -0.0126] |
| **deepsets (now GENUINELY invariant, full pipeline)** | **0.8535** | **-0.0386 [-0.0439, -0.0333]** | **0.8840** | **-0.0380 [-0.0436, -0.0324]** |
| residual (g2 + learned correction, zero-init + lambda) | 0.8380 | -0.0541 [-0.0594, -0.0490] | 0.8736 | -0.0485 [-0.0541, -0.0430] |

**Corrected reading, honestly.** g2_fixed still beats every trained arm in both cells (headline unchanged,
now on genuinely trustworthy evidence for the first time). But the number that changed is the OPPOSITE
direction from round 1's story: fixing the readout's remaining non-invariance did NOT shrink DeepSets' gap
further -- it WIDENED it, from round 1's partially-fixed **-0.011/-0.007** back out to **-0.039/-0.038**, now
clearly the SECOND-WORST arm (behind only the residual network), not "near-tied with GBM" as round 1 reported.
GBM is now unambiguously the second-closest arm to g2_fixed in both cells.

**Correction (2026-09-25, independent third-round review):** an earlier version of this paragraph attributed the
regression to the partially-fixed model "leaking non-invariant... information" -- language that reads as a claim
of forbidden or hidden information access, which is not what happened and is withdrawn. The pair-risk features
(`risk_ab/ac/bc`) are legitimate, deployable inputs, supplied to every other learned arm (ridge, gbm) as well --
there is nothing privileged or hidden about them. What the partially-fixed model actually did wrong was violate
its OWN claimed invariance property: it treated an unordered-edge quantity as if it were tied to a fixed,
arbitrary branch-numbering convention. Whether that ordering-dependence functioned as a genuinely useful (if
non-transferable) signal specific to this fixed topology's branch numbering, versus a pure architecture/
normalization/optimization confound from everything that changed between round 1 and round 2, is NOT yet
established by anything in this document -- three things changed at once (representation, normalization, and a
full retrain), and the evidence so far cannot attribute the score change to any one of them. See the ablation in
`results/phase5/STAGE4_MATCHED_RECURRENT.md`'s later section (repair round 3) for a bounded experiment designed
to separate these. The pattern across all three review rounds remains consistent: every time this model was
tested more rigorously, its measured performance either dropped or its apparent advantage was shown to be a bug
artifact -- there is no round in which the model's TRUE capability was underestimated by the tooling.

**Per the pre-registered decision rule** (promote a learned arm only if it beats g2_fixed by a material,
bootstrap-supported margin in both cells): still no arm is promoted -- g2_fixed remains the lead method, and
this repair round's evidence for that conclusion is now stronger, not weaker.

## Honest caveats, updated
- The pre-rejection checklist item `deepsets_loss_decreased` reads `True` in the repair-round-2 retrain (the
  earlier `False` reading, discussed in prior versions of this section, was specific to round 1's checkpoint and
  is superseded).
- 35 training outages (down from 45, to make room for a genuinely disjoint validation split) is a smaller
  charged-label budget than before; a larger budget was not attempted given the time already spent on repairs.
- **Residual-model diagnostic protocol (repair round 2, corrected round 3).** `ResidualNet`'s final layer is
  zero-initialized with a separate free scalar `lam` (`prediction = g2 + lam*r_theta(X)`), so the zero-correction
  point is a real, reachable step-(-1) state; `g2_fixed`'s row above IS that zero-correction baseline. Round 2's
  checkpoint-selection diagnostic pooled 12 randomly-sampled validation blocks into ONE ranking (P1 only) and
  reported all 3 seeds converging to the SAME 0.8691 -- an independent third review found this diagnostic too
  coarse (only 2 distinct achievable values across 17 checked epochs) and mismatched with final eval's own
  per-op-then-average protocol; "3 seeds converging" was 3 seeds landing on the better of 2 possible outcomes,
  not a striking result. **Fixed:** the diagnostic now uses the FULL VAL_OUTAGES x VAL_OPS cross-product (80
  pairs, not a 12-pair sample), ranked per-op then averaged, for BOTH cells. Corrected per-seed diagnostic
  scores are genuinely distinct: seed 0 = 0.8267, seed 1 = 0.8262, seed 2 = 0.8276 -- and **seed 1's selected
  checkpoint is epoch -1, the untrained zero-correction state itself**: training never found an epoch that beat
  the zero-correction baseline on validation for that seed, so the (corrected) selector correctly kept the
  starting point. On TEST, `residual_seed1`'s R-precision is therefore numerically IDENTICAL to `g2_fixed`
  (0.8921/0.9221) -- not a coincidence or a bug, a direct consequence of that seed's own selection.
  `residual`/`residual_seed2` (seeds 0/2, which DID find a nonzero correction) both score clearly BELOW
  `g2_fixed` on test (P1: 0.838/0.836; P2: 0.874/0.866), 90% CIs entirely negative. Also added:
  `residual_frozen_shrunk` (freeze the trained correction, sweep a prespecified `[0, 0.25, 0.5, 0.75, 1.0]`
  shrinkage grid on validation via the SAME per-op/dual-cell diagnostic, since `lam` trained jointly with the
  final layer is not independently identifiable as a validated shrinkage factor -- the layer's own weights can
  absorb any rescaling). Selected shrink = 0.25. On test: CI includes zero at P1
  (mean diff +0.00013, p=0.34) and a tiny but technically significant improvement at P2
  (+0.00029, CI excludes zero, p=0.048) -- practically negligible either way, well below any reasonable
  promotion bar, but a properly-separated estimate rather than reading a jointly-trained scalar as evidence. A
  tiny-batch overfit check (12 rows, one per distinct outage/op block with the largest single-row residual,
  chosen for genuine diversity after an earlier version of this same check accidentally selected 12 near-
  duplicate rows from one block and passed vacuously) confirms the network CAN memorize real signal when given a
  fair test (final MSE 4.9e-10 vs target variance 7.9e-6) -- ruling out an optimization/objective bug. Taken
  together: the residual arm's failure looks like a genuine absence of transferable signal in the residual
  target at this label budget, now on considerably more rigorous diagnostic footing than round 2's coarse,
  P1-only, pooled version.

## Historical repair round 3 ablation -- superseded for DeepSets by the all-seed table above
The review pointed out that round 2's readout fix changed representation, normalization, AND optimization
simultaneously, so the score change (gap widened, see above) couldn't be attributed to any one of them --
and that the earlier "leaking information" framing overclaimed the mechanism (the pair-risk features are
legitimate inputs available to every arm; the old model violated its own claimed invariance property, it did
not access forbidden information). The ablation compared `OrderedMLP` (a plain MLP directly on the same
15-column raw row, deliberately NOT invariant -- matched rough parameter count, 9,409 vs. DeepSets' 9,089) and
`PermAveragedModel` (wraps ANY trained model, averaging its output over all 6 relabelings -- EXACTLY invariant
by construction, 6 forward passes charged, regardless of whether the base model itself is invariant), both
trained/evaluated across the same 3 predeclared seeds. The table below is retained as history: ordered rows are
seed-family mean/std rows, while the DeepSets row is the legacy seed-0 alias only and is superseded by the
all-seed DeepSets section above.

| arm | P1 score | P1 vs g2_fixed | P2 score | P2 vs g2_fixed |
|---|---|---|---|---|
| **g2_fixed** | **0.8921** | -- | **0.9221** | -- |
| gbm | 0.8895 | -0.0026 | 0.9185 | -0.0036 |
| **ordered (raw, NOT invariant)** | **0.8796 (0.0030)** | -0.009 to -0.016, all sig. | **0.9097 (0.0053)** | -0.008 to -0.020, all sig. |
| **ordered, permutation-averaged (invariant)** | **0.8883 (0.0023)** | -0.001 to -0.006; one seed CI includes zero | **0.9165 (0.0045)** | 0.0000 to -0.011; one seed CI includes zero |
| **deepsets seed0 / legacy alias (pooling architecture, invariant)** | **0.8535** | **-0.039, sig.** | **0.8840** | **-0.038, sig.** |
| residual (best of 2 nonzero-correction seeds) | 0.8380 | -0.054, sig. | 0.8736 | -0.049, sig. |

**Matched-comparison reading.** The raw ordered family and the permutation-averaged ordered family both score
above the all-seed DeepSets family under the same data, features, split, and seed set, and the
permutation-averaged wrapper shows that exact invariance by prediction-time averaging can perform much closer to
g2_fixed than this pooled DeepSets architecture does in this experiment. This comparison does not identify
pooling as a causal mechanism, does not prove pooling is inherently worse in general, and does not establish
whether the ordered model's own non-invariant performance reflects transferable branch-position information or
an artifact specific to this fixed topology's branch numbering.
- **Scope caveat (unchanged, still accurate, restated for clarity):** `deepsets` here is a set model (pooled,
  order-independent by design intent), not a recurrent network. No actual RNN/GRU comparator for N-1->N-k
  transfer has been trained or evaluated in this matched Phase 5 experiment. The earlier Phase 4 GRU/DeepSets
  screen (`results/phase4/RESULTS.md`) used a different, smaller-budget setup and is not superseded by this
  experiment. A DeepSets result, buggy or fixed, is not evidence for or against recurrent architectures
  specifically in this matched setup; that question remains open here.
