# Phase 5 Stage 4 — matched recurrent/learned-correction experiment

**This document was corrected twice, after two independent evidence reviews of successive repair attempts.**
Every finding from both reviews was independently reproduced against this repo's own committed code before
being accepted, then fixed in `scripts/p5_matched_recurrent_experiment_v2.py` / `p5_matched_recurrent_eval_v2.py`
and `src/fdna/nn/pairset.py` / `pairset_features.py`. The original (buggy) scripts and their outputs are kept,
not deleted, at every stage.

## Identifiability check (unchanged, still respected)
g2 is exact for any set of size <=2 by construction, so the residual-correction arm is trained ONLY on charged
N-3 labels, never on N-1/N-2 rows where the residual is trivially zero.

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
  absorb any rescaling). Selected shrink = 0.25. On test: statistically indistinguishable from `g2_fixed` at P1
  (mean diff +0.00013, CI includes zero, p=0.34) and a tiny but technically significant improvement at P2
  (+0.00029, CI excludes zero, p=0.048) -- practically negligible either way, well below any reasonable
  promotion bar, but a properly-separated estimate rather than reading a jointly-trained scalar as evidence. A
  tiny-batch overfit check (12 rows, one per distinct outage/op block with the largest single-row residual,
  chosen for genuine diversity after an earlier version of this same check accidentally selected 12 near-
  duplicate rows from one block and passed vacuously) confirms the network CAN memorize real signal when given a
  fair test (final MSE 4.9e-10 vs target variance 7.9e-6) -- ruling out an optimization/objective bug. Taken
  together: the residual arm's failure looks like a genuine absence of transferable signal in the residual
  target at this label budget, now on considerably more rigorous diagnostic footing than round 2's coarse,
  P1-only, pooled version.

## Repair round 3: the DeepSets ablation -- isolating architecture from the other confounds
The review pointed out that round 2's readout fix changed representation, normalization, AND optimization
simultaneously, so the score change (gap widened, see above) couldn't be attributed to any one of them --
and that the earlier "leaking information" framing overclaimed the mechanism (the pair-risk features are
legitimate inputs available to every arm; the old model violated its own claimed invariance property, it did
not access forbidden information). To separate architecture from the rest: `OrderedMLP` (a plain MLP directly on
the same 15-column raw row, deliberately NOT invariant -- matched rough parameter count, 9,409 vs. DeepSets'
9,089) and `PermAveragedModel` (wraps ANY trained model, averaging its output over all 6 relabelings --
EXACTLY invariant by construction, 6 forward passes charged, regardless of whether the base model itself is
invariant), both trained/evaluated across the same 3 predeclared seeds as everything else this round.

| arm | P1 mean (std across seeds) | P1 vs g2_fixed | P2 mean (std) | P2 vs g2_fixed |
|---|---|---|---|---|
| **g2_fixed** | **0.8921** | -- | **0.9221** | -- |
| gbm | 0.8895 | -0.0026 | 0.9185 | -0.0036 |
| **ordered (raw, NOT invariant)** | **0.8796 (0.0030)** | -0.009 to -0.016, all sig. | **0.9097 (0.0053)** | -0.008 to -0.020, all sig. |
| **ordered, permutation-averaged (invariant)** | **0.8883 (0.0023)** | -0.001 to -0.006 (1/3 seeds n.s.) | **0.9165 (0.0045)** | 0.0000 to -0.011 (1/3 seeds n.s.) |
| **deepsets (pooling architecture, invariant)** | **0.8535** | **-0.039, sig.** | **0.8840** | **-0.038, sig.** |
| residual (best of 2 nonzero-correction seeds) | 0.8380 | -0.054, sig. | 0.8736 | -0.049, sig. |

**A clear, substantive finding.** The RAW ordered model -- not even invariant, no architectural constraint at
all -- already beats DeepSets by a wide margin (0.880 vs 0.854 at P1, 0.910 vs 0.884 at P2). Forcing invariance
onto it via POST-HOC PERMUTATION AVERAGING (not architectural pooling) closes MOST of the remaining gap to
g2_fixed, and for one of the three seeds in EACH cell, the permutation-averaged model is statistically
INDISTINGUISHABLE from g2_fixed (90% CI includes zero: P1 seed 0, p=0.85; P2 seed 0, p=0.49) -- something no
version of DeepSets has come close to in any round. This points specifically at the POOLING ARCHITECTURE, not
the invariance requirement itself, as the more likely explanation for DeepSets' underperformance: a model can be
made genuinely, exactly invariant (permutation-averaging is invariant by construction, unconditionally) and
still perform much closer to g2_fixed than the pooling-based approach does. This does not prove pooling is
inherently worse in general, and it does not establish whether the ordered model's own (non-invariant)
performance reflects a real, transferable use of branch-position information or an artifact specific to this
one fixed topology's branch numbering (out of scope, unchanged) -- but within this experiment, holding data,
features, split, and seeds fixed, pooling is the more likely culprit, not the earlier round's vaguer "leaking
information" story.
- **Scope caveat (unchanged, still accurate, restated for clarity):** `deepsets` here is a set model (pooled,
  order-independent by design intent), not a recurrent network. No actual RNN/GRU comparator for N-1->N-k
  transfer has been trained or evaluated anywhere in Phase 4/5 -- the earlier Phase 4 GRU/DeepSets screen
  (`results/phase4/RESULTS.md`) used a different, smaller-budget setup and is not superseded by this experiment.
  A DeepSets result, buggy or fixed, is not evidence for or against recurrent architectures specifically; that
  question remains open and untested, not decided in either direction.
