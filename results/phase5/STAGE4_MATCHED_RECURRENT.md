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
- **Residual-model diagnostic protocol added (repair round 2, per the review's request):** `ResidualNet`'s final
  layer is now zero-initialized with a separate free scalar `lam` (`prediction = g2 + lam*r_theta(X)`), so the
  zero-correction point is a real, reachable step-(-1) state, not an accident of checkpoint selection never
  being evaluated before the first optimizer step; `g2_fixed`'s row above IS that zero-correction baseline.
  Checkpoint selection now tracks BOTH value-MSE and posterior-composed R-precision on a small fixed validation
  subset (ties broken by value-MSE), selecting by R-precision. Three predeclared seeds (0,1,2) all converged to
  the SAME diagnostic R-precision (0.8691) at different epochs (4, 54, 69) with different final `lam` (0.57,
  0.44, 0.45) -- a striking, consistent result suggesting the correction's effect on ranking saturates quickly
  regardless of how long training continues, not seed noise. A tiny-batch overfit check (12 rows, one per
  distinct outage/op block with the largest single-row residual, chosen for genuine diversity after an earlier
  version of this same check accidentally selected 12 near-duplicate rows from one block and passed vacuously)
  confirms the network CAN memorize real signal when given a fair test (final MSE 4.9e-10 vs target variance
  7.9e-6) -- ruling out an optimization/objective bug as the explanation for the residual arm's poor test
  performance. Taken together: the residual arm's failure looks like a genuine absence of transferable signal in
  the residual target at this label budget, not an artifact of how it was trained or selected.
- **Scope caveat (unchanged, still accurate, restated for clarity):** `deepsets` here is a set model (pooled,
  order-independent by design intent), not a recurrent network. No actual RNN/GRU comparator for N-1->N-k
  transfer has been trained or evaluated anywhere in Phase 4/5 -- the earlier Phase 4 GRU/DeepSets screen
  (`results/phase4/RESULTS.md`) used a different, smaller-budget setup and is not superseded by this experiment.
  A DeepSets result, buggy or fixed, is not evidence for or against recurrent architectures specifically; that
  question remains open and untested, not decided in either direction.
