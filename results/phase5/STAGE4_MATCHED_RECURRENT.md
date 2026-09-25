# Phase 5 Stage 4 — matched recurrent/learned-correction experiment

**This document was corrected after an external evidence review found four real bugs.** Every one was
independently reproduced against this repo's own committed code before being accepted, then fixed in
`scripts/p5_matched_recurrent_experiment_v2.py` / `p5_matched_recurrent_eval_v2.py`. The original (buggy)
scripts and their output are kept, not deleted.

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

## Corrected result: g2_fixed still wins, but by a much smaller and more honest margin

| arm | P1 R-precision | P1 vs g2_fixed (90% CI) | P2 R-precision | P2 vs g2_fixed (90% CI) |
|---|---|---|---|---|
| **g2_fixed** | **0.892** | -- | **0.922** | -- |
| g2_clipped | 0.892 | 0.000 (exact tie) | 0.922 | 0.000 (exact tie) |
| gbm (tuned) | 0.890 | -0.003 [-0.005, -0.001] | 0.919 | -0.004 [-0.006, -0.001] |
| **deepsets (now genuinely invariant)** | **0.881** | **-0.011 [-0.014, -0.009]** | **0.915** | **-0.007 [-0.008, -0.006]** |
| ridge | 0.866 | -0.026 [-0.031, -0.021] | 0.906 | -0.016 [-0.019, -0.013] |
| residual (g2 + learned correction) | 0.838 | -0.054 [-0.059, -0.050] | 0.870 | -0.052 [-0.057, -0.047] |

**Corrected reading.** g2_fixed still beats every trained arm in both cells, and every 90% CI on
(learned-g2_fixed) still lies below zero -- **that headline conclusion is unchanged.** What changed
substantially: the properly-invariant DeepSets model's gap to g2_fixed shrank from -0.113/-0.122 (the
non-invariant, buggy version) to **-0.011/-0.007** -- now the SECOND-closest arm to g2_fixed, essentially on par
with GBM, not "worst of all five learned arms by a wide margin" as the original (bugged) report claimed. The
original finding that the commutative structured model performed worst was substantially an artifact of the
architecture bug, not a genuine property of set-based/commutative corrections. The residual-correction network
remains the clear loser in both versions.

**Per the pre-registered decision rule** (promote a learned arm only if it beats g2_fixed by a material,
bootstrap-supported margin in both cells): still no arm is promoted -- g2_fixed remains the lead method, now on
materially more trustworthy evidence.

## Honest caveats, updated
- The pre-rejection checklist item `deepsets_loss_decreased` (comparing the first vs. last tracked validation
  loss) reads `False` even though the model's actual BEST-checkpoint validation MSE (4.5e-6) was the lowest of
  any trained arm (ridge 1.26e-5, gbm 1.33e-5) -- the checklist's first-vs-last comparison is a weak proxy for
  "did the model ever improve," not a correctness bug, but worth flagging as an imprecise diagnostic rather than
  over- or under-reading it.
- 35 training outages (down from 45, to make room for a genuinely disjoint validation split) is a smaller
  charged-label budget than before; a larger budget was not attempted given the time already spent on repairs.
- **Correction (2026-09-25, independent second-round review):** the line that used to appear here claiming the
  edge-pooled architecture is "exactly invariant" was wrong. The edge encoder itself IS invariant, but the
  readout concatenates the raw, unpermuted risk features (`X[:,6:]`) after pooling in fixed slot order, so the
  full model is NOT genuinely permutation-invariant; the dedicated test used equal risk values and could not
  detect this. A repair (edge-token risk encoding + a shared scaler across risk slots) is in progress -- see
  `results/phase4/CLAIM_LEDGER.md` row 12 for status; the table above will be superseded once that fix is
  retrained.
- **Scope caveat (unchanged, still accurate, restated for clarity):** `deepsets` here is a set model (pooled,
  order-independent by design intent), not a recurrent network. No actual RNN/GRU comparator for N-1->N-k
  transfer has been trained or evaluated anywhere in Phase 4/5 -- the earlier Phase 4 GRU/DeepSets screen
  (`results/phase4/RESULTS.md`) used a different, smaller-budget setup and is not superseded by this experiment.
  A DeepSets result, buggy or fixed, is not evidence for or against recurrent architectures specifically; that
  question remains open and untested, not decided in either direction.
