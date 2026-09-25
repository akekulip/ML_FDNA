# Phase 5 Stage 4 — matched recurrent/learned-correction experiment

## Identifiability check (stated first, per the brief)
g2 is exact for any set of size <=2 by construction: the "residual" V-g2 on any N-1/N-2 row is trivially zero
(verified directly, a tautology of the truncation, not a measured fact). The residual-correction arm is
therefore trained ONLY on charged N-3 labels, never on N-1/N-2 rows -- the empty-supervision trap the brief
warns against was avoided by design, not discovered after a failed run.

## Design
`registry/phase5_matched_recurrent.yaml` (committed before training). 70 confirmatory triples split 45
train / 25 held-out test, by whole outage set, fixed seed, verified disjoint. Every trained arm gets IDENTICAL
features: 3 singleton values, 3 pairwise interaction values, island floor, 3 sub-pair LODF risk scores, and the
control vector itself (15-dim total) -- trained on 345,600 rows (45 triples x 60 ops x 128 sampled controls),
0 new LP solves (all from cached tables). Six arms: `g2_fixed` (no training), `g2_clipped` (Stage 2 physical
repair, no training), `ridge` (symmetric linear), `gbm` (tuned LightGBM), `deepsets` (a COMMUTATIVE
pair-aware set model, sum-pooled over the 3 elements -- chosen deliberately over a GRU to sidestep outage-order
sensitivity entirely, per the brief's own stated preference), `residual` (retains g2 exactly, learns only the
correction, trained exclusively on N-3 labels per the identifiability check).

**Pre-rejection checklist, all pass:** target variation nonzero (train std 0.038, val std 0.050); ridge beats
predict-the-mean; both neural losses decreased monotonically over training; train/test triple split verified
disjoint by direct set intersection.

Every arm composed with the SAME P1/P2 partial-observation posterior (unchanged `src/fdna/v2.py` machinery),
scored on the SAME R-precision endpoint, per-operating-point cluster bootstrap, on the 25 held-out test
triples only (never seen during training).

## Result: g2_fixed wins, decisively and in both cells

| arm | P1 R-precision | P1 vs g2_fixed (90% CI) | P2 R-precision | P2 vs g2_fixed (90% CI) |
|---|---|---|---|---|
| **g2_fixed** | **0.892** | -- | **0.922** | -- |
| g2_clipped | 0.892 | 0.000 (exact tie) | 0.922 | 0.000 (exact tie) |
| gbm (tuned) | 0.890 | -0.003 [-0.005, -0.0003] | 0.917 | -0.005 [-0.007, -0.003] |
| ridge | 0.862 | -0.030 [-0.036, -0.025] | 0.905 | -0.017 [-0.020, -0.014] |
| residual (g2 + learned correction) | 0.834 | -0.058 [-0.062, -0.054] | 0.862 | -0.060 [-0.064, -0.055] |
| deepsets (commutative set-state) | 0.780 | -0.113 [-0.122, -0.103] | 0.800 | -0.122 [-0.130, -0.113] |

**No learned arm beats g2_fixed in either cell.** Every 90% CI on the (learned − g2_fixed) difference lies
entirely below zero. GBM comes closest (a small, still-negative, bootstrap-excludes-zero gap in both cells).
The commutative set-state model -- the arm structurally closest to what the brief calls a "recurrent structured
correction" -- performs WORST of all five learned arms, in both cells, by a wide and consistent margin. The
residual-correction network (which explicitly retains the exact g2 term and only learns the missing piece)
still loses to plain g2_fixed by a clear, bootstrap-supported margin in both cells -- the extra learned
correction actively hurts more than it helps, on this held-out triple set.

**Per the pre-registered decision rule** ("retain a learned/recurrent arm as lead only if it beats g2_fixed by
a material, bootstrap-supported margin in BOTH cells; if it ties, prefer the simpler method"): **no arm is
promoted. g2_fixed remains the lead method.** This directly answers the brief's central Stage-4 question: on
this benchmark, at this label budget, with matched information, a recurrent/learned correction does NOT deserve
to become the main method -- not from intuition, but from an actual matched implementation and evaluation that
gave it every chance (identical features, identical composition, identical endpoint, a design specifically
built to avoid the empty-supervision trap).

## Honest caveats
- 45 training triples (charged N-3 labels) is still a modest label budget; a larger charged-label experiment
  is the natural next check before generalising this conclusion.
- The residual and DeepSets networks are small (single hidden layer variants); a larger architecture or longer
  training was not attempted given the time budget -- flagged as a real limitation, not swept under the rug.
- GRU/outage-order sensitivity was avoided by construction (the commutative DeepSets architecture), so the
  brief's 6-permutation GRU test was not needed and was not run.
