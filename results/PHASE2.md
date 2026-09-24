# Phase 2 — overnight programme log and results (living document)
Started 2026-09-23 23:00:52; hard wall-clock cap 10 h. Registry: `registry/registry.yaml` (locked before any experiment).
Nothing below has used the confirm (400-479) or replicate (500-579) seed blocks; screens use the inspected exploratory block 200-279.

## Screens completed
| Branch | Question | Result | Status |
|---|---|---|---|
| B3 (monotone value model) | does a hard monotone-in-control prior beat monotone-constrained trees at n<=25? | No: monotone NN is 0.06-0.13 R-precision below monotone GBM (novel cell); the monotone constraint itself costs trees ~0.025 | killed as hypothesis |
| B2 (corrupted wiring) | can a learned dependency layer beat tuned trees on raw flags when 20% of assumed edges are wrong? | No: 0.096-0.159 below trees. But it beats a generic MLP on the same raw flags by +0.014/+0.047/+0.092 (n=10/25/100) | killed as hypothesis; side finding recorded |

## Diagnoses that changed the programme
1. **Neural baselines were mis-trained (scale of the target).** First Branch 3 run had neural arms at 0.34-0.58 vs trees 0.70-0.93. Cause: shed fractions ~1e-2 give MSE ~1e-4 and starve gradients; multiplying the target by 100 lifts a plain MLP from 0.38 to 0.80 (n=25) and 0.64 to 0.86 (n=100) on validation. Protocol lesson: an under-trained neural baseline makes any structured model look good, so every neural arm needs a target-scale/recipe sanity check against trees before comparison. Recorded as an invalid run, not as a test of monotonicity.
2. **Trees do not need dependency structure when observations are complete** (v1 finding Q1 and B2): the label is a function of the control vector, and a tree on 14 raw flags learns it; tree + correct wiring is no better than tree + raw flags (0.796 vs 0.806 at n=25).
3. **Structure does help neural networks relative to generic neural networks** (B2 side finding), which reframes the open question: where can a structured NN beat *trees*? Only where information is missing (v2, gate E1) or the problem is too large for tree/enumeration approaches.

## Pending
E1 (oracle headroom under partial/stale observation; four parallel cells), then, only if E1 passes and v2 is frozen, B1 arms A1-A7.
