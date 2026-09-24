# Learning-curve experiment (pre-specified in `prereg/HYPOTHESIS_LC.md`, committed before fresh data existed)

Full output: `results/learning_curve_output.txt` (`scripts/lc_analysis.py`). Fresh operating points: seeds 300-379 (80),
generated after the hypothesis commit (a1ccb2e). The runs used labels hashed under the earlier label-spec hash; labels were
re-verified identical under the current code (operating points 0, 100, 200, 300 regenerated exactly). 5 replicates per training
size; hyper-parameters fixed to the values tuned on the full training set; early stopping on the fixed validation points.

## Result: H1 NOT SUPPORTED, and the observed direction is opposite
Primary cell (fresh points, novel control vectors, N-2, physics pair), R-precision difference explicit minus raw flags:

| training operating points | 5 | 10 | 25 | 50 | 100 |
|---|---|---|---|---|---|
| explicit - raw | -0.067 [-0.080,-0.055] | -0.028 [-0.036,-0.020] | -0.023 [-0.030,-0.015] | -0.020 [-0.030,-0.011] | -0.014 [-0.024,-0.004] |

H1 required at least +0.05 with a positive lower bound at n = 5, 10 and 25; all three fail, with the sign reversed. Mean
R-precision on that cell, physics pair: explicit 0.621/0.717/0.773/0.789/0.801 versus raw 0.688/0.745/0.796/0.809/0.815 for
n = 5/10/25/50/100. Sample-size ratio: raw flags reach the explicit set's n=100 value (0.801) at n=50; explicit never reaches raw's
n=100 value (0.815) within 100.

## Secondary results (not confirmatory)
- No-physics pair, novel cell: -0.047, -0.080, -0.0004, -0.007, -0.007 at n = 5/10/25/50/100 (the n=10 difference excludes 0).
- Familiar cell: the physics pair is within +/-0.013 of zero (explicit +0.004 to +0.006 for n>=10; -0.012 at n=5); the no-physics pair
  is +0.042 at n=5 but -0.014 at n=10 and -0.004 to -0.009 beyond, i.e. not monotone in n.
- All unseen N-2 pooled: explicit is at most 0.038 below raw (n=5) and within 0.007 by n=50, physics pair.
- The exploratory test points (200-279) give the same picture (novel, physics pair: -0.060, -0.028, -0.025, -0.017, -0.013).

## Interpretation and limits
- The explicit control vector is a deterministic function of the raw flags, so the raw set carries at least the same information;
  trees may exploit which specific component failed (or split on binary flags more readily) in ways the summary vector hides.
  That is a hypothesis for the mechanism, not something this experiment tested.
- Hyper-parameters were tuned at full size and early stopping used the validation set at every size, which may favour one set.
- One communication wiring with binary reachability truth; tree models only; 5 replicates.
- Nothing here bears on whether an FDNA layer could help with partial capacity, stale parameters or changed wiring.

## Decision (from the pre-committed rule)
No low-data gap in favour of explicit dependency calculation, so **stop the FDNA architecture claim on this benchmark**. Per the
rule, do not enlarge or re-tune the simulator to obtain a gap. Any further FDNA work needs a different question (for example
genuinely uncertain or stale dependency information, or a changed-wiring test) and a fresh approved plan.
