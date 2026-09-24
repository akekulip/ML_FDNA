# Milestone 1 report — reference generator and non-FDNA baselines (IEEE 30-bus, DC corrective shed)

Data, code and spec: repo commit at the time of writing (see `git log`); label spec hash: see `scripts/spec_hash.py` (covers `configs/spec.json` plus the LP, communication, grid and
operating-point code; the value was `cabbf45539d69daf` before the code was included), spec prose hash reported by `spec.doc_hash()`. All numbers below are from the exploratory test
operating points (seeds 200-279), which have been inspected; confirmatory claims use fresh points (see
`results/LEARNING_CURVE.md`). Tables regenerate with `scripts/{diagnostics,multiseed,paired,error_analysis}.py`.

## 1. Finding (narrow)
Under this fixed communication wiring and training regime, tuned tree models learn the binary dependency rule (control
centre up, unit controller up, at least one parent gateway up) from the 14 raw component flags about as well as from an
explicitly computed control-availability vector. On unseen N-2 with physics features the difference is -0.0004 in
R-precision (95% CI -0.005 to +0.004); without physics -0.007 (-0.015 to +0.0005). Both are far below the +0.05 convention
for a difference worth caring about. This does **not** show that an FDNA architecture cannot help: no FDNA layer, GNN,
low-data regime or changed wiring has been tested here.

## 2. What was built
DC corrective load-shed LP with local balancer, bounded remote redispatch, island handling and trip rule
(`src/fdna/lp.py`); synthetic 14-component communication layer with dual-homed units (`comm.py`); operating-point
sampler (`opgen.py`); order-independent dataset assembly (`dataset.py`); shared evaluation utilities (`evalutil.py`).
Tests: 21 pass (cvxpy cross-check, control-set monotonicity, structural-shed bound, ten hand-solved cases, generator
order independence, tie-break/subsample invariance, spec-hash enforcement). The cvxpy check verifies implementation
agreement only; both formulations share the benchmark assumptions in `prereg/SPEC.md`.

## 3. Benchmark composition (`scripts/diagnostics.py`)
1,471 failure sets (up to 4 failed of 14 components) map to only **501** distinct control vectors. Training uses 68 failure
sets / **47** control vectors. Of unseen-N-2 test rows, **36.1%** have a control vector that also occurs in training, so
the "unseen" cell mixes familiar and novel control states:

| cell | complete control loss | novel control vector | severe prevalence | failure-set sizes |
|---|---|---|---|---|
| n1_seen | 13.2% | 0% | 10.1% | mostly 1-2 |
| n1_unseen | 23.0% | 63.9% | 12.8% | 2/3/4 = 40/30/30% |
| n2_seen | 12.7% | 0% | 24.6% | mostly 1-2 |
| n2_unseen | 23.0% | 63.9% | 29.6% | 2/3/4 = 40/30/30% |

Inside unseen N-2, severe prevalence is 39.8% for familiar and 23.9% for novel control vectors. R-precision depends on
prevalence, so it must not be compared across these strata; a constant predictor now scores at each stratum's prevalence
(0.297 / 0.398 / 0.238), which validates the tie-handling.

## 4. Results (5 seeds, common training config; mean over 80 test operating points)
Unseen N-2, split by control-vector novelty (R-precision = recall at a budget equal to the number of severe cases):

| model | all unseen | familiar (prev 0.40) | novel (prev 0.24) | recall@20% (all) | AP (all) | MAE (all) |
|---|---|---|---|---|---|---|
| zero (chance) | 0.297 | 0.398 | 0.238 | - | - | - |
| elec | 0.640 | 0.753 | 0.566 | 0.475 | 0.677 | 0.0116 |
| elec+phys | 0.678 | 0.803 | 0.577 | 0.493 | 0.743 | 0.0109 |
| elec+raw_comm | 0.789 | 0.851 | 0.720 | 0.639 | 0.870 | 0.0058 |
| elec+ctrl (explicit) | 0.783 | 0.846 | 0.715 | 0.622 | 0.850 | 0.0065 |
| elec+phys+raw_comm | 0.874 | 0.925 | 0.827 | 0.673 | 0.941 | 0.0048 |
| elec+phys+ctrl (explicit) | 0.873 | 0.934 | 0.815 | 0.666 | 0.941 | 0.0045 |

Paired differences (A minus B, cluster bootstrap over operating points; * = interval excludes 0). Positive favours A.

| question | all unseen N-2 | familiar | novel |
|---|---|---|---|
| explicit vs raw, no physics | -0.007 [-0.015, +0.0005] | -0.005* | -0.005 [-0.017, +0.006] |
| explicit vs raw, with physics | -0.0004 [-0.005, +0.004] | +0.009* | -0.013* |
| control information (with physics: ctrl minus none) | +0.196* | +0.131* | +0.238* |
| physics features (with ctrl) | +0.091* | +0.088* | +0.099* |

Seed-to-seed SD of the operating-point-averaged R-precision on unseen N-2: 0.0007-0.0054. Control information is worth
+0.09 to +0.24 R-precision depending on stratum and feature set (+0.14 / +0.20 on all unseen N-2 without / with physics); physics features about +0.09. Where explicit and raw differ significantly the signs go both ways, but without physics features raw states are
clearly ahead in several cells: -0.050 on unseen-failure-set N-1 [-0.070, -0.033], -0.055 on seen N-1 [-0.076, -0.037] and
-0.014 on seen N-2 [-0.024, -0.007]; with physics features the significant differences are between -0.013 and +0.013. On the
unseen-N-2 cell that the question concerns, no difference exceeds 0.013 in magnitude, but the N-1 results show a tree
model can be measurably worse with the explicit vector than with raw flags, which is the opposite of an explicit-calculation
advantage.

## 5. Error analysis (descriptive; best model elec+phys+ctrl, unseen N-2, `scripts/error_analysis.py`)
Share of truly severe scenarios missed at a prevalence-matched budget: 17.7% for novel control vectors vs 6.3% familiar;
28.6% when structural shed (islanding) is present vs 7.2%; 36% when the DC no-control overload ratio is <=1 (severe cases
without an obvious overload) vs under 1% when the ratio exceeds 2; 22% within 0.2 points of the 1% threshold vs 8% more than
3 points away; 3% when no control is available vs 20% when most control is available. These tabulate where errors
concentrate; they do **not** isolate causes, and this report no longer claims the remaining error is "electrical".

## 6. Cost of the workflow (measured, single core)
LP solve about 1.7 ms (2.8 ms including model build). Post-outage DC-flow feature block 0.19 ms per (operating point,
outage), shared across all communication states. Tree inference 0.3-0.9 microseconds per scenario. On a 30-bus case the
exact LP is already cheap; no speed claim is made.

## 7. Physical-assumption sensitivity (development seeds, `scripts/trip_pilot.py`)
Share of N-2 groups whose shed varies with control by more than 0.5% of demand: 0.382 (frozen trip rule), 0.382
(surplus-only trips; 2.7% of scenarios have no feasible action), 0.368 (free trips). The value of control is not an artefact
of the trip rule.

## 8. Deviations and corrections to earlier statements
See `prereg/SPEC.md` amendments 1-7. In short: the +0.05 convention was not in the frozen spec; the first five-seed run used a
different training config than tuning (superseded); "unseen scores higher so the split is not stressful" and "remaining error
is electrical" were unsupported and are withdrawn; control-sensitivity (43% of test N-2 groups) exceeds the 15-40% target;
tie-handling and subsampling were order-dependent and are fixed (tests added).

## 9. Limitations
Single 30-bus DC benchmark, one synthetic communication wiring, binary reachability truth (any dependency computation can
reproduce it); tree models only; hyper-parameters tuned once on full training data; comparisons exploratory and unadjusted for
multiplicity; test rows are a 25% id-hash subsample; Roy & Hylviu read at abstract level only.
