# Post-hoc diagnostics of the learning-curve result (exploratory; `scripts/lc_diagnostics.py`)
Not pre-specified. Fresh operating points 300-379, cell n2_unseen|novel, physics feature sets, 5 replicates, R-precision
(mean over replicates and 80 operating points; no confidence intervals computed).

**Q1. Does the raw flag set carry information about y that the explicit control vector lacks? No.** Over 3,750,325 groups of
(operating point, outage, control vector) in the stored labels, 0 groups have more than one distinct y (median 2,583 failure sets
map to each control vector). The label is a function of the control vector, so the explicit vector is a sufficient statistic and
the raw flags add nothing about y.

**Q2. Is the deficit a hyper-parameter artefact? Largely, for n <= 10.** Each feature set was run with (own tuned params),
(the explicit set's params), (the raw set's params):

| params | features | n=5 | n=10 | n=25 |
|---|---|---|---|---|
| raw set's params | explicit (ctrl) | 0.691 | 0.737 | 0.775 |
| raw set's params | raw flags | 0.688 | 0.745 | 0.796 |
| explicit set's params | explicit (ctrl) | 0.621 | 0.717 | 0.773 |
| explicit set's params | raw flags | 0.651 | 0.737 | 0.777 |

Explicit minus raw: with the raw set's params +0.003 / -0.008 / -0.021; with the explicit set's params -0.030 / -0.020 / -0.004;
with each set's own params (the pre-specified run) -0.067 / -0.028 / -0.023. Under no regime does explicit beat raw by 0.05, so H1
stays unsupported, but the size of the deficit depends on tuning, and a residual raw edge of about 0.02 at n=25 appears under
two of three regimes.

**Q3. Which part of the explicit set matters (explicit set's params / raw set's params)?** fractions C only 0.616/0.670 at n=5;
C plus total 0.619/0.686; unit reachability U only 0.528/0.590 (worst); full explicit 0.621/0.691; raw plus explicit 0.610/0.687 -
i.e. per-generator commandable fractions carry the signal, and unit-level reachability alone is a poor representation for trees.

**Not established:** why raw flags keep a ~0.02 edge at n=25; whether it persists at larger n or on other wirings; the
statistical significance of these small gaps (single set of 5 replicates, no bootstrap).
