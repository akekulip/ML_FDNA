# Proposed future confirmation — local protocol freeze

Status: **not registered, opened, or labeled**. This freezes a proposed follow-up
to the exploratory development result. It does not report a confirmatory result.
The hardware checks reuse development data and do not open this manifest.

The sole candidate is `cv_full` with a **warm cache**, fixed target budget
**280 scalar LP labels per operating point**, and fixed beta=1. Cold operation
remains a no-go under the tested gate. The full lower-order cache is preexisting;
report its 227,328-label construction cost separately. This is a conditional
warm-operation claim on the same IEEE-30 topology and fixed 70 outage triples.

## Frozen experiment

- Proposed manifest: [proposed_confirmation_manifest.json](proposed_confirmation_manifest.json),
  280 new operating points, IDs 10000–10279. Outages are the same fixed 70 triples;
  this design tests new operating points, not new outage sets or topologies.
- Manifest SHA256: `14deb9ff8232f073f2f968b1c1892051789e7fedf9cb913440ce63bd373ac64f`.
- Cells: P1 (`q=.7`, stale `.3`, full coverage) and P2 (`q=.3`, stale `.2`,
  existing sparse coverage), unchanged from the development protocol.
- Comparators: posterior MC with affordable exact enumeration, and `proxy_full`
  with zero target queries. All policies use the same observation and hidden
  state per op/cell. Use observation RNG `[op_id, cell_index]`, cell indices 1/2,
  and ten paired target-draw replicates with the existing candidate RNG
  `[op_id, *outage, cell_index*100+replicate, 99]`.
- Allocation: four fixed posterior draws per candidate (70 candidates); duplicate
  requests share a label within each policy, while sample multiplicities remain.
  No target cache is carried across policies, cells, ops, or replicates.
- Primary endpoint: per-op severe-case R-precision, evaluated with raw scores and
  the existing stable candidate-key tie-breaker. Hidden severe counts are used
  only by the evaluator. Report no-severe cases; use a common eligible-op mask.
- Compute paired differences for all four cell × comparator contrasts. Resample
  operating-point rows and replicate columns together as paired units, using
  4,000 bootstrap draws and seed 9025. Every one-sided 95% lower bound must exceed
  **0.01 absolute R-precision**. No contrast, cell, or weak outcome may be removed.
  This is a conjunction claim, with no selection among candidate methods on the
  new data. Report failure if any contrast fails.
- Secondary endpoints remain raw/clipped probability MSE and per-op
  recall/precision at 10%, 20%, and 40%, including saturated outcomes. They cannot
  rescue a failed primary gate. Keep failures and infeasible methods visible.

## Sample-size rationale

[confirmation_design.json](confirmation_design.json) records the development
metrics hash, paired variability, input-manifest audit, and calculations. For each
contrast, form the 60 × 10 matrix of paired R-precision differences. Estimate the
mean's standard error with the same paired bootstrap and set
`sigma_equivalent = bootstrap_SD * sqrt(60)`.

Use the normal planning approximation
`N = ceil(((z(.95) + z(.975)) * sigma_equivalent / (mean_difference - .01))²)`.
An individual power target of .975 for each of four required contrasts gives
a .90 joint-power target by the union bound, without assuming the contrasts
are independent. This is an approximate design calculation, not a guarantee.

| Contrast | Development difference | Equivalent SD | Required N |
|---|---:|---:|---:|
| P1 vs MC | .03110 | .05104 | 77 |
| P1 vs proxy | .01564 | .02529 | 262 |
| P2 vs MC | .04177 | .06698 | 58 |
| P2 vs proxy | .02060 | .03088 | 111 |

Round the maximum requirement up to **280 operating points**. Selection on this
development data can overestimate future gains; both the normal approximation
and extrapolation of the bootstrap variability are limitations. The known unused
reserve tail, 660–699, has only 40 points and is insufficient for this design
(the limiting contrast's approximate power is .41). Preserve that tail unopened.

## Required before any future label generation

IDs 10000–10279 lie outside the current `fdna.blocks.BLOCKS` domain. The proposed
manifest has no overlap with explicit op IDs in the checked local manifests and
cached tables. That is a repository-scoped check, not proof of global non-use;
files without explicit IDs cannot establish freshness.

Before labeling, reconcile other execution records, register this new domain and
the two cell slots through an explicit registry amendment and block-lock support,
then commit the frozen protocol and source under the user's identity with a clean
tracked tree. Use the normal one-time block locks. Do not bypass these controls,
silently reuse the existing reserve lock, change the manifest after seeing labels,
or treat this local document as an opened block.

None of these fresh-data actions were executed during the current implementation.
