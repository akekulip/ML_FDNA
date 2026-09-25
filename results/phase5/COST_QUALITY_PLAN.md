# Development plan: complete seed evidence and charge screening costs

Approved in the 2026-09-25 conversation. Development only: operating points 600–659,
the existing 70 triples, P1/P2 observations. No fresh reserve access or retraining.

1. Complete the three-seed DeepSets evaluation, retain per-op/per-seed scores and
   provenance, correct pooling/equivalence language.
2. Add a budget-enforced oracle, deterministic sparse-control proxy, and tested
   fixed-draw control-variate estimator. Fix zero/one-query and outage-validation bugs.
3. Run cold/warm total-cost curves. Primary: per-op severe-case R-precision.
   Compare against MC and the same proxy without target queries; include full g2
   and affordable exact enumeration. Secondary: probability MSE, shortlist
   recall/precision at 10/20/40%, including saturated outcomes.
4. Freeze the smallest development budget clearing an absolute 0.01 margin over
   both baselines in both cells, or publish a no-go. No confirmatory gate changes
   based on fresh data. Record query logs, predictions, artifact/source hashes,
   measured replay and actual-LP timings separately.
5. Independent review, targeted/full tests, reconcile docs and preserve negatives.

## Implementation rulings

- Work in the current checkout with explicit file ownership: it holds the required
  generated labels and checkpoints. Preserve four pre-existing document deletions.
- Use fixed per-candidate draw counts bounded by remaining budget divided by 70.
  Cache duplicate requests but retain sample multiplicities; never stop MC based on
  observed labels or on reaching a random count of unique draws.
- At sufficient budget, the baseline may enumerate posterior support, using only
  known posterior probabilities to decide affordability. Also expose full-grid
  enumeration cost. This prevents penalizing the baseline for redundant MC draws.
- Apply the same affordable-enumeration fallback to CV. Secondary legacy bound
  policies retain their prior 16-query/candidate cap; their unused budget is visible.
- Cold and warm gates are independent claims. A warm-only candidate never
  establishes cold-start efficiency. Execution payloads may be shared across
  identical query plans, but every policy row retains its own budget and charges.
- Sparse anchors are the top posterior-mass controls, counts 2/4/8/16/32/64.
  Interpolate by nearest full-vector L1 distance, ties by control index. Compute
  the thresholded interpolant's exact posterior expectation before target draws.
- Freeze margin 0.01 for this development gate: one extra retrieved severe case
  per 100 severe cases on average. Report uncertainty and the actual severe counts;
  do not lower the threshold if this workload has insufficient headroom.
- Cache replay is simulated LP accounting, not measured solver runtime. A bounded
  real-LP replay checks label agreement and times actual scalar queries separately.

## Task/interface review

| Tasks | Shared contract | Ruling |
|---|---|---|
| Neural eval / runner | Existing cached labels, distinct outputs | No shared mutations |
| Oracle / runner | BudgetedOracle, g2_proxy, estimate_probability | Fixed before dispatch |
| Boundary fixes / oracle | Valid canonical outages, independent modules | Same rejection rules |
| All tasks / verification | Tests and source hashes | Run after integration; no fresh labels |
| Cost / gate | Per-op budget and feasible baseline | Both baselines required; no hidden truth access |
