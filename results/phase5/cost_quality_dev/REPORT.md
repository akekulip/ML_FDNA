# Development cost–quality result

**Exploratory only.** These are reused operating points 600–659 and the existing 70 triples.
No fresh confirmatory block was opened. Primary endpoint: per-operating-point severe-case R-precision.

| Cache regime | Frozen development gate | Selected method / budget |
|---|---|---|
| cold | no_go | None |
| warm | candidate | cv_full / 280 |

### Selected warm candidate at 280 labels/op

| Cell | Comparator | R-precision difference | One-sided 95% lower bound |
|---|---|---:|---:|
| P1_v2b | mc | +0.0311 | +0.0207 |
| P1_v2b | proxy_full | +0.0156 | +0.0104 |
| P2_v2c | mc | +0.0418 | +0.0285 |
| P2_v2c | proxy_full | +0.0206 | +0.0141 |

The gate requires a paired one-sided 95% lower bound above **0.01 absolute R-precision**
against **both** posterior MC and the same proxy without target queries in **both** P1 and P2.
Cold and warm are separate operational claims. A warm result assumes its explicitly priced lower-order cache already exists.
Selection across development candidates is exploratory; these intervals are not post-selection confirmatory evidence.

## Cost and evaluation contract

- Full lower-order cache: **227,328** scalar LP labels/op.
- This manifest needs 40 unique single outages and 182 unique pairs, each at 1,024 controls; unused single-outage labels are not charged.
- Full target enumeration: **71,680** scalar LP labels/op; posterior-support enumeration can be cheaper.
- MC and CV use exact posterior-support enumeration when affordable, decided before accessing target labels.
- Every cold curve charges unique lower-order and N-3 queries; warm curves charge target queries and report sunk construction separately.
- Fixed draw counts retain sampling multiplicities. Repeated labels are evaluated once. Unused budget is reported.
- R-precision severe counts and hidden labels are evaluator-only. Raw CV scores are ranked without clipping.
- Undefined R-precision for zero-severe snapshots is counted explicitly, with common paired eligibility.
- Each operating point/cell has one common observation. Ten target-sampling repeats measure MC variation, not new observations or topologies.
- Infeasible methods have missing plotted points. Full metric and feasibility tables are in `summary.json` / `curves.csv`.
- Curves show point estimates; paired uncertainty is in `comparisons.json`. Sparse warm caches assume the same observation-specific anchors already exist.
- Legacy monotone-bound comparators retain a 16-query/candidate cap; surplus budget is not silently spent.

## Paired differences at 1,120 labels/op

| Regime | Cell | Method | Comparator | Difference | One-sided 95% lower bound |
|---|---|---|---|---:|---:|
| cold | P1_v2b | cv_sparse2 | mc | -0.0023 | -0.0057 |
| cold | P1_v2b | cv_sparse2 | proxy_sparse2 | +0.0349 | +0.0231 |
| cold | P2_v2c | cv_sparse2 | mc | -0.0044 | -0.0081 |
| cold | P2_v2c | cv_sparse2 | proxy_sparse2 | +0.0453 | +0.0323 |
| warm | P1_v2b | cv_full | mc | +0.0094 | +0.0045 |
| warm | P1_v2b | cv_full | proxy_full | +0.0171 | +0.0115 |
| warm | P1_v2b | cv_sparse2 | mc | +0.0025 | +0.0001 |
| warm | P1_v2b | cv_sparse2 | proxy_sparse2 | +0.0397 | +0.0269 |
| warm | P1_v2b | cv_sparse64 | mc | +0.0090 | +0.0043 |
| warm | P1_v2b | cv_sparse64 | proxy_sparse64 | +0.0164 | +0.0106 |
| warm | P2_v2c | cv_full | mc | +0.0186 | +0.0111 |
| warm | P2_v2c | cv_full | proxy_full | +0.0205 | +0.0140 |
| warm | P2_v2c | cv_sparse2 | mc | +0.0006 | -0.0010 |
| warm | P2_v2c | cv_sparse2 | proxy_sparse2 | +0.0503 | +0.0360 |
| warm | P2_v2c | cv_sparse64 | mc | +0.0162 | +0.0095 |
| warm | P2_v2c | cv_sparse64 | proxy_sparse64 | +0.0192 | +0.0123 |

## Secondary endpoints and limitations

Probability-estimation MSE (raw and clipped), and recall/precision at 10%, 20%, and 40% are all retained.
Saturated shortlist endpoints remain in the report; they cannot support a claim of incremental benefit.
The posterior and labels are specific to the existing finite-grid IEEE-30 benchmark.
Sparse interpolation is a surrogate, not a certified physical bound.

## Verification and reproduction

Artifact hashes, input/source hashes, and replay timing are in `provenance.json`.
Complete prediction/query records are generated at `data_hik/cost_quality_dev/` and are not checked into Git.
Run `.venv/bin/python scripts/p5_cost_quality.py` for a serial full reproduction.
Run `.venv/bin/python scripts/p5_cost_quality_replay.py` to reconstruct metrics, audit costs, and replay the frozen real-LP timing workload.
The optional figure renderer needs pandas and Matplotlib. In this environment, both already exist in system Python:
`MPLCONFIGDIR=/tmp/fdna-matplotlib python3 scripts/p5_cost_quality_report.py`.
No new project dependencies were added.

### Independent query-accounting audit

Audited 604,800 policy evaluations and independently reconstructed all 429,600 feasible rows; failures: 0.

### Measured LP timing

Replayed 22,717 scalar LP queries; summed solve time: 60.78 seconds.
Maximum absolute cached-label deviation: 1.4e-08.
Timing covers three selected operating points at budget 1,120, not all 60 development points; warm lower-order cache cost is reported but not timed. Full selected warm-policy hardware verification is separate.

### Actual CPU solver verification

The frozen warm candidate and both baselines were rerun with real `ScenarioLP` target queries
on all 60 development operating points, both cells, and ten sampling repeats.
Actual unique scalar solves: 140,803; maximum label deviation: 1.47e-08.
Maximum score/metric differences: 0 / 0; failures: 0.
Realized severe labels were also re-solved. The warm lower-order cache remains an input,
and probability-MSE truth remains the frozen exact-posterior snapshot. This verifies numerical
reproduction of the selected warm result; it is not a new-data confirmation or a physical-grid experiment.
Full scope, accounting, timing, and provenance: `HARDWARE_VERIFICATION.json`.

## Decision

The selected regime-specific development candidate is recorded in `gate.json`.
It is not a confirmed result. The warm result assumes the full lower-order cache already exists;
building it costs more than enumerating these 70 target outages directly. There is no cold-start promotion.
The proposed follow-up protocol and sample-size rationale are in `CONFIRMATION_PROTOCOL.md`; it is not yet registered.
no fresh data are opened by this development run or its hardware verification.
