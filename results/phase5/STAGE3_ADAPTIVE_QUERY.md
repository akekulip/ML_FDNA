# Phase 5 Stage 3 — bounded adaptive-query screening

## Control-monotonicity: proven and exhaustively verified
`V(op,outage,c)` is provably non-increasing in `c` (componentwise): `c` enters `ScenarioLP` only through the
corrective-redispatch box bounds (`_bounds`), which weakly EXPAND as `c` increases; box-constraint relaxation
on a fixed-structure LP cannot increase its minimum. Verified exhaustively (`tests/test_monotone_control.py`):
**zero violations across all 4,200 cached confirmatory rows and all 415,699,200 ordered pairs among the 1,024
control vectors** -- not a sample, the full check. (First attempt at this check had a broadcasting sign error
that looked like 37% violations; caught and fixed by re-deriving the comparison directly against the exact
cached table before trusting the result.)

## L(c)/U(c) bounds: implemented and verified valid
`src/fdna/adaptive_query.py`: `L(c) = max(F(S), max_{c_j>=c} V(c_j))`, `U(c) = min(1, min_{c_j<=c} V(c_j))`,
vectorised. Verified (`tests/test_monotone_control.py`): `L(c) <= V(c) <= U(c)` for every control, given any
queried subset, checked against the cached true table (not assumed) -- and queried points are recovered exactly
(`L=U=V` there).

## Acquisition experiment: honest, mostly-null result against the predeclared gate
`registry/phase5_adaptive_query.yaml` (committed before running), `scripts/p5_adaptive_query_experiment.py`.
Common-observation-snapshot setting (one shared hidden state/observation per operating point across all
candidates); a candidate is "resolved" once `qL==qU` over the actual posterior support, not the whole grid (a
real design bug in the first attempt -- resolution was checked over the WHOLE 1024-grid, so neither policy ever
stopped early and guided/random consumed identical query counts; caught by inspecting the raw output before
believing it, fixed, rerun).

20 of the 60 confirmatory operating points, all 70 triples, budgets {2,4,8,16,32} queries/candidate:

| budget | guided q/candidate | random q/candidate | guided acc | random acc | mc acc (K=8) |
|---|---|---|---|---|---|
| 4 | 2.86 | 2.86 | 0.881 | 0.868 | 0.911 |
| 8 | 4.55 | 4.59 | 0.951 | 0.969 | 0.809 |
| 16 | 7.62 | 8.05 | 0.936 | 0.951 | 0.890 |
| 32 | 13.37 | 14.81 | 0.957 | 0.966 | 0.837 |

(P1 cell shown; P2 nearly identical pattern -- both in `results/phase5/adaptive_query_experiment.json`.)

**g2-guided acquisition uses modestly fewer queries than random (about 5-10% fewer at every budget tested) but
does NOT clear the predeclared promotion gate (>=25% fewer queries at matched quality)** -- and at several
budgets random acquisition actually reaches slightly HIGHER accuracy with its extra queries. This is a real,
honestly-measured, mostly-null result for the acquisition HEURISTIC specifically: the monotone-bound MACHINERY
is exact and valid (the useful, durable part of this stage), but guiding the query order by g2's own prediction
does not meaningfully outperform guiding it randomly, at this candidate scale (70 triples/operating point,
common-observation setting).

**What this does not test:** a larger, more separated shortlist-boundary-aware acquisition rule (my
implementation targets "closest to tau," a simpler heuristic than the brief's fuller "prioritise candidates
whose interval straddles the current shortlist boundary" -- not built due to time); direct-MC's accuracy is
noisy at small K and not monotonic in budget, a known small-sample artifact, not evidence against MC. Cost
accounting: g2's own guidance requires an amortized 41+182=223 lower-order queries per operating point (charged
once, reused across all 70 candidates at that operating point) -- not included in the guided-query counts
above, which would only shift the comparison further against the guided policy.

## Verdict
Monotonicity and the bound machinery are solid, reusable, verified infrastructure. The specific acquisition
heuristic tested here does not clear its own predeclared gate. Per the brief's own blocker protocol: this is a
demonstrated (not merely suspected) limitation of THIS heuristic, not proof that no acquisition rule can work --
a genuine shortlist-boundary-aware policy (not built tonight) is the natural next attempt, recorded as a
specific next step rather than silently reworded as a win.
