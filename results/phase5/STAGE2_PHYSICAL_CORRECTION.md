# Phase 5 Stage 2 — physical (island-floor) correction

## V(empty)=0, proven not assumed
`nominal_rating()` calibrates branch ratings from the base-case (N-0) dispatch flows with margin; `sample_op()`
only accepts a dispatch that is ALREADY flow-feasible under that same rating array. At N-0, setting
shed=redispatch=0 in `ScenarioLP` reduces its equality constraint to exactly the accepted dispatch's own balance
equation, so the same (rating-feasible) flow solution applies for ANY control vector -- shed=0 is feasible,
hence optimal, for every c. Verified (`tests/test_vempty.py`): exactly zero across 15 operating points x 10
random control vectors each (150 checks, not just the one fixed control state used elsewhere).

## Three composition variants, compared on the CACHED confirmatory tables (no new LP solves)
`src/fdna/physical_correction.py`: `g2_plain` (as before), `g2_clipped` (clip to `[F(S),1]`, `F` = exact
generator-less-island demand fraction, `src/fdna/hik_diag.minimal_cut_struct`), `g2_residual` (Mobius
truncation of the residual `V(T,c)-F(T)`, floor added back exactly once). Correctness-gated
(`tests/test_physical_correction.py`, 5/5 pass): `island_floor` matches `ScenarioLP.struct_mw` exactly; clipping
never violates its own bounds; the residual form collapses to exactly `F(S)` when every sub-term's residual is
zero (a constructed degenerate check).

**Finding: `g2_residual` is algebraically identical to `g2_plain` on every one of the 70 confirmatory triples.**
Not a bug -- verified directly: zero of the 70 triples are a genuinely NEW order-3 cut
(`src/fdna/hik_diag.is_new_cut`, checked exhaustively over the manifest). `g2_residual` only differs from
`g2_plain` when `F` itself has nonzero order-3 Mobius mass, which requires a new cut; without one, the
correction is provably a no-op. `g2_clipped` gives a small MAE improvement in the `all_gen_or_mixed` class
(~25% relative, e.g. 0.000209 vs 0.000279 at full control) but **missed-severe counts are IDENTICAL across all
three variants at both control states tested (1/1/1 at full control, 12/12/12 at no control)** -- the
predeclared exploratory gate (20% relative missed-severe reduction) is NOT met by any variant on this sample.
Honest null result for the physical correction, on this specific confirmatory manifest.

## Hand-checkable N-3 counterexample (required deliverable)
`(9, 26, 33)`: verified a genuine new 3-way cut (`is_new_cut`). At operating point 0 (seed 0), full control:
every one of the 6 N-1/N-2 sub-labels is EXACTLY zero (`tests/test_physical_correction.py`, permanent
regression test), yet the N-3 label is 0.050 -- concrete, hand-verifiable proof that low-order information
cannot in general identify a higher-order interaction. **A second, distinct finding while building this case:**
the shed here is NOT the simple structural floor (`struct_mw` is exactly 0 for this triple/op) -- it comes from
a congestion/trip-rule effect in the surviving connected network. So at least TWO distinct mechanisms produce
emergent N-3 shed: (1) new generator-less islands (`F(S)`, addressed by `g2_clipped`/`g2_residual`), and (2)
network congestion/trip effects with no structural floor at all (addressed by neither variant here). This is
why the confirmatory sample's null result is not surprising -- it contained examples of neither mechanism-1
(no new cuts) in a form the correction could exploit, and mechanism-2 is untouched by any variant tested.

## What this means for the programme
The physical correction is a real, well-motivated idea, correctly implemented and gated, but this specific
confirmatory sample cannot demonstrate its value (no genuinely new cuts in it) and cannot rule out congestion-
driven emergent shed either. A manifest specifically enriched for new-cut triples (using the free,
LP-solve-less `is_new_cut`/`generalized_det` screen already built in Stage 3 of the earlier session) would be
needed to properly test `g2_residual`'s value -- not done tonight; recorded as a specific next step.
