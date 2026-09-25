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

**Corrected reasoning (external review, finding 8):** the original write-up used `is_new_cut(S)==False` as the
justification. That is WRONG in general -- `is_new_cut` asks only whether `S` disconnects the graph when no
proper subset does; it says nothing about whether `F` (the island-floor function) itself has nonzero
third-order Mobius mass, which is the quantity that actually determines whether `g2_residual` can differ from
`g2_plain`. A three-node counterexample (`tests/test_physical_correction.py`) makes this concrete: `is_new_cut`
is False there, yet `F(S) - Mobius_order_2[F](S) = -1`, clearly nonzero, and `g2_residual` correctly recovers
`F(S)=1` where `g2_plain` gives `2`. **The correct test was run directly** on all 4,200 (op,triple) rows of the
real confirmatory sample: `F(S) - Mobius_order_2[F](S)` is exactly zero everywhere, verified exhaustively (not
sampled) -- so the empirical finding (`g2_residual`==`g2_plain` on this manifest) still holds, but for the
right, directly-verified reason, not the wrong proxy the original write-up used.

`g2_clipped` gives a small MAE improvement in the `all_gen_or_mixed` class (~25% relative, e.g. 0.000209 vs
0.000279 at full control). **Missed-severe counts, corrected (external review, finding 7 -- the original eval
script had a sign error, reconstructing `truth-err` instead of `truth+err`, silently hiding real misses):
47/47/47 at full control (was wrongly reported as 1/1/1), 38/38/38 at no control (was wrongly reported as
12/12/12).** The three variants still give IDENTICAL counts to each other (the comparative finding is
unaffected by the sign bug), so the predeclared exploratory gate (20% relative missed-severe reduction) is
still NOT met by any variant -- an honest null result, now on corrected absolute numbers.

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
confirmatory sample cannot demonstrate its value (`F`'s third-order Mobius mass is exactly zero throughout it,
verified directly) and cannot rule out congestion-driven emergent shed either. A manifest specifically enriched
for nonzero third-order floor interaction (test `F(S) - Mobius_order_2[F](S)` directly, not `is_new_cut`) would
be needed to properly test `g2_residual`'s value -- not done tonight; recorded as a specific next step.

**A stronger physical mechanism, proposed by the external review and independently verified:** the existing
`island_floor` only catches generator-LESS islands; an island CAN contain a generator and still lack enough
reachable capacity to serve its load. `F_capacity(S,c) = sum_I max(0, D_I - sum_{g in I} U_g(c)) / total_demand`
(`D_I` = island demand, `U_g(c)` = each generator's control-dependent reachable output). Verified on a small toy
LP (two islands, each with a generator; loads 20/80 MW, base generation 50/50 MW, one generator capped at 60 MW
reachable): generator-less floor gives 0%, the capacity floor gives 20%, and the repository's own exact LP also
gives 20% -- exact agreement. This is Stage R5's next physical-correction candidate, not yet implemented.
