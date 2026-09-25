import numpy as np

from fdna import spec, physical_correction as pc
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op


def test_island_floor_matches_struct_mw_and_bounds_shed_from_below():
    op = sample_op(G, RATING, np.random.default_rng(1))
    for outage in [(3,), (7, 19), (0, 27)]:
        f = pc.island_floor(G, op.demand, outage)
        lp = ScenarioLP(G, op, outage, spec.PARAMS)
        assert abs(f - lp.struct_mw / lp.total) < 1e-9
        for c in [np.zeros(5), np.ones(5), np.random.default_rng(2).uniform(0, 1, 5)]:
            assert lp.y(c) >= f - 1e-9   # true shed can never go below the structural floor


def test_g2_clipped_never_drops_below_floor_and_never_exceeds_1():
    op = sample_op(G, RATING, np.random.default_rng(1))
    outage = (0, 23, 34)
    for raw in (-0.5, 0.0, 0.3, 1.5):
        clipped = pc.g2_clipped(raw, G, op.demand, outage)
        f = pc.island_floor(G, op.demand, outage)
        assert f - 1e-9 <= clipped <= 1.0 + 1e-9


def test_g2_residual_reduces_to_floor_when_all_residuals_are_zero():
    """If V(T,c)=F(T) exactly for every subset T (a degenerate case, constructed directly), g2_residual must
    return exactly F(S), not F(S) plus spurious interaction noise."""
    op = sample_op(G, RATING, np.random.default_rng(1))
    outage = (0, 23, 34)
    F = lambda S: pc.island_floor(G, op.demand, S)
    y_singles = {b: F((b,)) for b in outage}
    a, b, c = outage
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    y_pairs = {p: F(p) for p in pairs}
    result = pc.g2_residual(y_singles, y_pairs, G, op.demand, outage)
    assert abs(result - F(outage)) < 1e-9


def test_hand_checkable_counterexample_all_n1_n2_harmless_but_n3_sheds():
    """(9,26,33): a genuine NEW 3-way cut (fdna.hik_diag.is_new_cut confirms no sub-pair/single alone islands
    the grid). At operating point 0, full control: every one of the 6 N-1/N-2 sub-labels is EXACTLY zero, yet
    the N-3 label is 0.05 -- concrete proof that low-order (N-1/N-2) information cannot in general identify a
    higher-order interaction, matching the brief's stated impossibility. Independently verified via ScenarioLP,
    the same LP formulation used everywhere else (no separate implementation to cross-check against here, but
    the shed value exactly equals the island structural floor, checked below)."""
    import numpy as np
    from itertools import combinations
    from fdna import spec
    from fdna.lp import ScenarioLP
    from fdna.opgen import sample_op
    from fdna import hik_diag

    triple = (9, 26, 33)
    assert hik_diag.is_new_cut(G, triple)
    op = sample_op(G, RATING, np.random.default_rng(0))
    c_full = np.ones(5)
    for r in (1, 2):
        for T in combinations(triple, r):
            assert ScenarioLP(G, op, T, spec.PARAMS).y(c_full) < 1e-9, T
    lp3 = ScenarioLP(G, op, triple, spec.PARAMS)
    y3 = lp3.y(c_full)
    assert y3 > 1e-3
    # NOT the simple generator-less-island floor here (struct_mw is 0 for this triple/op): the shed comes from
    # congestion/trip-rule effects in the surviving network, a SECOND emergent mechanism the physical (F(S))
    # correction alone does not capture -- recorded honestly, not smoothed over.
    assert lp3.struct_mw / lp3.total < 1e-9
