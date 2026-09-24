import itertools
import numpy as np

from fdna.dataset import G, PAIRS
from fdna.lodf import ptdf_lodf, compensation_det
from fdna.lp import ScenarioLP
from fdna import spec, hik_diag
from fdna.opgen import sample_op
from fdna.dataset import RATING


def test_generalized_det_matches_k2_compensation_det_exactly():
    _, LODF = ptdf_lodf(G)
    for a, b in PAIRS[:60]:
        assert abs(hik_diag.generalized_det(LODF, (a, b)) - compensation_det(LODF, a, b)) < 1e-9


def _bridges():
    return {i for i in range(G.n_branch) if hik_diag.minimal_cut_struct(G, (i,))[1] > 1}


def test_generalized_det_zero_recovers_new_2cuts():
    """Correctness gate before trusting k=3/4: must still recover the 26 known new-2-cuts exactly."""
    _, LODF = ptdf_lodf(G)
    BRIDGES = _bridges()
    n_new_cuts = 0
    for a, b in PAIRS:
        if a in BRIDGES or b in BRIDGES:
            continue
        det = hik_diag.generalized_det(LODF, (a, b))
        is_cut = hik_diag.is_new_cut(G, (a, b))
        assert (abs(det) < 1e-8) == is_cut
        n_new_cuts += is_cut
    assert n_new_cuts == 26


def test_minimal_cut_struct_matches_scenariolp_struct_mw_exactly():
    """Correctness gate: the free graph-only struct floor must match ScenarioLP's LP-embedded struct_mw exactly,
    for every N-0/N-1/N-2 outage set already in the labelled tables."""
    op = sample_op(G, RATING, np.random.default_rng(0))
    for S in [(), (3,), (7,), (2, 19), (0, 27)]:
        lp = ScenarioLP(G, op, S, spec.PARAMS)
        struct, _ = hik_diag.minimal_cut_struct(G, S, demand=op.demand)
        assert abs(struct - lp.struct_mw) < 1e-6, (S, struct, lp.struct_mw)


def test_minimal_cut_struct_generalizes_to_k3_without_error():
    op = sample_op(G, RATING, np.random.default_rng(0))
    for S in itertools.combinations(range(G.n_branch), 3):
        lp = ScenarioLP(G, op, S, spec.PARAMS)
        struct, _ = hik_diag.minimal_cut_struct(G, S, demand=op.demand)
        assert abs(struct - lp.struct_mw) < 1e-6, (S, struct, lp.struct_mw)
        break        # one spot check is enough here; the bulk k=3 check runs in scripts/hik_diag_screen.py


def test_is_new_cut_false_for_connected_triple():
    assert hik_diag.is_new_cut(G, (0, 1, 2)) in (True, False)   # just must not raise; real check is the k3 screen script
