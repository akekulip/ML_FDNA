import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fdna.dataset import G, PAIRS
from fdna.lodf import compensation_det, ptdf_lodf, risk_table


def _n_components(removed):
    k = np.array([i for i in range(G.n_branch) if i not in removed])
    n, _ = connected_components(coo_matrix((np.ones(len(k)), (G.frm[k], G.to[k])), shape=(G.n_bus,) * 2), directed=False)
    return n


BRIDGES = {i for i in range(G.n_branch) if _n_components((i,)) > 1}


def test_det_zero_exactly_recovers_new_topological_2cuts():
    """Det=0 detects a NEW 2-cut (neither branch alone islands, together they do) -- STEP1.md's 'new_2cut' axis (26/820).
    Pairs where one member is already a single-line bridge are a different regime: LODF is built on the base (fully
    connected) topology, so it is not the right tool there -- those 117 pairs are excluded, matching STEP1's own split."""
    n_checked = n_islands = 0
    for a, b in PAIRS:
        if a in BRIDGES or b in BRIDGES:
            continue
        n_checked += 1
        det = compensation_det(ptdf_lodf(G)[1], a, b)
        islands = _n_components((a, b)) > 1
        n_islands += islands
        assert (abs(det) < 1e-8) == islands, (a, b, det, islands)
    assert n_islands == 26, n_islands  # matches STEP1.md new_2cut count


def test_det_symmetric_and_bounded():
    _, LODF = ptdf_lodf(G)
    for a, b in PAIRS[:50]:
        assert abs(compensation_det(LODF, a, b) - compensation_det(LODF, b, a)) < 1e-9


def test_risk_table_matches_direct_computation():
    r = risk_table(G, PAIRS[:20])
    _, LODF = ptdf_lodf(G)
    for i, (a, b) in enumerate(PAIRS[:20]):
        d = compensation_det(LODF, a, b)
        want = np.inf if d == 0 else 1 / abs(d)
        assert np.isclose(r[i], want) or (np.isinf(r[i]) and np.isinf(want))
