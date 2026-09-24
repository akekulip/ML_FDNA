import numpy as np

from fdna import hik, v2data
from fdna.dataset import G


def test_outage_key_canonical_sorted_dedup():
    assert hik.outage_key((5, 2)) == (2, 5)
    assert hik.outage_key((3, 3, 1)) == (1, 3)
    assert hik.outage_key(()) == ()


def test_is_feasible_rejects_duplicates_and_out_of_range():
    assert hik.is_feasible(G, (1, 2))
    assert not hik.is_feasible(G, (1, 1))
    assert not hik.is_feasible(G, (-1, 2))
    assert not hik.is_feasible(G, (G.n_branch, 2))


def test_n_possible_ksets_matches_math_comb():
    from math import comb
    assert hik.n_possible_ksets(G.n_branch, 3) == comb(G.n_branch, 3)
    assert hik.n_possible_ksets(G.n_branch, 4) == comb(G.n_branch, 4)


def test_manifest_sets_are_distinct_and_canonical():
    m = hik.sample_kset_manifest(G, k=3, n_sets=40, seed=1)
    assert len(m.outage_sets) == 40
    assert len(set(m.outage_sets)) == 40
    for s in m.outage_sets:
        assert s == hik.outage_key(s) and len(s) == 3
    assert m.seed == 1 and set(m.strata) <= {"representative", "stress"}


def test_manifest_reproducible_from_seed():
    m1 = hik.sample_kset_manifest(G, k=3, n_sets=20, seed=7)
    m2 = hik.sample_kset_manifest(G, k=3, n_sets=20, seed=7)
    assert m1.outage_sets == m2.outage_sets


def test_label_kset_matches_old_oracle_on_n0_n1_n2():
    """Backward-compatibility gate (brief 6.2): the new hik.label_kset path must reproduce the EXISTING
    value-table numbers exactly for k=0/1/2, using the same op and the same CV rows, before any k=3/4 label
    is trusted."""
    tr = v2data.load_vtable("data_v2", "train")
    CV = tr["CV"]
    op_id = int(tr["op_ids"][0])
    for j in range(3):                          # a mix of N-0, N-1 rows present in the train table
        outage = tuple(int(x) for x in tr["conts"][0, j] if x >= 0)
        y_old = tr["V"][0, j].astype(np.float64)
        y_new, any_infeasible, note = hik.label_kset(G, op_id, outage, CV)
        assert not any_infeasible, note
        assert np.allclose(y_old, y_new, atol=1e-6), (outage, np.abs(y_old - y_new).max())
    # a genuine N-2 row (not present as a single outage in the train table's N-0/N-1 slots)
    va = v2data.load_vtable("data_v2", "val")
    n2j = int(np.flatnonzero(va["conts"][0, :, 1] >= 0)[0])
    outage = tuple(int(x) for x in va["conts"][0, n2j])
    op_id2 = int(va["op_ids"][0])
    y_old = va["V"][0, n2j].astype(np.float64)
    y_new, any_infeasible, note = hik.label_kset(G, op_id2, outage, va["CV"])
    assert not any_infeasible, note
    assert np.allclose(y_old, y_new, atol=1e-6)


def test_manifest_size_error_when_exceeding_total():
    import pytest
    with pytest.raises(ValueError):
        hik.sample_kset_manifest(G, k=G.n_branch - 1, n_sets=10**9, seed=1)
