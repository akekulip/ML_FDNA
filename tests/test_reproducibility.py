import numpy as np
import pandas as pd
import pytest

from fdna import dataset, evalutil, spec


def test_generator_is_order_independent():
    jobs = [("train", 0), ("train", 1), ("val", 100)]
    a = dataset.assemble([dataset.job(j) for j in jobs])
    b = dataset.assemble([dataset.job(j) for j in reversed(jobs)])
    pd.testing.assert_frame_equal(a[0], b[0])
    for k in a[1]:
        np.testing.assert_array_equal(a[1][k], b[1][k])


def test_subsample_and_tiebreak_stable_under_row_permutation():
    lab = pd.DataFrame(dict(op=np.repeat(np.arange(5), 40), b1=np.tile(np.arange(40), 5), b2=-1, fs=3))
    perm = np.random.default_rng(0).permutation(len(lab))
    s0, s1 = evalutil.in_subsample(lab, 0.3), evalutil.in_subsample(lab.iloc[perm], 0.3)
    np.testing.assert_array_equal(s0[perm], s1)
    np.testing.assert_allclose(evalutil.tiebreak_key(lab)[perm], evalutil.tiebreak_key(lab.iloc[perm]))
    assert 0.2 < s0.mean() < 0.4


def test_zero_baseline_is_row_order_independent_and_near_chance():
    rng = np.random.default_rng(1)
    n = 20000
    lab = pd.DataFrame(dict(op=0, b1=np.arange(n), b2=-1, fs=0))
    y = (rng.random(n) < 0.3).astype(float) * 0.05  # 30% severe
    key = evalutil.tiebreak_key(lab)
    r = evalutil.rprec(y, np.zeros(n), key)
    perm = rng.permutation(n)
    assert evalutil.rprec(y[perm], np.zeros(n), key[perm]) == pytest.approx(r)
    assert r == pytest.approx(0.3, abs=0.02)  # chance level, not dataset-order artefact
    # dataset order carries an ordering signal (severe rows first) - must not help
    y2 = np.sort(y)[::-1]
    assert evalutil.rprec(y2, np.zeros(n), evalutil.tiebreak_key(lab)) == pytest.approx(0.3, abs=0.02)


def test_spec_hash_mismatch_raises(tmp_path):
    (tmp_path / "spec_hash.txt").write_text("deadbeef")
    with pytest.raises(RuntimeError, match="mismatch"):
        evalutil.check_spec_hash(tmp_path)
    (tmp_path / "spec_hash.txt").write_text(spec.spec_hash())
    assert evalutil.check_spec_hash(tmp_path) == spec.spec_hash()
