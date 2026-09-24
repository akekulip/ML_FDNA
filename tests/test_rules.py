import numpy as np
import pytest

from fdna import rules


def test_holm_and_family_assertion():
    adj = rules.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert adj["a"] == pytest.approx(0.004) and adj["b"] <= adj["c"] <= adj["d"]
    with pytest.raises(ValueError):
        rules.holm({"a": 0.1, "b": 0.2}, expected_family=6)


def test_p_value_has_a_floor_and_margin_test_is_stricter_than_zero_test():
    rng = np.random.default_rng(0)
    d = rng.normal(0.03, 0.01, (80, 3))
    s0 = rules.boot(d, margin=0.0, n_boot=2000)
    s5 = rules.boot(d, margin=0.05, n_boot=2000)
    assert 0 < s0["p"] < 0.01 and s5["p"] > 0.9            # reliable but below the 0.05 SESOI
    assert rules.boot(np.full((10, 2), 0.5), n_boot=500)["p"] > 0                          # never exactly 0


def test_two_way_bootstrap_widens_when_replicate_noise_is_large():
    rng = np.random.default_rng(1)
    base = rng.normal(0.05, 0.005, (80, 1))
    stable = base + rng.normal(0, 0.001, (80, 4))
    shifty = base + rng.normal(0, 0.05, (1, 4))             # replicate-level (training) noise shared by all operating points
    w = lambda x: (lambda s: s["hi90"] - s["lo90"])(rules.boot(x, n_boot=3000))
    assert w(shifty) > 3 * w(stable)


def test_nan_is_refused_and_iu_takes_the_max_p():
    with pytest.raises(ValueError):
        rules.boot(np.array([0.1, np.nan]))
    rng = np.random.default_rng(2)
    big, tiny = rng.normal(0.08, 0.01, (60, 2)), rng.normal(0.001, 0.01, (60, 2))
    r = rules.intersection_union({"P1": big, "P2": tiny}, n_boot=2000)
    assert r["p"] > 0.05 and r["stats"]["P1"]["p"] < 0.01


def test_tost_negligible():
    rng = np.random.default_rng(3)
    assert rules.tost_negligible(rng.normal(0.0, 0.005, (80, 3)), n_boot=2000)["negligible"]
    assert not rules.tost_negligible(rng.normal(0.04, 0.005, (80, 3)), n_boot=2000)["negligible"]
