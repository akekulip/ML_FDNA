import numpy as np

from fdna import rules


def test_holm_is_monotone_and_conservative():
    adj = rules.holm({"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5})
    assert adj["a"] == 0.004 and adj["b"] <= adj["c"] <= adj["d"]
    assert all(adj[k] >= p for k, p in {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.5}.items())


def test_claim_needs_sesoi_and_all_components():
    rng = np.random.default_rng(0)
    big = rng.normal(0.08, 0.02, 80)
    small = rng.normal(0.02, 0.02, 80)
    neg = rng.normal(-0.02, 0.02, 80)
    assert rules.claim({"vs_comp": big, "vs_i6": big, "vs_i7": big}, "vs_comp")["effect_ok"]
    assert not rules.claim({"vs_comp": small, "vs_i6": big}, "vs_comp")["effect_ok"]      # below SESOI
    assert not rules.claim({"vs_comp": big, "vs_i7": neg}, "vs_comp")["effect_ok"]         # loses to a control
    v = rules.verdicts({"X": rules.claim({"c": big}, "c"), "Y": rules.claim({"c": neg}, "c")})
    assert v["X"]["supported"] and not v["Y"]["supported"]
