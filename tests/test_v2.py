import numpy as np
import pytest

from fdna import v2
from fdna.blocks import LOCK_DIR, open_block


@pytest.fixture(scope="module")
def w():
    return v2.build_world()


def test_prior_normalised_and_vectors(w):
    assert np.exp(w.logprior).sum() == pytest.approx(1.0)
    assert len(w.CV) == 1024 and w.cidx.max() == 1023


def test_posterior_matches_bruteforce_and_sums_to_one(w):
    rng = np.random.default_rng(0)
    for s in (0.0, 0.2):
        st = v2.sample_states(w, rng, 20)
        obs = v2.emit(w, st, 0.3, s, rng)
        post = v2.posterior(w, obs, s)
        np.testing.assert_allclose(post.sum(1), 1.0, atol=1e-5)
        # brute force for one observation: prior x product of emission probabilities over observed flags
        o = obs[0]
        lik = np.ones(len(w.X))
        for j in range(v2.N_FLAG):
            if o[j] < 0:
                continue
            t = w.T[:, j]
            e = np.where(o[j] == 1, np.where(t, 1.0, s), np.where(t, 0.0, 1 - s))
            lik *= e
        ref = np.exp(w.logprior) * lik
        ref /= ref.sum()
        np.testing.assert_allclose(post[0], ref, atol=1e-5)
        assert post[0][st[0]] > 0  # truth always has positive posterior mass


def test_no_missing_no_stale_pins_unit_flags(w):
    rng = np.random.default_rng(1)
    st = v2.sample_states(w, rng, 50)
    obs = v2.emit(w, st, 0.0, 0.0, rng)
    post = v2.posterior(w, obs, 0.0)
    # every state with posterior mass must reproduce all 13 observed flags
    for b in range(5):
        support = post[b] > 1e-9
        assert (w.T[support] == (obs[b] == 1)).all()


def test_control_vector_posterior_marginals(w):
    rng = np.random.default_rng(2)
    st = v2.sample_states(w, rng, 10)
    obs = v2.emit(w, st, 0.1, 0.2, rng)
    pc = v2.posterior_over_controls(w, v2.posterior(w, obs, 0.2))
    np.testing.assert_allclose(pc.sum(1), 1.0, atol=1e-4)
    m = v2.control_marginals(w, pc)
    assert m.shape == (10, 5) and (m >= -1e-6).all() and (m <= 1 + 1e-6).all()


def test_block_lock_refuses_reuse(tmp_path, monkeypatch):
    import fdna.blocks as b
    monkeypatch.setattr(b, "LOCK_DIR", tmp_path)
    assert list(open_block("confirm", "H_test")) == list(range(400, 480))
    with pytest.raises(RuntimeError):
        open_block("confirm", "H_test")
    assert list(open_block("replicate", "H_test"))[0] == 500
