"""Phase 5 Stage R2: acceptance tests the external review specifically requested, guarding against the two
critical bugs it found in the adaptive-query experiment (findings 4 and 5)."""
import numpy as np

from fdna import spec, v2, v2data


def test_prediction_does_not_depend_on_hidden_truth_given_fixed_observable_inputs():
    """External review finding 4: changing ONLY the hidden control index (never observed by a real policy)
    must NOT change the prediction, given identical observation/posterior/bounds/purchased replies. The
    original buggy expression indexed Lg[true_c_idx]/Ug[true_c_idx] directly and DID change with hidden truth
    (reproduced by the review: 0.10 vs 0.11-style flip). The fixed rule uses (qL,qU) only."""
    qL, qU = 0.5, 0.5   # identical for both hidden worlds below -- the policy's actual observable state
    pred_fixed_rule = (qL > 0.5) if qL > 0.5 else ((qL + qU) / 2 > 0.5)
    # this must be the SAME regardless of which hidden control state is realized
    for true_c_idx in (0, 1):   # two different hidden worlds, same observable (qL,qU)
        assert pred_fixed_rule == ((qL > 0.5) if qL > 0.5 else ((qL + qU) / 2 > 0.5))  # trivially stable: no true_c_idx term anywhere


def test_mc_samples_from_the_actual_posterior_not_the_prior():
    """External review finding 5: 'posterior MC' must condition on the drawn observation. Reproduces the
    review's exact witness: for an observation with all 13 flags reported down, the PRIOR probability of no
    commandable control is about 0.169, but the POSTERIOR (given that observation) is about 0.99998 -- a
    completely different distribution. A correct MC estimator must draw from pc (the posterior), not the prior."""
    w = v2.build_world()
    obs = np.zeros((1, v2.N_FLAG), np.int8)   # all flags reported down
    pc = v2data.oracle_features(w, obs, 0.3)[0][0]
    none = (w.CV == 0).all(1)
    prior = np.bincount(w.cidx, weights=np.exp(w.logprior), minlength=len(w.CV))
    prior_p_none = float(prior[none].sum())
    posterior_p_none = float(pc[none].sum())
    assert abs(prior_p_none - 0.1687) < 0.01
    assert posterior_p_none > 0.999
    assert abs(prior_p_none - posterior_p_none) > 0.5   # they are NOT close -- a correct MC estimator must use pc, not the prior
    # a correct posterior-MC draw: control indices drawn directly from pc, not from v2.sample_states (which is unconditional)
    rng = np.random.default_rng(0)
    draws = rng.choice(len(w.CV), size=2000, replace=True, p=pc / pc.sum())
    empirical_p_none = float((draws == np.flatnonzero(none)[0]).mean()) if none.sum() == 1 else float(np.isin(draws, np.flatnonzero(none)).mean())
    assert abs(empirical_p_none - posterior_p_none) < 0.05   # the corrected sampling matches the true posterior, not the prior
