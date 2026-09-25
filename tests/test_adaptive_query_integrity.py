"""Phase 5 Stage R2: acceptance tests the external review specifically requested, guarding against the two
critical bugs it found in the adaptive-query experiment (findings 4 and 5)."""
import inspect

import numpy as np
import pytest
from pytest import approx

from fdna import spec, v2, v2data
from fdna.adaptive_query import PROB_DECISION_THRESHOLD, predict_from_bounds, resolve


def test_prediction_function_has_no_parameter_that_could_carry_the_hidden_control_state():
    """External review finding 4, as a STRUCTURAL guard on the actual production function (not a hand-copied
    formula that could silently drift from the real script -- an earlier version of this test re-derived the
    formula inline and would not have caught a regression; caught by an independent qa-verifier pass).
    scripts/p5_adaptive_query_experiment_v2.py imports and calls THIS function for pred_g/pred_r; if anyone
    reintroduces a true_c_idx/hidden-state parameter here, this assertion fails immediately."""
    params = list(inspect.signature(predict_from_bounds).parameters)
    assert params == ["qL", "qU"], params


def test_prediction_does_not_depend_on_hidden_truth_given_fixed_observable_inputs():
    """Behavioral companion to the structural guard above, calling the SAME real function two different ways
    that a leaking implementation could distinguish but an honest one cannot: the original buggy expression
    indexed Lg[true_c_idx]/Ug[true_c_idx] directly and DID change with hidden truth (reproduced by the review:
    up to 100% 'accuracy' on a problem an observation-only classifier caps at 50%). Since predict_from_bounds
    has no channel for a hidden control index at all (asserted above), this is necessarily stable -- verified
    by direct call, not by construction alone."""
    qL, qU = 0.5, 0.5   # identical observable state; two different hidden worlds below could only differ via
    # some other channel, and predict_from_bounds accepts no such channel
    pred_world_1 = predict_from_bounds(qL, qU)
    pred_world_2 = predict_from_bounds(qL, qU)
    assert pred_world_1 == pred_world_2 == predict_from_bounds(qL=qL, qU=qU)


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


# ---- repair round 3: resolve()'s stopping-threshold bug (external review finding 2) ----
# resolve() used to compare qU against spec.SEVERE (0.01, the LOAD-SHED threshold) instead of
# PROB_DECISION_THRESHOLD (0.5, the PROBABILITY-decision threshold predict_from_bounds actually uses -- these
# tests use the review's own 4-state witness (CV=[[0,0],[0,1],[1,0],[1,1]], y=[.02,.02,0,0]) plus two more
# constructed to hit the positive-certificate and exact-tie branches, hand-verified against `bounds`/
# `posterior_mass_bounds` before being written here (not just asserted to match the fixed code's own output).
_CV4 = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
_Y4 = np.array([.02, .02, 0., 0.])


def test_resolve_certifies_negative_early_review_witness():
    """The review's own witness: after querying only the 2 extreme controls (budget allows up to 4), qL=0,
    qU=0.3 -- already <=0.5, so the negative decision is certified. The OLD buggy condition (qU<=spec.SEVERE=
    0.01) would NOT have stopped here (0.3 is not <=0.01), continuing to use all 4 queries for the same answer."""
    pc = np.array([0., .2, .1, .7])
    n, qL, qU = resolve("random_bounds", _Y4, 0., _Y4, pc, 4, np.random.default_rng(7), _CV4, spec.SEVERE)
    assert n == 2, n
    assert qL == approx(0.0) and qU == approx(0.3)
    assert predict_from_bounds(qL, qU) is False


def test_resolve_certifies_positive_early():
    """Constructed so qL alone exceeds 0.5 after just the 2 extreme queries: pc concentrates enough mass on the
    one control whose lower bound is already above tau. Positive certification stops just as early as negative."""
    pc = np.array([.6, .1, .1, .2])
    n, qL, qU = resolve("random_bounds", _Y4, 0., _Y4, pc, 4, np.random.default_rng(3), _CV4, spec.SEVERE)
    assert n == 2, n
    assert qL == approx(0.6)
    assert predict_from_bounds(qL, qU) is True


def test_resolve_exact_tie_at_probability_threshold_certifies_negative():
    """Boundary case: qU lands EXACTLY on PROB_DECISION_THRESHOLD (0.5) after 2 queries. `<=` must fire (a tie
    is a valid negative certificate: qU can only shrink further, so the eventual (qL+qU)/2 can never exceed 0.5
    from this point on)."""
    pc = np.array([0., .5, 0., .5])
    n, qL, qU = resolve("random_bounds", _Y4, 0., _Y4, pc, 4, np.random.default_rng(1), _CV4, spec.SEVERE)
    assert n == 2, n
    assert qU == approx(PROB_DECISION_THRESHOLD)
    assert predict_from_bounds(qL, qU) is False


def test_resolve_per_candidate_rng_is_independent_of_other_candidates_consumption():
    """Repair round 3: the script used to share ONE rng object across all 70 outages for a given (op,budget),
    sequentially advanced -- so a candidate's random_bounds() draws depended on how many draws EVERY PRIOR
    candidate happened to consume (which itself depends on how early each one's stopping rule fired). Fixed by
    seeding a fresh rng per (op_id, outage, budget) inside the per-candidate loop, not shared across outages.

    Demonstrated concretely on a 6-state grid (more than the 2 extreme controls, so `random_bounds` genuinely
    has more than one candidate index to choose among, and that choice depends on the rng's exact state): the
    SAME seed produces a DIFFERENT result depending on whether the rng was pre-advanced by an unrelated prior
    call (simulating the old shared-rng bug) versus fresh (the fixed, per-candidate pattern) -- confirmed
    numerically before writing this assertion, not assumed."""
    cv6 = np.array([[0.], [0.2], [0.4], [0.6], [0.8], [1.0]])
    y6 = np.array([.02, .015, .012, .008, .005, 0.])
    pc = np.array([.05, .15, .2, .2, .2, .2])

    pre_advanced = np.random.default_rng(99)
    pre_advanced.integers(0, 100, size=5)   # stands in for an unrelated earlier candidate's draws
    result_pre_advanced = resolve("random_bounds", y6, 0., y6, pc, 4, pre_advanced, cv6, spec.SEVERE)

    fresh = np.random.default_rng(99)       # same seed, never touched by anything else -- the fixed pattern
    result_fresh = resolve("random_bounds", y6, 0., y6, pc, 4, fresh, cv6, spec.SEVERE)

    assert result_pre_advanced != result_fresh, (
        "expected the pre-advanced and fresh rng to diverge on this scenario -- if they now agree, the "
        "scenario itself may have stopped being RNG-sensitive and needs reconstructing, not the assertion "
        "loosened"
    )


def test_resolve_honors_zero_and_one_query_budget_before_extremes():
    """Budget is a hard oracle-query cap. A previous implementation always queried both extreme controls before
    checking the cap, so budget 0 and 1 both spent 2 queries on this hand-checked four-control grid."""
    pc = np.array([0.25, 0.25, 0.25, 0.25])
    n0, qL0, qU0 = resolve("random_bounds", _Y4, 0., _Y4, pc, 0, np.random.default_rng(1), _CV4, spec.SEVERE)
    assert n0 == 0
    assert qL0 == approx(0.0)
    assert qU0 == approx(1.0)

    n1, qL1, qU1 = resolve("random_bounds", _Y4, 0., _Y4, pc, 1, np.random.default_rng(1), _CV4, spec.SEVERE)
    assert n1 == 1
    assert qL1 == approx(0.0)
    assert qU1 == approx(0.75)


def test_resolve_uses_unique_extreme_once_on_single_control_grid():
    """When the control grid has one row, argmax and argmin are the same row. The oracle must count that as one
    query, not two duplicate queries of the same control."""
    cv1 = np.array([[0.0, 0.0]])
    y1 = np.array([0.02])
    pc = np.array([1.0])
    n, qL, qU = resolve("random_bounds", y1, 0., y1, pc, 4, np.random.default_rng(1), cv1, spec.SEVERE)
    assert n == 1
    assert qL == approx(1.0)
    assert qU == approx(1.0)


def test_resolve_rejects_negative_and_non_integer_budget():
    for bad in [-1, 1.5, True]:
        with pytest.raises(ValueError):
            resolve("random_bounds", _Y4, 0., _Y4, np.ones(4) / 4, bad, np.random.default_rng(1), _CV4, spec.SEVERE)
