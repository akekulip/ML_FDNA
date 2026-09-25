"""Phase 5 repair round 3: the control-variate estimator implemented as an actual policy in
scripts/p5_adaptive_query_experiment_v3.py. p_hat_beta = beta*E_p[h0] + mean_j(h(c_j) - beta*h0(c_j)), reusing
the SAME draws plain posterior MC already uses. These tests exercise the formula and the full-panel diagnostic
directly (reimplemented inline, matching the script's own logic exactly -- the script itself is not imported,
consistent with how other adaptive-query scripts in this repo are tested via extracted formulas/witnesses
rather than script import, since the scripts execute top-level data loading on import)."""
import numpy as np


def p_hat_beta(beta, E_p_h0, h_at_draws, h0_at_draws):
    return beta * E_p_h0 + float((h_at_draws - beta * h0_at_draws).mean())


def test_beta_zero_reduces_exactly_to_plain_posterior_mc():
    rng = np.random.default_rng(0)
    h_at_draws = rng.integers(0, 2, size=16).astype(np.float64)
    h0_at_draws = rng.integers(0, 2, size=16).astype(np.float64)
    E_p_h0 = 0.37  # arbitrary -- must not matter at beta=0
    mc_score = float(h_at_draws.mean())
    assert p_hat_beta(0.0, E_p_h0, h_at_draws, h0_at_draws) == mc_score


def cv_diagnostic_row(pn, h, h0):
    """Mirrors scripts/p5_adaptive_query_experiment_v3.py::cv_diagnostic_row exactly (see that function for the
    real per-candidate version operating on y_true_grid/g2grid; this takes h/h0 directly for a unit test)."""
    Eh, Eh0 = float((pn * h).sum()), float((pn * h0).sum())
    Ehh0 = float((pn * h * h0).sum())
    var_h = float((pn * h ** 2).sum() - Eh ** 2); var_h0 = float((pn * h0 ** 2).sum() - Eh0 ** 2)
    cov = Ehh0 - Eh * Eh0
    var_diff = var_h + var_h0 - 2 * cov
    if var_h <= 1e-12:
        category = "zero_variance_harmed_by_proxy" if var_diff > 1e-12 else "zero_variance_tie"
        ratio = None
    else:
        category, ratio = "nondegenerate", var_diff / var_h
    return dict(var_h=var_h, var_diff=var_diff, category=category, ratio=ratio)


def test_reviews_exact_zero_variance_harm_example_is_categorized_not_dropped():
    """The review's own worked example (section 3): posterior [0.5,0.5], true indicators [0,0] (plain MC has
    ZERO variance -- the outcome is certain), proxy indicators [0,1] (g2's threshold call disagrees on the
    second state). v2's diagnostic silently EXCLUDED this exact case (var_h<=1e-12 filtered it out before
    computing a ratio); the v3 panel must categorize it as real, measurable harm, never drop it."""
    pn = np.array([0.5, 0.5]); h = np.array([0., 0.]); h0 = np.array([0., 1.])
    row = cv_diagnostic_row(pn, h, h0)
    assert row["var_h"] == 0.0
    assert row["var_diff"] == 0.25
    assert row["category"] == "zero_variance_harmed_by_proxy"
    assert row["ratio"] is None  # undefined (0/0), not silently coerced to a number


def test_zero_variance_tie_when_proxy_agrees_everywhere():
    """Negative companion: if the proxy agrees with truth everywhere the posterior has support, var_diff is
    also 0 -- a genuine tie, not harm. Distinguishes the two zero-variance categories."""
    pn = np.array([0.5, 0.5]); h = np.array([0., 0.]); h0 = np.array([0., 0.])
    row = cv_diagnostic_row(pn, h, h0)
    assert row["category"] == "zero_variance_tie"
    assert row["ratio"] is None


def test_multiplicity_is_not_silently_deduplicated():
    """mc_idx is drawn WITH replacement; h_at_draws.mean() must weight by draw multiplicity, not by the count
    of DISTINCT sampled states -- reviewing requirement 4: 'deduplicating evaluations is valid; averaging
    distinct sampled states uniformly generally changes the posterior estimator.'"""
    mc_idx = np.array([0, 0, 0, 1])  # state 0 drawn 3x, state 1 drawn 1x
    y_true_grid = np.array([0.0, 1.0])  # state 0 -> not severe, state 1 -> severe (using thr=0.5 directly on y)
    h_at_draws = (y_true_grid[mc_idx] > 0.5).astype(np.float64)
    mean_with_multiplicity = float(h_at_draws.mean())            # 1/4 = 0.25
    mean_over_unique_states = float(np.unique(y_true_grid[mc_idx] > 0.5).astype(np.float64).mean())  # (0+1)/2 = 0.5
    assert mean_with_multiplicity == 0.25
    assert mean_with_multiplicity != mean_over_unique_states
