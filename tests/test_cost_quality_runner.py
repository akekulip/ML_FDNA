import numpy as np

from scripts.p5_cost_quality import execute_policy


CV = np.array([[0.], [1.]])
OUTAGES = [(0, 1, 2), (0, 1, 3)]


def test_infeasible_construction_never_queries_target():
    def forbidden(*args):
        raise AssertionError('infeasible policy queried a label')
    out = execute_policy('cv', forbidden, 600, OUTAGES, CV, np.array([.5, .5]),
                         budget=3, construction_cost=4, seed=0, proxy=np.zeros((2, 2)))
    assert out['feasible'] is False
    assert out['target_queries'] == 0


def test_mc_support_enumeration_is_charged_and_exact():
    seen = []
    def backend(op, outage, index):
        seen.append((op, outage, index))
        return [.02, 0.][index]
    out = execute_policy('mc', backend, 600, OUTAGES, CV, np.array([.25, .75]),
                         budget=4, construction_cost=0, seed=0)
    np.testing.assert_allclose(out['scores'], [.25, .25])
    assert out['target_queries'] == 4
    assert len(seen) == 4
    assert out['mode'] == 'posterior_support_enumeration'


def test_global_budget_is_respected_without_severe_count_input():
    def backend(op, outage, index):
        return .02 if index == 0 else 0.
    out = execute_policy('mc', backend, 600, OUTAGES, CV, np.array([.5, .5]),
                         budget=3, construction_cost=0, seed=0)
    assert out['feasible']
    assert out['draws_per_candidate'] == 1
    assert out['target_queries'] == 2
    assert len(out['scores']) == 2
