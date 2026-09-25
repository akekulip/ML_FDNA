"""Independent fixtures for endpoint grouping and development decisions."""
import numpy as np

from fdna.cost_quality import screening_metrics, select_gate, paired_summary


def test_rprecision_uses_severe_count_only_in_evaluator():
    result = screening_metrics(np.array([1, 0, 1, 0], bool),
                               np.array([.8, .9, .7, .1]),
                               np.arange(4), np.array([.8, .2, .7, .1]))
    assert result['rprecision'] == .5
    assert np.isclose(result['probability_mse'], .1225)
    assert result['precision_20'] == 0.


def test_no_severe_is_explicit_not_perfect_screening():
    result = screening_metrics(np.zeros(4, bool), np.zeros(4), np.arange(4), np.zeros(4))
    assert result['rprecision'] is None
    assert result['recall_20'] is None
    assert result['n_severe'] == 0
    assert result['precision_20'] == 0.


def test_gate_requires_two_cells_and_both_baselines_at_same_budget():
    def row(cell, comparator, mean, lower, budget=140):
        return dict(regime='warm', method='cv_full', budget=budget, cell=cell,
                    comparator=comparator, mean=mean, lo95=lower, n_ops=60)
    rows = [row(c, b, .03, .02) for c in ('P1_v2b', 'P2_v2c') for b in ('mc', 'proxy_full')]
    assert select_gate(rows, .01)['warm']['status'] == 'candidate'
    rows[-1]['lo95'] = -.01
    assert select_gate(rows, .01)['warm']['status'] == 'no_go'
    assert select_gate(rows[:-1], .01)['warm']['status'] == 'no_go'


def test_paired_bootstrap_keeps_operating_points_as_clusters():
    # Repeated sampling seeds within an op must not masquerade as extra ops.
    x = np.tile(np.array([[-.1], [.3]]), (1, 4))
    a = paired_summary(x, n_boot=1000)
    b = paired_summary(x[:, :1], n_boot=1000)
    assert np.isclose(a['mean'], .1)
    assert a['n_ops'] == 2
    assert a['lo95'] == b['lo95']
