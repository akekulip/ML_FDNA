from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from fdna.query_budget import BudgetExceeded, BudgetedOracle, bound_probability, estimate_probability, g2_proxy


class FixedRng:
    def __init__(self, draws):
        self.draws = np.asarray(draws, dtype=int)

    def choice(self, n, size, replace, p):
        assert replace is True
        assert n == len(p)
        assert size == len(self.draws)
        return self.draws.copy()


def test_budgeted_oracle_canonicalizes_cache_and_preserves_initial_cache_copy():
    backing_calls = []
    initial = {(7, (2, 0), 3): 0.25}
    oracle = BudgetedOracle(
        lambda op_id, outage, control_idx: backing_calls.append((op_id, outage, control_idx)) or 0.9,
        budget=1,
        initial_cache=initial,
    )
    initial[(7, (0, 2), 3)] = 0.75

    assert oracle.query(7, (0, 2), 3) == 0.25
    assert oracle.query(7, [2, 0], 3) == 0.25
    assert backing_calls == []
    assert oracle.spent == 0
    assert oracle.remaining == 1
    assert oracle.cache == {(7, (0, 2), 3): 0.25}


def test_budgeted_oracle_charges_unique_misses_and_rejects_bad_inputs():
    calls = []

    def backend(op_id, outage, control_idx):
        calls.append((op_id, outage, control_idx))
        return float(op_id + sum(outage) + control_idx)

    oracle = BudgetedOracle(backend, budget=1)

    assert oracle.query(5, (4, 1), 2) == 12.0
    assert oracle.query(5, (1, 4), 2) == 12.0
    assert oracle.events == [{"op_id": 5, "outage": (1, 4), "control_idx": 2, "value": 12.0}]
    assert oracle.spent == 1
    assert oracle.remaining == 0
    assert calls == [(5, (1, 4), 2)]

    with pytest.raises(BudgetExceeded):
        oracle.query(5, (1,), 2)
    assert calls == [(5, (1, 4), 2)]

    for bad in [(-1,), (1, 1), (1.5,), ("1",)]:
        with pytest.raises(ValueError):
            oracle.query(0, bad, 0)
    with pytest.raises(ValueError):
        oracle.query(-1, (1,), 0)
    with pytest.raises(ValueError):
        oracle.query(0, (1,), -1)


def test_budgeted_oracle_requires_finite_labels():
    oracle = BudgetedOracle(lambda op_id, outage, control_idx: math.nan, budget=1)
    with pytest.raises(ValueError):
        oracle.query(0, (1,), 0)
    assert oracle.cache == {}
    assert oracle.events == []


def test_g2_proxy_uses_only_lower_order_queries_and_shares_them_across_triples():
    calls = []

    def backend(op_id, outage, control_idx):
        assert len(outage) < 3, "proxy must not query hidden triple truth"
        calls.append((op_id, outage, control_idx))
        return 10.0 * sum(outage) + control_idx + 0.1 * len(outage)

    oracle = BudgetedOracle(backend, budget=100)
    CV = np.array([[0.0, 0.0], [1.0, 0.0]])
    pc = np.array([0.5, 0.5])

    grid, anchors = g2_proxy(oracle, 3, [(2, 0, 1), (3, 1, 2)], CV, pc)

    expected_first_control0 = (10.0 * 1 + 0.2) + (10.0 * 2 + 0.2) + (10.0 * 3 + 0.2)
    expected_first_control0 -= (0.1 + 10.1 + 20.1)
    assert grid.shape == (2, 2)
    assert grid[0, 0] == pytest.approx(expected_first_control0)
    assert anchors.tolist() == [0, 1]

    unique_lower_order_keys = {
        (3, tuple(combo), control_idx)
        for triple in [(0, 1, 2), (1, 2, 3)]
        for order in (1, 2)
        for combo in itertools.combinations(triple, order)
        for control_idx in (0, 1)
    }
    assert oracle.spent == len(unique_lower_order_keys)
    assert set(calls) == unique_lower_order_keys


def test_g2_proxy_selects_deterministic_anchors_and_interpolates_l1_ties_to_lowest_index():
    def backend(op_id, outage, control_idx):
        return float(100 * sum(outage) + control_idx)

    oracle = BudgetedOracle(backend, budget=100)
    CV = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [4.0, 0.0]])
    pc = np.array([0.1, 0.4, 0.4, 0.1])

    grid, anchors = g2_proxy(oracle, 1, [(0, 1, 2)], CV, pc, n_anchors=2)

    assert anchors.tolist() == [1, 2]
    assert grid.shape == (1, 4)
    assert grid[0, 0] == grid[0, 1]
    assert grid[0, 3] == grid[0, 1]
    assert oracle.spent == 12


def test_estimate_probability_direct_mc_preserves_multiplicities_and_caches_duplicates():
    calls = []

    def backend(op_id, outage, control_idx):
        calls.append(control_idx)
        return [0.0, 1.0][control_idx]

    oracle = BudgetedOracle(backend, budget=10)

    got = estimate_probability(
        oracle,
        0,
        (5, 4),
        np.array([0.25, 0.75]),
        n_draws=4,
        rng=FixedRng([0, 0, 0, 1]),
        tau=0.5,
    )

    assert got == 0.25
    assert calls == [0, 1]
    assert oracle.spent == 2


def test_estimate_probability_calls_query_once_per_unique_sampled_control():
    class CountingOracle(BudgetedOracle):
        def __init__(self):
            super().__init__(lambda op_id, outage, control_idx: [0.0, 1.0][control_idx], budget=10)
            self.query_calls = []

        def query(self, op_id, outage, control_idx):
            self.query_calls.append(control_idx)
            return super().query(op_id, outage, control_idx)

    oracle = CountingOracle()

    got = estimate_probability(
        oracle,
        0,
        (5,),
        np.array([0.5, 0.5]),
        n_draws=5,
        rng=FixedRng([1, 1, 1, 0, 1]),
        tau=0.5,
    )

    assert got == 0.8
    assert oracle.query_calls == [0, 1]


def test_estimate_probability_beta_zero_is_plain_mc_even_with_proxy():
    oracle = BudgetedOracle(lambda op_id, outage, control_idx: [0.0, 1.0][control_idx], budget=10)
    proxy = np.array([100.0, -100.0])

    got = estimate_probability(
        oracle,
        0,
        (1,),
        np.array([0.2, 0.8]),
        n_draws=5,
        rng=FixedRng([1, 1, 0, 1, 0]),
        proxy=proxy,
        beta=0.0,
        tau=0.5,
    )

    assert got == 0.6


def test_estimate_probability_matches_hand_enumerated_unbiased_expectation():
    pc = np.array([0.25, 0.75])
    truth = np.array([0.0, 1.0])
    proxy = np.array([1.0, 0.0])
    exact_probability = float(np.dot(pc, truth > 0.5))
    weighted_outputs = []

    for draw in [0, 1]:
        oracle = BudgetedOracle(lambda op_id, outage, control_idx: truth[control_idx], budget=10)
        estimate = estimate_probability(
            oracle,
            0,
            (9,),
            pc,
            n_draws=1,
            rng=FixedRng([draw]),
            proxy=proxy,
            beta=0.5,
            tau=0.5,
        )
        weighted_outputs.append(pc[draw] * estimate)

    assert sum(weighted_outputs) == pytest.approx(exact_probability)


def test_estimate_probability_uses_exact_proxy_expectation_and_can_report_harm_raw():
    proxy = np.array([0.0, 1.0])
    pc = np.array([0.5, 0.5])

    oracle = BudgetedOracle(lambda op_id, outage, control_idx: 0.0, budget=10)
    got = estimate_probability(
        oracle,
        0,
        (1,),
        pc,
        n_draws=2,
        rng=FixedRng([1, 1]),
        proxy=proxy,
        beta=1.0,
        tau=0.5,
    )

    assert got == -0.5


def test_estimate_probability_validates_probability_and_proxy_dimensions_before_querying():
    oracle = BudgetedOracle(lambda op_id, outage, control_idx: 1.0, budget=10)

    bad_cases = [
        dict(pc=np.array([0.5, -0.5]), proxy=None),
        dict(pc=np.array([0.5, 0.6]), proxy=None),
        dict(pc=np.array([[1.0]]), proxy=None),
        dict(pc=np.array([1.0]), proxy=np.array([0.0, 1.0])),
    ]
    for case in bad_cases:
        with pytest.raises(ValueError):
            estimate_probability(oracle, 0, (1,), case["pc"], n_draws=1, rng=FixedRng([0]), proxy=case["proxy"])
    assert oracle.spent == 0


def test_estimator_respects_budget_before_backend_on_duplicate_sampling():
    oracle = BudgetedOracle(lambda op_id, outage, control_idx: 1.0, budget=1)

    with pytest.raises(BudgetExceeded):
        estimate_probability(
            oracle,
            0,
            (1,),
            np.array([0.5, 0.5]),
            n_draws=2,
            rng=FixedRng([0, 1]),
            tau=0.5,
        )
    assert oracle.spent == 1


def test_bound_probability_uses_oracle_queries_only_and_respects_query_cap():
    calls = []
    CV = np.array([[0.0, 0.0], [1.0, 1.0]])
    pc = np.array([0.6, 0.4])

    def backend(op_id, outage, control_idx):
        assert outage == (7, 8)
        calls.append(control_idx)
        return [1.0, 0.0][control_idx]

    oracle = BudgetedOracle(backend, budget=10)

    prior_midpoint = bound_probability(
        oracle, 0, (8, 7), CV, pc, max_queries=0, floor=0.0, rng=FixedRng([0]), tau=0.5
    )
    assert prior_midpoint == 0.5
    assert calls == []

    one_query_midpoint = bound_probability(
        oracle, 0, (8, 7), CV, pc, max_queries=1, floor=0.0, rng=FixedRng([0]), tau=0.5
    )
    assert one_query_midpoint == 0.3
    assert calls == [1]
    assert oracle.spent == 1
