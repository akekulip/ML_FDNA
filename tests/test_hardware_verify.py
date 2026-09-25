import numpy as np

from scripts.p5_hardware_verify import MemoizedScalarBackend, compare_event_labels, run_live_policy


def test_live_policy_charges_each_policy_but_memoizes_physical_solves():
    calls = []

    def backend(op_id, outage, control_idx):
        calls.append((op_id, outage, control_idx))
        return 0.25 + control_idx

    memo = MemoizedScalarBackend(backend)
    outages = [(1, 2, 3)]
    CV = np.array([[0.0]])
    pc = np.array([1.0])

    first = run_live_policy("mc", memo, 7, outages, CV, pc, budget=1, seed=100)
    second = run_live_policy("mc", memo, 7, outages, CV, pc, budget=1, seed=100)

    assert first["target_queries"] == 1
    assert second["target_queries"] == 1
    assert memo.unique_solves == 1
    assert memo.requests == 2
    assert calls == [(7, (1, 2, 3), 0)]
    assert first["events"] == [{"op_id": 7, "outage": (1, 2, 3), "control_idx": 0, "value": 0.25}]


def test_compare_event_labels_checks_values_and_severity_classes():
    live = [
        {"op_id": 7, "outage": (1, 2, 3), "control_idx": 0, "value": 0.02},
        {"op_id": 7, "outage": (1, 2, 3), "control_idx": 1, "value": 0.20},
    ]
    frozen = [
        {"op_id": 7, "outage": [1, 2, 3], "control_idx": 0, "value": 0.02000000001},
        {"op_id": 7, "outage": [1, 2, 3], "control_idx": 1, "value": 0.0},
    ]

    report = compare_event_labels(live, frozen, tolerance=1e-7, severe_threshold=0.01)

    assert report["compared"] == 2
    assert report["missing_live"] == 0
    assert report["missing_frozen"] == 0
    assert report["value_mismatches"] == 1
    assert report["severity_mismatches"] == 1
    assert report["max_abs_error"] == 0.2
