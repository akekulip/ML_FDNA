import gzip
import json

import pytest

from scripts.p5_cost_quality_replay import (
    audit_trace_dir,
    load_selected_trace_details,
    load_trace_records_for_replay,
    load_traces,
)


def _write_jsonl_gz(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_raw_gz(path, lines):
    with gzip.open(path, "wt") as f:
        for line in lines:
            f.write(line + "\n")


def _event(outage, control_idx, value=0.25):
    return {"op_id": 600, "outage": list(outage), "control_idx": control_idx, "value": value}


def _metric_row(**overrides):
    row = {
        "regime": "warm",
        "cell": "P1_v2b",
        "op_id": 600,
        "replicate": 0,
        "budget": 2,
        "method": "mc",
        "construction_trace": None,
        "execution_trace": 20,
        "snapshot_trace": 30,
        "feasible": True,
        "target_queries": 2,
        "total_queries": 2,
        "n_candidates": 4,
        "n_severe": 2,
        "rprecision": 1.0,
        "probability_mse": 0.01,
        "clipped_probability_mse": 0.01,
        "recall_10": 0.5,
        "precision_10": 1.0,
        "recall_20": 0.5,
        "precision_20": 1.0,
        "recall_40": 1.0,
        "precision_40": 1.0,
    }
    row.update(overrides)
    return row


def _fixture(tmp_path, traces, metrics):
    _write_jsonl_gz(tmp_path / "traces.jsonl.gz", traces)
    _write_jsonl_gz(tmp_path / "metrics.jsonl.gz", metrics)
    return tmp_path


def test_auditor_accepts_cold_construction_plus_target_and_warm_target_only(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 10, "kind": "construction", "op_id": 600, "cell": "P1_v2b", "proxy": "sparse2",
             "anchors": [0], "spent": 2, "events": [_event((0,), 0), _event((1, 2), 0)]},
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.9, 0.8, 0.1, 0.0], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P1_v2b",
             "labels": [True, True, False, False], "exact_probability": [0.8, 0.7, 0.2, 0.1],
             "keys": [0.4, 0.3, 0.2, 0.1]},
        ],
        [
            _metric_row(regime="cold", budget=4, construction_trace=10, total_queries=4),
            _metric_row(regime="warm", budget=2, construction_trace=10, total_queries=2),
        ],
    )

    report = audit_trace_dir(trace_dir)

    assert report["metrics_records"] == 2
    assert report["trace_records"] == 3
    assert report["failures"] == []
    assert report["metrics_reconstructed"] == 2
    assert report["metric_reconstruction_cache_entries"] == 1
    assert report["metric_reconstruction_cache_hits"] == 1
    assert report["regime_summary"]["cold"]["evaluations"] == 1
    assert report["regime_summary"]["cold"]["total_unique_keys"] == 4
    assert report["regime_summary"]["warm"]["total_unique_keys"] == 2


def test_auditor_reports_duplicate_events_overbudget_and_count_mismatch(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 1, "kind": "construction", "op_id": 600, "cell": "P1_v2b", "proxy": "full",
             "anchors": [0], "spent": 2, "events": [_event((0,), 0), _event((0,), 0)]},
            {"trace_id": 2, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.9, 0.8, 0.1, 0.0], "events": [_event((0, 1, 2), 0)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P1_v2b",
             "labels": [True, True, False, False], "exact_probability": [0.8, 0.7, 0.2, 0.1],
             "keys": [0.4, 0.3, 0.2, 0.1]},
        ],
        [
            _metric_row(regime="cold", budget=2, construction_trace=1, execution_trace=2, total_queries=4),
        ],
    )

    report = audit_trace_dir(trace_dir)

    assert any("duplicate event keys" in failure for failure in report["failures"])
    assert any("spent 2 but has 1 unique event keys" in failure for failure in report["failures"])
    assert any("target_queries 2 but execution trace has 1 unique event keys" in failure for failure in report["failures"])
    assert any("total_queries 4 but cold unique charge is 2" in failure for failure in report["failures"])
    assert any("exceeds budget 2" in failure for failure in report["failures"])


def test_auditor_skips_infeasible_metric_rows_explicitly(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 7, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "cv_sparse2",
             "seed": 100, "available_target_budget": -1, "feasible": False, "target_queries": 0,
             "scores": None, "events": []},
        ],
        [
            {"regime": "cold", "cell": "P1_v2b", "op_id": 600, "replicate": 0, "budget": 1,
             "method": "cv_sparse2", "construction_trace": None, "execution_trace": 7, "feasible": False,
             "target_queries": 0, "total_queries": None},
        ],
    )

    report = audit_trace_dir(trace_dir)

    assert report["failures"] == []
    assert report["skipped_infeasible"] == 1


def test_trace_summary_loader_discards_event_sets_and_values(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 4, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.2], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
        ],
        [],
    )

    traces, failures, summary = load_traces(trace_dir / "traces.jsonl.gz")

    assert failures == []
    assert summary["trace_id_min"] == 4
    assert summary["trace_id_max"] == 4
    assert traces[4]["unique_count"] == 2
    assert traces[4]["order_set"] == [3]
    assert "_unique_keys" not in traces[4]
    assert "_event_values" not in traces[4]
    assert "events" not in traces[4]


def test_auditor_rejects_mixed_or_wrong_event_order(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 5, "kind": "construction", "op_id": 600, "cell": "P1_v2b", "proxy": "full",
             "anchors": [0], "spent": 2, "events": [_event((0,), 0), _event((0, 1, 2), 0)]},
            {"trace_id": 6, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 1, "feasible": True, "target_queries": 1,
             "scores": [0.2], "events": [_event((0, 1), 0)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P1_v2b",
             "labels": [True], "exact_probability": [0.2], "keys": [0.1]},
        ],
        [
            _metric_row(regime="cold", budget=3, construction_trace=5, execution_trace=6, target_queries=1,
                        total_queries=3, n_candidates=1, n_severe=1, probability_mse=0.0,
                        clipped_probability_mse=0.0, recall_10=1.0, precision_10=1.0,
                        recall_20=1.0, precision_20=1.0),
        ],
    )

    report = audit_trace_dir(trace_dir)

    assert any("construction trace has event orders [1, 3]" in failure for failure in report["failures"])
    assert any("execution trace has event orders [2]" in failure for failure in report["failures"])


def test_replay_detail_loader_keeps_warm_construction_values_unloaded(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 10, "kind": "construction", "op_id": 600, "cell": "P1_v2b", "proxy": "full",
             "anchors": [0], "spent": 1, "events": [_event((0,), 0)]},
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 1, "feasible": True, "target_queries": 1,
             "scores": [0.2], "events": [_event((0, 1, 2), 0)]},
        ],
        [
            {"regime": "warm", "cell": "P1_v2b", "op_id": 600, "replicate": 0, "budget": 1, "method": "mc",
             "construction_trace": 10, "execution_trace": 20, "feasible": True, "target_queries": 1,
             "total_queries": 1},
        ],
    )

    summaries, details, selected = load_trace_records_for_replay(
        trace_dir,
        ops=(600,),
        cells=("P1_v2b",),
        budget=1,
        methods=("mc",),
        regimes=("warm",),
        replicate=0,
    )

    assert len(selected) == 1
    assert summaries[10]["unique_count"] == 1
    assert 20 in details
    assert 10 not in details


def test_replay_selection_reuses_validated_trace_index_without_reloading(tmp_path, monkeypatch):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 1, "feasible": True, "target_queries": 1,
             "scores": [0.2], "events": [_event((0, 1, 2), 0)]},
        ],
        [
            {"regime": "warm", "cell": "P1_v2b", "op_id": 600, "replicate": 0, "budget": 1, "method": "mc",
             "construction_trace": None, "execution_trace": 20, "feasible": True, "target_queries": 1,
             "total_queries": 1},
        ],
    )
    trace_index = load_traces(trace_dir / "traces.jsonl.gz")

    import scripts.p5_cost_quality_replay as replay

    monkeypatch.setattr(replay, "load_traces", lambda path: pytest.fail("trace index should be reused"))

    summaries, details, selected = load_trace_records_for_replay(
        trace_dir,
        ops=(600,),
        cells=("P1_v2b",),
        budget=1,
        methods=("mc",),
        regimes=("warm",),
        replicate=0,
        trace_index=trace_index,
    )

    assert selected[0]["execution_trace"] == 20
    assert summaries[20]["unique_count"] == 1
    assert details[20]["event_values"]


def test_selected_detail_loader_avoids_parsing_unselected_canonical_lines_and_falls_back(monkeypatch, tmp_path):
    trace_path = tmp_path / "traces.jsonl.gz"
    selected = {"trace_id": 42, "kind": "execution", "events": [_event((0, 1, 2), 0, value=0.75)]}
    fallback_selected = {"kind": "execution", "trace_id": 43, "events": [_event((0, 1, 2), 1, value=0.5)]}
    unselected = {
        "trace_id": 99,
        "kind": "execution",
        "events": [_event((0, 1, 2), i, value=0.1) for i in range(200)],
    }
    _write_raw_gz(
        trace_path,
        [
            json.dumps(unselected, separators=(",", ":")),
            json.dumps(selected, separators=(",", ":")),
            json.dumps(fallback_selected, separators=(",", ":")),
        ],
    )
    real_loads = json.loads
    loaded_trace_ids = []

    def tracking_loads(line):
        record = real_loads(line)
        loaded_trace_ids.append(record["trace_id"])
        return record

    import scripts.p5_cost_quality_replay as replay

    monkeypatch.setattr(replay.json, "loads", tracking_loads)

    details = load_selected_trace_details(trace_path, {42, 43})

    assert loaded_trace_ids == [42, 43]
    assert sorted(details) == [42, 43]
    assert details[42]["event_values"][(600, (0, 1, 2), 0)] == 0.75
    assert details[43]["event_values"][(600, (0, 1, 2), 1)] == 0.5


def test_replay_selection_rejects_empty_workload(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 1, "feasible": True, "target_queries": 1,
             "scores": [0.2], "events": [_event((0, 1, 2), 0)]},
        ],
        [],
    )

    with pytest.raises(ValueError, match="no metric rows selected"):
        load_trace_records_for_replay(
            trace_dir,
            ops=(600,),
            cells=("P1_v2b",),
            budget=1,
            methods=("mc",),
            regimes=("warm",),
            replicate=0,
        )



def test_auditor_requires_snapshot_trace_for_feasible_metric_row(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.9, 0.8, 0.1, 0.0], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
        ],
        [_metric_row(snapshot_trace=None)],
    )

    report = audit_trace_dir(trace_dir)

    assert report["metrics_reconstructed"] == 0
    assert any("missing required snapshot_trace" in failure for failure in report["failures"])


def test_auditor_rejects_trace_op_or_cell_mismatch(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 601, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.9, 0.8, 0.1, 0.0], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P2_v2c",
             "labels": [True, True, False, False], "exact_probability": [0.8, 0.7, 0.2, 0.1],
             "keys": [0.4, 0.3, 0.2, 0.1]},
        ],
        [_metric_row()],
    )

    report = audit_trace_dir(trace_dir)

    assert report["metrics_reconstructed"] == 0
    assert any("execution_trace op_id 601 does not match row op_id 600" in failure for failure in report["failures"])
    assert any("snapshot_trace cell 'P2_v2c' does not match row cell 'P1_v2b'" in failure for failure in report["failures"])


def test_auditor_reconstructs_metrics_from_snapshot_and_scores(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.9, 0.8, 0.1, 0.0], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P1_v2b",
             "labels": [True, True, False, False], "exact_probability": [0.8, 0.7, 0.2, 0.1],
             "keys": [0.4, 0.3, 0.2, 0.1]},
        ],
        [_metric_row()],
    )

    report = audit_trace_dir(trace_dir)

    assert report["failures"] == []
    assert report["metrics_reconstructed"] == 1


def test_auditor_reconstructs_rank_ties_with_key_order(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.5, 0.5, 0.4, 0.1], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P1_v2b",
             "labels": [False, True, True, False], "exact_probability": [0.5, 0.5, 0.4, 0.1],
             "keys": [0.9, 0.1, 0.2, 0.3]},
        ],
        [_metric_row(
            n_severe=2,
            rprecision=0.5,
            probability_mse=0.0,
            clipped_probability_mse=0.0,
            recall_10=0.5,
            precision_10=1.0,
            recall_20=0.5,
            precision_20=1.0,
            recall_40=0.5,
            precision_40=0.5,
        )],
    )

    report = audit_trace_dir(trace_dir)

    assert report["failures"] == []
    assert report["metrics_reconstructed"] == 1


def test_auditor_reports_metric_reconstruction_mismatch(tmp_path):
    trace_dir = _fixture(
        tmp_path,
        [
            {"trace_id": 20, "kind": "execution", "op_id": 600, "cell": "P1_v2b", "method": "mc",
             "seed": 100, "available_target_budget": 2, "feasible": True, "target_queries": 2,
             "scores": [0.9, 0.8, 0.1, 0.0], "events": [_event((0, 1, 2), 0), _event((0, 1, 2), 1)]},
            {"trace_id": 30, "kind": "evaluation_snapshot", "op_id": 600, "cell": "P1_v2b",
             "labels": [True, True, False, False], "exact_probability": [0.8, 0.7, 0.2, 0.1],
             "keys": [0.4, 0.3, 0.2, 0.1]},
        ],
        [_metric_row(rprecision=0.5)],
    )

    report = audit_trace_dir(trace_dir)

    assert report["metrics_reconstructed"] == 1
    assert any("metric rprecision stored 0.5 reconstructed 1.0" in failure for failure in report["failures"])
