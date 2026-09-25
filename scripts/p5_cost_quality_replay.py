"""Audit and bounded LP-replay checks for the development cost-quality traces.

The default audit streams the runner's gzip JSONL artifacts and checks query accounting without loading the
large logs into memory. The optional LP replay is deliberately bounded to the frozen small workload used for
wall-time sanity checks, not the full 60-op development run.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import re
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_OPS = (600, 620, 640)
DEFAULT_CELLS = ("P1_v2b", "P2_v2c")
DEFAULT_BUDGET = 1120
DEFAULT_METHODS = ("mc", "cv_sparse2", "cv_full")
DEFAULT_REGIMES = ("cold", "warm")
DEFAULT_REPLICATE = 0
LABEL_TOLERANCE = 1e-7
METRIC_TOLERANCE = 1e-12
METRIC_FIELDS = (
    "n_candidates",
    "n_severe",
    "rprecision",
    "probability_mse",
    "clipped_probability_mse",
    "recall_10",
    "precision_10",
    "recall_20",
    "precision_20",
    "recall_40",
    "precision_40",
)
TRACE_ID_PREFIX_RE = re.compile(r'^\s*\{\s*"trace_id"\s*:\s*(\d+)\s*[,}]')


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _canonical_event_key(event: dict[str, Any]) -> tuple[int, tuple[int, ...], int]:
    return (int(event["op_id"]), tuple(int(x) for x in event["outage"]), int(event["control_idx"]))


def _summarize_events(events: Iterable[dict[str, Any]]) -> tuple[int, int, list[int]]:
    keys = []
    for event in events:
        key = _canonical_event_key(event)
        keys.append(key)
    orders = sorted({len(key[1]) for key in keys})
    return len(set(keys)), len(keys) - len(set(keys)), orders


def load_traces(trace_path: Path) -> tuple[dict[int, dict[str, Any]], list[str], dict[str, Any]]:
    traces: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    kind_counts: dict[str, int] = defaultdict(int)
    trace_id_min: int | None = None
    trace_id_max: int | None = None
    for line_no, record in iter_jsonl_gz(trace_path):
        trace_id = int(record["trace_id"])
        trace_id_min = trace_id if trace_id_min is None else min(trace_id_min, trace_id)
        trace_id_max = trace_id if trace_id_max is None else max(trace_id_max, trace_id)
        if trace_id in traces:
            failures.append(f"trace {trace_id}: duplicate trace_id at line {line_no}")
        kind_counts[str(record.get("kind"))] += 1
        stored = {
            "trace_id": trace_id,
            "kind": record.get("kind"),
            "op_id": record.get("op_id"),
            "cell": record.get("cell"),
        }
        if record.get("kind") not in {"construction", "execution", "evaluation_snapshot"}:
            traces[trace_id] = stored
            continue
        events = record.get("events", [])
        if events is None:
            events = []
        unique_count, duplicates, orders = _summarize_events(events)
        stored.update(
            unique_count=unique_count,
            order_set=orders,
            duplicate_events=duplicates,
            spent=record.get("spent"),
            target_queries=record.get("target_queries"),
        )
        if record.get("kind") == "execution":
            stored["scores"] = record.get("scores")
        if record.get("kind") == "evaluation_snapshot":
            stored["labels"] = record.get("labels")
            stored["exact_probability"] = record.get("exact_probability")
            stored["keys"] = record.get("keys")
        traces[trace_id] = stored
        if duplicates:
            failures.append(f"trace {trace_id}: {duplicates} duplicate event keys")
        expected = record.get("spent") if record.get("kind") == "construction" else record.get("target_queries")
        if expected is not None and int(expected) != unique_count:
            name = "spent" if record.get("kind") == "construction" else "target_queries"
            failures.append(f"trace {trace_id}: {name} {expected} but has {unique_count} unique event keys")
        if record.get("kind") == "construction" and any(order not in (1, 2) for order in orders):
            failures.append(f"trace {trace_id}: construction trace has event orders {orders}")
        if record.get("kind") == "execution" and orders not in ([], [3]):
            failures.append(f"trace {trace_id}: execution trace has event orders {orders}")
    return traces, failures, {
        "trace_records": len(traces),
        "trace_id_min": trace_id_min,
        "trace_id_max": trace_id_max,
        "trace_kind_counts": dict(sorted(kind_counts.items())),
    }


def _rank_order(scores: np.ndarray, keys: np.ndarray) -> np.ndarray:
    return np.lexsort((keys, -scores))


def reconstruct_metrics(snapshot: dict[str, Any], execution: dict[str, Any]) -> dict[str, Any]:
    labels = np.asarray(snapshot.get("labels"), dtype=bool)
    scores = np.asarray(execution.get("scores"), dtype=float)
    keys = np.asarray(snapshot.get("keys"), dtype=float)
    exact = np.asarray(snapshot.get("exact_probability"), dtype=float)
    if not (labels.shape == scores.shape == keys.shape == exact.shape):
        raise ValueError("snapshot labels/keys/exact_probability and execution scores must have identical shapes")
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("metric reconstruction requires nonempty one-dimensional arrays")
    if not (np.isfinite(scores).all() and np.isfinite(keys).all() and np.isfinite(exact).all()):
        raise ValueError("metric reconstruction inputs must be finite")
    order = _rank_order(scores, keys)
    n_severe = int(labels.sum())
    out: dict[str, Any] = {
        "n_candidates": int(labels.size),
        "n_severe": n_severe,
        "rprecision": None if n_severe == 0 else float(labels[order[:n_severe]].mean()),
        "probability_mse": float(np.mean((scores - exact) ** 2)),
        "clipped_probability_mse": float(np.mean((np.clip(scores, 0.0, 1.0) - exact) ** 2)),
    }
    for pct in (10, 20, 40):
        k = max(1, int(round((pct / 100.0) * labels.size)))
        top = labels[order[:k]]
        out[f"recall_{pct}"] = None if n_severe == 0 else float(top.sum() / n_severe)
        out[f"precision_{pct}"] = float(top.mean())
    return out


def _metric_values_match(stored: Any, reconstructed: Any) -> bool:
    if stored is None or reconstructed is None:
        return stored is None and reconstructed is None
    if isinstance(stored, (int, np.integer)) and isinstance(reconstructed, (int, np.integer)):
        return int(stored) == int(reconstructed)
    try:
        return abs(float(stored) - float(reconstructed)) <= METRIC_TOLERANCE
    except (TypeError, ValueError):
        return stored == reconstructed


def _trace_summary(
    traces: dict[int, dict[str, Any]],
    trace_id: Any,
    failures: list[str],
    metric_line: int,
    role: str,
    *,
    required: bool = False,
):
    if trace_id is None:
        if required:
            failures.append(f"metric line {metric_line}: missing required {role}_trace")
        return None
    try:
        trace = traces[int(trace_id)]
    except (KeyError, ValueError, TypeError):
        failures.append(f"metric line {metric_line}: missing {role} trace {trace_id}")
        return None
    return trace


def _trace_matches_metric_row(trace: dict[str, Any], row: dict[str, Any], failures: list[str], metric_line: int, role: str) -> bool:
    matches = True
    trace_op = trace.get("op_id")
    row_op = row.get("op_id")
    try:
        op_matches = int(trace_op) == int(row_op)
    except (TypeError, ValueError):
        op_matches = trace_op == row_op
    if not op_matches:
        failures.append(f"metric line {metric_line}: {role}_trace op_id {trace_op!r} does not match row op_id {row_op!r}")
        matches = False
    if trace.get("cell") != row.get("cell"):
        failures.append(
            f"metric line {metric_line}: {role}_trace cell {trace.get('cell')!r} "
            f"does not match row cell {row.get('cell')!r}"
        )
        matches = False
    return matches


def audit_loaded_traces(
    trace_dir: Path | str,
    traces: dict[int, dict[str, Any]],
    failures: list[str],
    trace_summary: dict[str, Any],
) -> dict[str, Any]:
    trace_dir = Path(trace_dir)
    failures = list(failures)
    regime_summary: dict[str, dict[str, int]] = defaultdict(lambda: dict(evaluations=0, total_unique_keys=0))
    metrics_records = 0
    skipped_infeasible = 0
    feasible_records = 0
    metrics_reconstructed = 0
    metric_reconstruction_cache_hits = 0
    metric_reconstruction_cache: dict[tuple[int, int], dict[str, Any]] = {}
    for line_no, row in iter_jsonl_gz(trace_dir / "metrics.jsonl.gz"):
        metrics_records += 1
        if not row.get("feasible", False):
            skipped_infeasible += 1
            continue
        feasible_records += 1
        construction_trace = _trace_summary(traces, row.get("construction_trace"), failures, line_no, "construction")
        execution_trace = _trace_summary(traces, row.get("execution_trace"), failures, line_no, "execution", required=True)
        snapshot_trace = _trace_summary(traces, row.get("snapshot_trace"), failures, line_no, "snapshot", required=True)
        regime = str(row["regime"])
        target_queries = int(row["target_queries"])
        budget = int(row["budget"])
        execution_count = int(execution_trace.get("unique_count", 0)) if execution_trace is not None else 0
        construction_count = int(construction_trace.get("unique_count", 0)) if construction_trace is not None else 0
        execution_ready = False
        snapshot_ready = False
        if execution_trace is not None:
            if execution_trace.get("kind") != "execution":
                failures.append(f"metric line {line_no}: execution_trace points to {execution_trace.get('kind')!r}")
            else:
                execution_ready = _trace_matches_metric_row(execution_trace, row, failures, line_no, "execution")
        if construction_trace is not None and construction_trace.get("kind") != "construction":
            failures.append(f"metric line {line_no}: construction_trace points to {construction_trace.get('kind')!r}")
        if snapshot_trace is not None:
            if snapshot_trace.get("kind") != "evaluation_snapshot":
                failures.append(f"metric line {line_no}: snapshot_trace points to {snapshot_trace.get('kind')!r}")
            else:
                snapshot_ready = _trace_matches_metric_row(snapshot_trace, row, failures, line_no, "snapshot")
        if snapshot_ready and execution_ready and snapshot_trace is not None and execution_trace is not None:
            try:
                cache_key = (int(snapshot_trace["trace_id"]), int(execution_trace["trace_id"]))
                if cache_key in metric_reconstruction_cache:
                    reconstructed = metric_reconstruction_cache[cache_key]
                    metric_reconstruction_cache_hits += 1
                else:
                    reconstructed = reconstruct_metrics(snapshot_trace, execution_trace)
                    metric_reconstruction_cache[cache_key] = reconstructed
                metrics_reconstructed += 1
                for field in METRIC_FIELDS:
                    if field not in row:
                        failures.append(f"metric line {line_no}: missing metric {field}")
                    elif not _metric_values_match(row[field], reconstructed[field]):
                        failures.append(
                            f"metric line {line_no}: metric {field} stored {row[field]!r} "
                            f"reconstructed {reconstructed[field]!r}"
                        )
            except ValueError as exc:
                failures.append(f"metric line {line_no}: metric reconstruction failed: {exc}")
        if target_queries != execution_count:
            failures.append(
                f"metric line {line_no}: target_queries {target_queries} but execution trace has "
                f"{execution_count} unique event keys"
            )
        if regime == "cold":
            unique_charge = construction_count + execution_count
            expected_total = row.get("total_queries")
            charge_name = "cold unique charge"
        elif regime == "warm":
            unique_charge = execution_count
            expected_total = row.get("total_queries")
            charge_name = "warm target unique charge"
        else:
            failures.append(f"metric line {line_no}: unknown regime {regime!r}")
            unique_charge = execution_count
            expected_total = row.get("total_queries")
            charge_name = "unique charge"
        if expected_total is not None and int(expected_total) != unique_charge:
            failures.append(f"metric line {line_no}: total_queries {expected_total} but {charge_name} is {unique_charge}")
        if expected_total is not None and int(expected_total) > budget:
            failures.append(f"metric line {line_no}: reported total_queries {expected_total} exceeds budget {budget}")
        if unique_charge > budget:
            failures.append(f"metric line {line_no}: {charge_name} {unique_charge} exceeds budget {budget}")
        regime_summary[regime]["evaluations"] += 1
        regime_summary[regime]["total_unique_keys"] += unique_charge
    return {
        **trace_summary,
        "metrics_records": metrics_records,
        "feasible_records": feasible_records,
        "skipped_infeasible": skipped_infeasible,
        "metrics_reconstructed": metrics_reconstructed,
        "metric_reconstruction_cache_entries": len(metric_reconstruction_cache),
        "metric_reconstruction_cache_hits": metric_reconstruction_cache_hits,
        "regime_summary": {k: dict(v) for k, v in sorted(regime_summary.items())},
        "failures": failures,
    }


def audit_trace_dir(trace_dir: Path | str) -> dict[str, Any]:
    trace_dir = Path(trace_dir)
    traces, failures, trace_summary = load_traces(trace_dir / "traces.jsonl.gz")
    return audit_loaded_traces(trace_dir, traces, failures, trace_summary)


def load_trace_records_for_replay(
    trace_dir: Path,
    *,
    ops: tuple[int, ...],
    cells: tuple[str, ...],
    budget: int,
    methods: tuple[str, ...],
    regimes: tuple[str, ...],
    replicate: int,
    trace_index: tuple[dict[int, dict[str, Any]], list[str], dict[str, Any]] | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], list[dict[str, Any]]]:
    if trace_index is None:
        traces, failures, _ = load_traces(trace_dir / "traces.jsonl.gz")
    else:
        traces, failures, _ = trace_index
    if failures:
        raise ValueError("trace audit failures must be fixed before LP replay: " + "; ".join(failures[:5]))
    selected = []
    detail_ids: set[int] = set()
    for _, row in iter_jsonl_gz(trace_dir / "metrics.jsonl.gz"):
        if (
            int(row["op_id"]) in ops
            and row["cell"] in cells
            and int(row["budget"]) == budget
            and row["method"] in methods
            and row["regime"] in regimes
            and int(row["replicate"]) == replicate
        ):
            selected.append(row)
            if row.get("feasible", False):
                detail_ids.add(int(row["execution_trace"]))
                if row["regime"] == "cold" and row.get("construction_trace") is not None:
                    detail_ids.add(int(row["construction_trace"]))
    if not selected:
        raise ValueError("no metric rows selected for requested LP replay workload")
    details = load_selected_trace_details(trace_dir / "traces.jsonl.gz", detail_ids)
    return traces, details, selected


def _trace_id_prefix(line: str) -> int | None:
    match = TRACE_ID_PREFIX_RE.match(line)
    return int(match.group(1)) if match else None


def load_selected_trace_details(trace_path: Path, trace_ids: set[int]) -> dict[int, dict[str, Any]]:
    if not trace_ids:
        return {}
    details: dict[int, dict[str, Any]] = {}
    remaining = set(trace_ids)
    with gzip.open(trace_path, "rt") as handle:
        lines = enumerate(handle, 1)
        for _, line in lines:
            if not line.strip():
                continue
            prefix_id = _trace_id_prefix(line)
            if prefix_id is not None and prefix_id not in remaining:
                continue
            record = json.loads(line)
            trace_id = int(record["trace_id"])
            if trace_id not in remaining:
                continue
            event_values = {}
            for event in record.get("events") or []:
                event_values.setdefault(_canonical_event_key(event), float(event["value"]))
            details[trace_id] = {
                "trace_id": trace_id,
                "kind": record.get("kind"),
                "event_values": event_values,
            }
            remaining.remove(trace_id)
            if not remaining:
                break
    if remaining:
        raise ValueError(f"selected trace ids not found: {sorted(remaining)}")
    return details


def _key_from_tuple(key: tuple[int, tuple[int, ...], int]) -> str:
    op, outage, control_idx = key
    return f"{op}:{','.join(str(x) for x in outage)}:{control_idx}"


def _load_control_grid():
    CV = None
    for k in (1, 2, 3):
        with np.load(Path(f"data_hik/n{k}_confirm.npz")) as data:
            if CV is None:
                CV = data["CV"].copy()
            elif not np.array_equal(CV, data["CV"]):
                raise ValueError("label tables disagree on control-grid ordering")
    if CV is None:
        raise ValueError("no CV grid loaded")
    return CV


def replay_lp(trace_dir: Path, output_dir: Path, *, ops=DEFAULT_OPS, cells=DEFAULT_CELLS, budget=DEFAULT_BUDGET,
              methods=DEFAULT_METHODS, regimes=DEFAULT_REGIMES, replicate=DEFAULT_REPLICATE,
              trace_index: tuple[dict[int, dict[str, Any]], list[str], dict[str, Any]] | None = None) -> dict[str, Any]:
    ops = tuple(int(op) for op in ops)
    if any(op < 600 or op > 659 for op in ops):
        raise ValueError("LP replay op ids must stay within the already-open 600..659 development block")
    traces, trace_details, selected = load_trace_records_for_replay(
        trace_dir,
        ops=ops,
        cells=tuple(cells),
        budget=int(budget),
        methods=tuple(methods),
        regimes=tuple(regimes),
        replicate=int(replicate),
        trace_index=trace_index,
    )
    CV = _load_control_grid()
    from fdna.dataset import G, RATING
    from fdna.lp import ScenarioLP
    from fdna.opgen import sample_op
    from fdna import spec

    @lru_cache(maxsize=None)
    def lp(op_id: int, outage: tuple[int, ...]):
        op = sample_op(G, RATING, np.random.default_rng(op_id))
        return ScenarioLP(G, op, outage, spec.PARAMS)

    def exact_label(key: tuple[int, tuple[int, ...], int]) -> tuple[float, float]:
        op_id, outage, control_idx = key
        start = time.perf_counter()
        value = float(lp(op_id, outage).y(CV[control_idx]))
        return value, time.perf_counter() - start

    per_row = []
    global_max_deviation = 0.0
    total_lp_queries = 0
    total_lp_wall = 0.0
    label_failures = []
    for row in selected:
        if not row.get("feasible", False):
            per_row.append({"row": row, "skipped": "infeasible"})
            continue
        lp.cache_clear()
        construction_summary = traces[int(row["construction_trace"])] if row.get("construction_trace") is not None else None
        construction_detail = trace_details.get(int(row["construction_trace"])) if row.get("construction_trace") is not None else None
        execution_detail = trace_details[int(row["execution_trace"])]
        construction_values = construction_detail["event_values"] if construction_detail is not None else {}
        execution_values = execution_detail["event_values"]
        event_values = dict(execution_values)
        if row["regime"] == "cold":
            event_values.update(construction_values)
        keys = set(event_values)
        row_max = 0.0
        row_wall = 0.0
        for key in sorted(keys):
            value, elapsed = exact_label(key)
            row_wall += elapsed
            cached = float(event_values[key])
            delta = abs(value - cached)
            row_max = max(row_max, delta)
            if not np.isfinite(value) or not np.isfinite(cached) or delta > LABEL_TOLERANCE:
                label_failures.append(
                    f"{row['regime']} {row['cell']} op{row['op_id']} {row['method']} "
                    f"{_key_from_tuple(key)} cached={cached} exact={value} delta={delta}"
                )
        global_max_deviation = max(global_max_deviation, row_max)
        total_lp_queries += len(keys)
        total_lp_wall += row_wall
        per_row.append({
            "regime": row["regime"],
            "cell": row["cell"],
            "op_id": int(row["op_id"]),
            "replicate": int(row["replicate"]),
            "budget": int(row["budget"]),
            "method": row["method"],
            "unique_replayed_keys": len(keys),
            "warm_sunk_construction_keys_not_timed": int(construction_summary.get("unique_count", 0))
            if row["regime"] == "warm" and construction_summary is not None else 0,
            "max_label_deviation": row_max,
            "lp_wall_s": row_wall,
            "trace_keys": [_key_from_tuple(key) for key in sorted(keys)],
        })
    if label_failures:
        raise ValueError("LP replay label mismatches above tolerance: " + "; ".join(label_failures[:10]))
    report = {
        "scope": "bounded actual-LP replay for frozen development sanity workload",
        "disclaimer": "Timing covers 3 selected ops, not the full 60-op development wall time; warm cache lower-order sunk cost is reported but not timed.",
        "selection": {
            "ops": list(ops),
            "cells": list(cells),
            "budget": int(budget),
            "methods": list(methods),
            "regimes": list(regimes),
            "replicate": int(replicate),
        },
        "selected_metric_rows": len(selected),
        "replayed_metric_rows": sum(1 for row in per_row if "skipped" not in row),
        "skipped_infeasible": sum(1 for row in per_row if row.get("skipped") == "infeasible"),
        "total_unique_lp_queries_charged_with_per_scenario_cache_reset": total_lp_queries,
        "label_tolerance": LABEL_TOLERANCE,
        "label_mismatch_count": 0,
        "max_label_deviation": global_max_deviation,
        "lp_wall_s": total_lp_wall,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "rows": per_row,
    }
    write_json(output_dir / "lp_replay.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("data_hik/cost_quality_dev"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase5/cost_quality_dev"))
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--ops", type=int, nargs="+", default=list(DEFAULT_OPS))
    parser.add_argument("--cells", nargs="+", default=list(DEFAULT_CELLS))
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    parser.add_argument("--regimes", nargs="+", default=list(DEFAULT_REGIMES))
    parser.add_argument("--replicate", type=int, default=DEFAULT_REPLICATE)
    args = parser.parse_args(argv)
    if any(op < 600 or op > 659 for op in args.ops):
        parser.error("--ops must stay within the already-open 600..659 development block")

    trace_index = load_traces(args.trace_dir / "traces.jsonl.gz")
    audit = audit_loaded_traces(args.trace_dir, *trace_index)
    write_json(args.output_dir / "trace_audit.json", audit)
    if audit["failures"]:
        print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
        return 1
    if not args.audit_only:
        replay_lp(
            args.trace_dir,
            args.output_dir,
            ops=tuple(args.ops),
            cells=tuple(args.cells),
            budget=args.budget,
            methods=tuple(args.methods),
            regimes=tuple(args.regimes),
            replicate=args.replicate,
            trace_index=trace_index,
        )
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
