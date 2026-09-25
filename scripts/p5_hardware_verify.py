"""Hardware-backed verification for the frozen Phase 5 warm gate candidate.

This verifier does not regenerate the development run. It replays the frozen warm candidate
(cv_full budget 280) and its mc/proxy_full baselines with real ScenarioLP target labels,
using the existing n1/n2 tables only to rebuild the preexisting full warm proxy cache.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from fdna import spec, v2, v2data
from fdna.cost_quality import screening_metrics, select_gate
from fdna.evalutil import scenario_uniform
from fdna.query_budget import BudgetedOracle, g2_proxy
from scripts.p5_cost_quality import CELLS, execute_policy, summarize

GATE_METHOD = "cv_full"
VERIFY_METHODS = ("mc", "proxy_full", "cv_full")
DEFAULT_BUDGET = 280
DEFAULT_REPLICATES = 10
LABEL_TOLERANCE = 1e-7

_CV: np.ndarray | None = None
_LOWER_TABLES: dict[tuple[int, tuple[int, ...]], np.ndarray] | None = None
_OUTAGES: list[tuple[int, int, int]] | None = None
_WORLD: Any = None



def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def _package_versions() -> dict[str, str]:
    import scipy

    return {"numpy": np.__version__, "scipy": scipy.__version__}


def _hash_existing(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): sha256(path) for path in paths if path.exists()}


def _development_provenance_matches(provenance_path: Path, current_hashes: dict[str, str]) -> dict[str, dict[str, Any]]:
    if not provenance_path.exists():
        return {}
    provenance = json.loads(provenance_path.read_text())
    recorded = {}
    recorded.update(provenance.get("inputs", {}))
    recorded.update(provenance.get("artifacts", {}))
    recorded["config_sha256"] = provenance.get("config_sha256")
    out = {}
    for path, current in current_hashes.items():
        key = path
        expected = recorded.get(key)
        if expected is None and path == "results/phase5/cost_quality_dev/config.json":
            expected = recorded.get("config_sha256")
        if expected is None:
            out[path] = {
                "recorded_sha256": None,
                "current_sha256": current,
                "matches": None,
                "status": "not_recorded_in_development_provenance",
            }
        else:
            out[path] = {
                "recorded_sha256": expected,
                "current_sha256": current,
                "matches": expected == current,
                "status": "match" if expected == current else "mismatch",
            }
    return out


def _report_content_sha256(report: dict[str, Any]) -> str:
    payload = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                yield line_no, json.loads(line)


def leading_trace_id(line: str) -> int | None:
    prefix = '{"trace_id":'
    if not line.startswith(prefix):
        return None
    end = line.find(',', len(prefix))
    if end < 0:
        return None
    try:
        return int(line[len(prefix):end])
    except ValueError:
        return None


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _event_key(event: dict[str, Any]) -> tuple[int, tuple[int, ...], int]:
    return (int(event["op_id"]), tuple(int(x) for x in event["outage"]), int(event["control_idx"]))


class MemoizedScalarBackend:
    """Share physical scalar solves across verification rows while preserving policy charges."""

    def __init__(self, backend: Callable[[int, tuple[int, ...], int], float]) -> None:
        self._backend = backend
        self._cache: dict[tuple[int, tuple[int, ...], int], float] = {}
        self.requests = 0
        self.unique_solves = 0

    def __call__(self, op_id: int, outage: Iterable[int], control_idx: int) -> float:
        self.requests += 1
        key = (int(op_id), tuple(int(x) for x in outage), int(control_idx))
        if key not in self._cache:
            self._cache[key] = float(self._backend(*key))
            self.unique_solves += 1
        return self._cache[key]

    @property
    def cache_size(self) -> int:
        return len(self._cache)


def run_live_policy(
    kind: str,
    backend: MemoizedScalarBackend,
    op_id: int,
    outages: list[tuple[int, ...]],
    CV: np.ndarray,
    pc: np.ndarray,
    *,
    budget: int,
    seed: int,
    proxy: np.ndarray | None = None,
    floors: np.ndarray | None = None,
    construction_cost: int = 0,
) -> dict[str, Any]:
    result = execute_policy(
        kind,
        backend,
        int(op_id),
        outages,
        CV,
        pc,
        budget=int(budget),
        construction_cost=int(construction_cost),
        seed=int(seed),
        proxy=proxy,
        floors=floors,
    )
    result["events"] = [
        {
            "op_id": int(event["op_id"]),
            "outage": tuple(int(x) for x in event["outage"]),
            "control_idx": int(event["control_idx"]),
            "value": float(event["value"]),
        }
        for event in result.get("events", [])
    ]
    if result.get("scores") is not None:
        result["scores"] = [float(x) for x in result["scores"]]
    return result


def compare_event_labels(
    live_events: Iterable[dict[str, Any]],
    frozen_events: Iterable[dict[str, Any]],
    *,
    tolerance: float = LABEL_TOLERANCE,
    severe_threshold: float = spec.SEVERE,
) -> dict[str, Any]:
    live = {_event_key(event): float(event["value"]) for event in live_events}
    frozen = {_event_key(event): float(event["value"]) for event in frozen_events}
    keys = sorted(set(live) | set(frozen))
    max_abs_error = 0.0
    value_mismatches = 0
    severity_mismatches = 0
    compared = 0
    for key in keys:
        if key not in live or key not in frozen:
            continue
        compared += 1
        delta = abs(live[key] - frozen[key])
        max_abs_error = max(max_abs_error, delta)
        if delta > tolerance:
            value_mismatches += 1
        if (live[key] > severe_threshold) != (frozen[key] > severe_threshold):
            severity_mismatches += 1
    return {
        "compared": compared,
        "missing_live": len(set(frozen) - set(live)),
        "missing_frozen": len(set(live) - set(frozen)),
        "value_mismatches": value_mismatches,
        "severity_mismatches": severity_mismatches,
        "max_abs_error": max_abs_error,
    }


def _load_manifest_ops(manifest_path: Path, op_start: int, ops: int) -> tuple[list[int], list[tuple[int, int, int]]]:
    manifest = json.loads(manifest_path.read_text())
    selected_ops = [int(op) for op in sorted(manifest["op_ids"]) if int(op) >= op_start][:ops]
    if len(selected_ops) != ops:
        raise ValueError("requested operating points exceed the development manifest")
    if any(op < 600 or op > 659 for op in selected_ops):
        raise ValueError("hardware verification is restricted to frozen development ops 600..659")
    outages = [tuple(int(x) for x in outage) for outage in sorted(manifest["outage_sets"])]
    return selected_ops, outages


def _load_cv_and_lower_tables(data_dir: Path, ops: set[int]) -> tuple[np.ndarray, dict[tuple[int, tuple[int, ...]], np.ndarray]]:
    CV = None
    tables: dict[tuple[int, tuple[int, ...]], np.ndarray] = {}
    for k in (1, 2):
        with np.load(data_dir / f"n{k}_confirm.npz") as data:
            if CV is None:
                CV = data["CV"].copy()
            elif not np.array_equal(CV, data["CV"]):
                raise ValueError("lower-order label tables disagree on control-grid ordering")
            for op, outage, values in zip(data["op_ids"], data["outages"], data["V"]):
                op_id = int(op)
                if op_id in ops:
                    tables[(op_id, tuple(int(x) for x in outage[:k]))] = np.asarray(values, dtype=float)
    if CV is None:
        raise ValueError("no control grid loaded")
    return CV, tables


def _init_worker(data_dir: str, ops: list[int], outages: list[tuple[int, int, int]]) -> None:
    global _CV, _LOWER_TABLES, _OUTAGES, _WORLD
    _CV, _LOWER_TABLES = _load_cv_and_lower_tables(Path(data_dir), set(int(op) for op in ops))
    _OUTAGES = [tuple(int(x) for x in outage) for outage in outages]
    _WORLD = v2.build_world()


def _require_worker_data() -> tuple[np.ndarray, dict[tuple[int, tuple[int, ...]], np.ndarray], list[tuple[int, int, int]], Any]:
    if _CV is None or _LOWER_TABLES is None or _OUTAGES is None or _WORLD is None:
        raise RuntimeError("worker data was not initialized")
    return _CV, _LOWER_TABLES, _OUTAGES, _WORLD


def _load_selected_frozen(trace_dir: Path, ops: set[int], budget: int, replicates: int) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    selected = []
    detail_ids: set[int] = set()
    for _, row in iter_jsonl_gz(trace_dir / "metrics.jsonl.gz"):
        if (
            row.get("regime") == "warm"
            and int(row.get("op_id")) in ops
            and int(row.get("budget")) == int(budget)
            and row.get("method") in VERIFY_METHODS
            and int(row.get("replicate")) < int(replicates)
            and row.get("cell") in CELLS
        ):
            selected.append(row)
            if row.get("execution_trace") is not None:
                detail_ids.add(int(row["execution_trace"]))
            if row.get("snapshot_trace") is not None:
                detail_ids.add(int(row["snapshot_trace"]))
    expected = len(ops) * len(CELLS) * replicates * len(VERIFY_METHODS)
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} frozen metric rows, expected {expected}")
    details: dict[int, dict[str, Any]] = {}
    remaining = set(detail_ids)
    with gzip.open(trace_dir / "traces.jsonl.gz", "rt") as handle:
        for line in handle:
            if not remaining:
                break
            if not line.strip():
                continue
            trace_id = leading_trace_id(line)
            if trace_id is None:
                trace_id = int(json.loads(line)["trace_id"])
            if trace_id not in remaining:
                continue
            record = json.loads(line)
            details[trace_id] = record
            remaining.remove(trace_id)
    if remaining:
        raise ValueError(f"selected trace ids not found: {sorted(remaining)[:10]}")
    return selected, details


def _lower_backend(op_id: int, outage: tuple[int, ...], control_idx: int) -> float:
    _, lower_tables, _, _ = _require_worker_data()
    return float(lower_tables[(int(op_id), tuple(outage))][int(control_idx)])


def _make_physical_backend(op_id: int) -> MemoizedScalarBackend:
    CV, _, _, _ = _require_worker_data()
    from fdna.dataset import G, RATING
    from fdna.lp import ScenarioLP
    from fdna.opgen import sample_op

    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    lp_cache: dict[tuple[int, ...], ScenarioLP] = {}

    def solve(request_op: int, outage: tuple[int, ...], control_idx: int) -> float:
        if int(request_op) != int(op_id):
            raise ValueError(f"worker for op {op_id} received op {request_op}")
        outage = tuple(int(x) for x in outage)
        if outage not in lp_cache:
            lp_cache[outage] = ScenarioLP(G, op, outage, spec.PARAMS)
        return float(lp_cache[outage].y(CV[int(control_idx)]))

    return MemoizedScalarBackend(solve)


def _metric_diff(stored: dict[str, Any], live: dict[str, Any]) -> tuple[float, int]:
    max_diff = 0.0
    mismatches = 0
    for key, value in live.items():
        if key not in stored:
            mismatches += 1
            continue
        other = stored[key]
        if value is None or other is None:
            if value != other:
                mismatches += 1
            continue
        delta = abs(float(value) - float(other))
        max_diff = max(max_diff, delta)
        if delta > 1e-12:
            mismatches += 1
    return max_diff, mismatches


def _scores_diff(live: list[float] | None, frozen: Any) -> tuple[float, int]:
    if live is None and frozen is None:
        return 0.0, 0
    if live is None or frozen is None:
        return float("inf"), 1
    a = np.asarray(live, dtype=float)
    b = np.asarray(frozen, dtype=float)
    if a.shape != b.shape:
        return float("inf"), 1
    diff = np.abs(a - b)
    return float(diff.max(initial=0.0)), int(np.count_nonzero(diff > 1e-12))


def _verify_op_task(op_id: int, rows: list[dict[str, Any]], details: dict[int, dict[str, Any]], budget: int) -> dict[str, Any]:
    CV, _, outages, world = _require_worker_data()
    from fdna.dataset import G, RATING
    from fdna.opgen import sample_op
    from fdna.physical_correction import island_floor

    physical = _make_physical_backend(op_id)
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    floors = np.array([island_floor(G, op.demand, outage) for outage in outages], dtype=float)
    by_key = {(row["cell"], int(row["replicate"]), row["method"]): row for row in rows}
    replicate_count = max(int(row["replicate"]) for row in rows) + 1
    live_rows = []
    failures = []
    totals = defaultdict(float)
    started = time.perf_counter()
    for cell_index, (cell, (q, stale, coverage)) in enumerate(CELLS.items(), 1):
        rng = np.random.default_rng([op_id, cell_index])
        state = v2.sample_states(world, rng, 1)
        obs = v2.emit(world, state, q, stale, rng, coverage)
        pc = v2data.oracle_features(world, obs, stale)[0][0].astype(float)
        pc /= pc.sum()
        keys = scenario_uniform(
            np.full(len(outages), op_id),
            np.array(outages)[:, 0],
            np.array(outages)[:, 1],
            np.array(outages)[:, 2],
            salt=17,
        )
        lower_oracle = BudgetedOracle(_lower_backend, 10**12)
        proxy_grid, _ = g2_proxy(lower_oracle, op_id, outages, CV, pc, None)
        hidden_idx = int(world.cidx[state][0])
        live_hidden_values = np.array([physical(op_id, outage, hidden_idx) for outage in outages], dtype=float)
        totals["fresh_realized_label_checks"] += len(outages)
        live_labels = live_hidden_values > spec.SEVERE
        any_row = by_key[(cell, 0, GATE_METHOD)]
        snapshot = details[int(any_row["snapshot_trace"])]
        frozen_labels = np.asarray(snapshot["labels"], dtype=bool)
        if not np.array_equal(live_labels, frozen_labels):
            mismatch = int(np.count_nonzero(live_labels != frozen_labels))
            failures.append(f"op {op_id} {cell}: {mismatch} realized severe-label mismatches")
            totals["realized_severity_mismatches"] += mismatch
        for rep in range(replicate_count):
            seed = cell_index * 100 + rep
            for method in VERIFY_METHODS:
                frozen_row = by_key[(cell, rep, method)]
                frozen_execution = details[int(frozen_row["execution_trace"])]
                kind = "cv" if method == "cv_full" else "proxy" if method == "proxy_full" else "mc"
                result = run_live_policy(
                    kind,
                    physical,
                    op_id,
                    outages,
                    CV,
                    pc,
                    budget=budget,
                    seed=seed,
                    proxy=proxy_grid if kind in {"cv", "proxy"} else None,
                    floors=floors,
                    construction_cost=0,
                )
                if bool(result["feasible"]) != bool(frozen_row["feasible"]):
                    failures.append(f"op {op_id} {cell} rep {rep} {method}: feasible mismatch")
                event_report = compare_event_labels(result.get("events", []), frozen_execution.get("events", []))
                totals["event_labels_compared"] += event_report["compared"]
                totals["event_missing_live"] += event_report["missing_live"]
                totals["event_missing_frozen"] += event_report["missing_frozen"]
                totals["event_value_mismatches"] += event_report["value_mismatches"]
                totals["event_severity_mismatches"] += event_report["severity_mismatches"]
                totals["max_event_abs_error"] = max(totals["max_event_abs_error"], event_report["max_abs_error"])
                score_max, score_bad = _scores_diff(result.get("scores"), frozen_execution.get("scores"))
                totals["score_mismatches"] += score_bad
                totals["max_score_abs_diff"] = max(totals["max_score_abs_diff"], score_max)
                totals["independent_policy_target_queries"] += int(result["target_queries"])
                if int(result["target_queries"]) != int(frozen_row["target_queries"]):
                    failures.append(
                        f"op {op_id} {cell} rep {rep} {method}: target_queries "
                        f"{result['target_queries']} != frozen {frozen_row['target_queries']}"
                    )
                    totals["query_count_mismatches"] += 1
                live_row = {
                    "regime": "warm",
                    "cell": cell,
                    "op_id": int(op_id),
                    "replicate": int(rep),
                    "budget": int(budget),
                    "method": method,
                    "feasible": bool(result["feasible"]),
                    "target_queries": int(result["target_queries"]),
                    "construction_queries": int(frozen_row.get("construction_queries", 0)),
                    "charged_construction_queries": 0,
                    "total_queries": int(result["target_queries"]) if result["feasible"] else None,
                    "unused_budget": int(budget) - int(result["target_queries"]) if result["feasible"] else None,
                    "execution_trace": int(frozen_row["execution_trace"]),
                    "snapshot_trace": int(frozen_row["snapshot_trace"]),
                    "hardware_event_max_abs_error": event_report["max_abs_error"],
                }
                if result["feasible"]:
                    metrics = screening_metrics(
                        live_labels,
                        result["scores"],
                        keys,
                        np.asarray(snapshot["exact_probability"], dtype=float),
                    )
                    live_row.update(metrics)
                    metric_max, metric_bad = _metric_diff(frozen_row, metrics)
                    totals["metric_mismatches"] += metric_bad
                    totals["max_metric_abs_diff"] = max(totals["max_metric_abs_diff"], metric_max)
                live_rows.append(live_row)
    return {
        "op_id": int(op_id),
        "rows": live_rows,
        "failures": failures,
        "unique_physical_solves": physical.unique_solves,
        "physical_requests": physical.requests,
        "worker_wall_s": time.perf_counter() - started,
        "totals": dict(totals),
    }


def _group_rows_by_op(rows: list[dict[str, Any]], details: dict[int, dict[str, Any]]) -> dict[int, tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    detail_ids_by_op: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        op_id = int(row["op_id"])
        grouped[op_id].append(row)
        detail_ids_by_op[op_id].add(int(row["execution_trace"]))
        detail_ids_by_op[op_id].add(int(row["snapshot_trace"]))
    return {op: (op_rows, {trace_id: details[trace_id] for trace_id in detail_ids_by_op[op]}) for op, op_rows in grouped.items()}


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", compresslevel=3) as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")


def _gate_diff(frozen_gate: dict[str, Any], live_gate: dict[str, Any]) -> dict[str, Any]:
    diffs = []
    max_abs = 0.0
    frozen_evidence = frozen_gate["warm"]["selected"]["evidence"]
    live_evidence = live_gate["warm"]["selected"]["evidence"] if live_gate["warm"]["selected"] else []
    live_by_key = {(row["cell"], row["comparator"]): row for row in live_evidence}
    for row in frozen_evidence:
        key = (row["cell"], row["comparator"])
        other = live_by_key.get(key)
        if other is None:
            diffs.append({"key": key, "missing": "live"})
            continue
        for field in ("mean", "lo95", "hi95"):
            delta = abs(float(other[field]) - float(row[field]))
            max_abs = max(max_abs, delta)
            if delta > 1e-12:
                diffs.append({"key": key, "field": field, "frozen": row[field], "live": other[field], "delta": delta})
    return {"max_abs_diff": max_abs, "differences": diffs[:20], "difference_count": len(diffs)}


def verify_hardware(
    *,
    trace_dir: Path,
    output_dir: Path,
    data_dir: Path,
    ops_count: int,
    op_start: int,
    replicates: int,
    budget: int,
    max_workers: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    ops, outages = _load_manifest_ops(data_dir / "manifest_k3_confirm.json", op_start, ops_count)
    rows, details = _load_selected_frozen(trace_dir, set(ops), budget, replicates)
    grouped = _group_rows_by_op(rows, details)
    live_rows: list[dict[str, Any]] = []
    failures = []
    totals = defaultdict(float)
    worker_reports = []
    if max_workers <= 1:
        _init_worker(str(data_dir), ops, outages)
        for op_id in ops:
            report = _verify_op_task(op_id, *grouped[op_id], budget)
            worker_reports.append(report)
    else:
        with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(str(data_dir), ops, outages)) as pool:
            futures = {pool.submit(_verify_op_task, op_id, *grouped[op_id], budget): op_id for op_id in ops}
            for future in as_completed(futures):
                worker_reports.append(future.result())
    for report in sorted(worker_reports, key=lambda item: item["op_id"]):
        live_rows.extend(report["rows"])
        failures.extend(report["failures"])
        totals["unique_physical_solves"] += report["unique_physical_solves"]
        totals["physical_requests"] += report["physical_requests"]
        totals["worker_wall_s_sum"] += report["worker_wall_s"]
        for key, value in report["totals"].items():
            totals[key] = max(totals[key], value) if key.startswith("max_") else totals[key] + value
    live_rows.sort(key=lambda row: (row["op_id"], row["cell"], row["replicate"], row["method"]))
    live_rows_path = trace_dir / "hardware_verify_live_rows.jsonl.gz"
    _write_rows(live_rows_path, live_rows)
    live_rows_sha256 = sha256(live_rows_path)
    _, comparisons = summarize(live_rows)
    live_gate = select_gate(comparisons)
    live_gate["scope"] = "hardware verification of frozen warm development gate; no fresh holdout or retraining"
    frozen_gate = json.loads((output_dir / "gate.json").read_text())
    gate_diff = _gate_diff(frozen_gate, live_gate)
    complete_frozen_scope = ops_count == 60 and replicates == DEFAULT_REPLICATES
    source_paths = [
        Path(__file__),
        Path("scripts/p5_cost_quality.py"),
        Path("src/fdna/query_budget.py"),
        Path("src/fdna/cost_quality.py"),
        Path("src/fdna/evalutil.py"),
        Path("src/fdna/v2.py"),
        Path("src/fdna/v2data.py"),
        Path("src/fdna/lp.py"),
        Path("src/fdna/grid.py"),
        Path("src/fdna/opgen.py"),
        Path("src/fdna/spec.py"),
        Path("src/fdna/physical_correction.py"),
        Path("src/fdna/dataset.py"),
    ]
    input_paths = [
        data_dir / "manifest_k3_confirm.json",
        data_dir / "n1_confirm.npz",
        data_dir / "n2_confirm.npz",
        Path("configs/spec.json"),
        output_dir / "gate.json",
        output_dir / "config.json",
        output_dir / "provenance.json",
        trace_dir / "metrics.jsonl.gz",
        trace_dir / "traces.jsonl.gz",
    ]
    input_hashes = _hash_existing(input_paths)
    report = {
        "scope": "hardware-backed verification of frozen warm cv_full budget280 candidate and mc/proxy_full baselines",
        "inputs": {
            "trace_dir": str(trace_dir),
            "data_dir": str(data_dir),
            "gate": str(output_dir / "gate.json"),
            "warm_proxy_cache_source": "frozen n1/n2 cached lower-order labels only",
            "target_label_source": "live ScenarioLP CPU solves",
            "exact_probability_source_for_mse": "frozen evaluation snapshots; realized labels and target queries are live ScenarioLP",
        },
        "selection": {
            "ops": ops,
            "budget": budget,
            "replicates": replicates,
            "methods": list(VERIFY_METHODS),
            "complete_frozen_scope": complete_frozen_scope,
        },
        "counts": {
            "frozen_rows_checked": len(rows),
            "live_rows_written": len(live_rows),
            "event_labels_compared": int(totals["event_labels_compared"]),
            "unique_physical_solves_within_op_workers": int(totals["unique_physical_solves"]),
            "physical_backend_requests_before_memoization": int(totals["physical_requests"]),
            "independent_policy_target_queries": int(totals["independent_policy_target_queries"]),
            "fresh_realized_label_checks": int(totals["fresh_realized_label_checks"]),
        },
        "mismatches": {
            "failures": failures[:100],
            "failure_count": len(failures),
            "event_missing_live": int(totals["event_missing_live"]),
            "event_missing_frozen": int(totals["event_missing_frozen"]),
            "event_value_mismatches": int(totals["event_value_mismatches"]),
            "event_severity_mismatches": int(totals["event_severity_mismatches"]),
            "realized_severity_mismatches": int(totals["realized_severity_mismatches"]),
            "query_count_mismatches": int(totals["query_count_mismatches"]),
            "score_mismatches": int(totals["score_mismatches"]),
            "metric_mismatches": int(totals["metric_mismatches"]),
        },
        "maxima": {
            "event_abs_error": float(totals["max_event_abs_error"]),
            "score_abs_diff": float(totals["max_score_abs_diff"]),
            "metric_abs_diff": float(totals["max_metric_abs_diff"]),
        },
        "regenerated_gate": live_gate,
        "gate_diff": gate_diff,
        "artifacts": {
            "live_rows": str(live_rows_path),
            "live_rows_sha256": live_rows_sha256,
            "report": str(output_dir / "HARDWARE_VERIFICATION.json"),
        },
        "provenance": {
            "git_head": _git_head(),
            "sources_sha256": _hash_existing(source_paths),
            "inputs_sha256": input_hashes,
            "development_provenance_matches": _development_provenance_matches(output_dir / "provenance.json", input_hashes),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_model": _cpu_model(),
            "package_versions": _package_versions(),
            "max_workers": max_workers,
            "thread_env": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        },
        "elapsed_s": time.perf_counter() - started,
    }
    report["artifacts"]["report_content_sha256_excluding_this_field"] = _report_content_sha256(report)
    write_json(output_dir / "HARDWARE_VERIFICATION.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("data_hik/cost_quality_dev"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/phase5/cost_quality_dev"))
    parser.add_argument("--data-dir", type=Path, default=Path("data_hik"))
    parser.add_argument("--ops", type=int, default=60, help="prefix of frozen development ops to verify")
    parser.add_argument("--op-start", type=int, default=600)
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--max-workers", type=int, default=6)
    args = parser.parse_args(argv)
    if not 1 <= args.ops <= 60:
        parser.error("--ops must be 1..60")
    if not 1 <= args.replicates <= DEFAULT_REPLICATES:
        parser.error(f"--replicates must be 1..{DEFAULT_REPLICATES}")
    if args.budget != DEFAULT_BUDGET:
        parser.error("this verifier is scoped to the frozen budget280 gate candidate")
    report = verify_hardware(
        trace_dir=args.trace_dir,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        ops_count=args.ops,
        op_start=args.op_start,
        replicates=args.replicates,
        budget=args.budget,
        max_workers=args.max_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    bad = report["mismatches"]["failure_count"]
    bad += report["mismatches"]["event_missing_live"] + report["mismatches"]["event_missing_frozen"]
    bad += report["mismatches"]["event_value_mismatches"] + report["mismatches"]["event_severity_mismatches"]
    bad += report["mismatches"]["realized_severity_mismatches"] + report["mismatches"]["query_count_mismatches"]
    bad += report["mismatches"]["score_mismatches"] + report["mismatches"]["metric_mismatches"]
    if report["selection"]["complete_frozen_scope"]:
        bad += report["gate_diff"]["difference_count"]
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
