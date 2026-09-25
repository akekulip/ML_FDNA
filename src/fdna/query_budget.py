from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import numpy as np

from fdna.adaptive_query import bounds, posterior_mass_bounds


class BudgetExceeded(RuntimeError):
    """Raised before an uncached oracle query would exceed the query budget."""


def _canonical_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a nonnegative integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _canonical_outage(outage: Iterable[Any]) -> tuple[int, ...]:
    if isinstance(outage, (str, bytes)):
        raise ValueError("outage must be an iterable of distinct nonnegative integers")
    try:
        items = tuple(_canonical_nonnegative_int(x, "outage item") for x in outage)
    except TypeError as exc:
        raise ValueError("outage must be an iterable of distinct nonnegative integers") from exc
    if len(items) == 0:
        raise ValueError("outage must be nonempty")
    if len(set(items)) != len(items):
        raise ValueError("outage must contain distinct items")
    return tuple(sorted(items))


def _finite_float(value: Any, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _cache_key(op_id: Any, outage: Iterable[Any], control_idx: Any) -> tuple[int, tuple[int, ...], int]:
    return (
        _canonical_nonnegative_int(op_id, "op_id"),
        _canonical_outage(outage),
        _canonical_nonnegative_int(control_idx, "control_idx"),
    )


class BudgetedOracle:
    def __init__(
        self,
        backend: Callable[[int, tuple[int, ...], int], float],
        budget: int,
        initial_cache: Mapping[tuple[Any, Iterable[Any], Any], Any] | None = None,
    ) -> None:
        if not callable(backend):
            raise ValueError("backend must be callable")
        self._backend = backend
        self._budget = _canonical_nonnegative_int(budget, "budget")
        self.cache: dict[tuple[int, tuple[int, ...], int], float] = {}
        if initial_cache is not None:
            for key, value in initial_cache.items():
                if len(key) != 3:
                    raise ValueError("initial_cache keys must be (op_id, outage, control_idx)")
                self.cache[_cache_key(key[0], key[1], key[2])] = _finite_float(value, "cached value")
        self.events: list[dict[str, Any]] = []
        self.spent = 0

    @property
    def remaining(self) -> int:
        return self._budget - self.spent

    def query(self, op_id: int, outage: Iterable[Any], control_idx: int) -> float:
        key = _cache_key(op_id, outage, control_idx)
        if key in self.cache:
            return self.cache[key]
        if self.spent >= self._budget:
            raise BudgetExceeded("query budget exceeded")

        value = _finite_float(self._backend(*key), "oracle label")
        self.cache[key] = value
        self.spent += 1
        event = {"op_id": key[0], "outage": key[1], "control_idx": key[2], "value": value}
        self.events.append(event)
        return value


def _probability_vector(pc: Any, expected_len: int | None = None) -> np.ndarray:
    arr = np.asarray(pc, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("pc must be a nonempty one-dimensional probability vector")
    if expected_len is not None and len(arr) != expected_len:
        raise ValueError("pc length does not match the control grid")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("pc must contain finite nonnegative probabilities")
    if not np.isclose(float(arr.sum()), 1.0):
        raise ValueError("pc must sum to 1")
    return arr


def _control_grid(CV: Any) -> np.ndarray:
    arr = np.asarray(CV, dtype=float)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("CV must be a nonempty two-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("CV must be finite")
    return arr


def _g2_value(oracle: BudgetedOracle, op_id: int, outage: tuple[int, ...], control_idx: int) -> float:
    pairs = sum(oracle.query(op_id, pair, control_idx) for pair in itertools.combinations(outage, 2))
    singles = sum(oracle.query(op_id, (item,), control_idx) for item in outage)
    return pairs - singles


def g2_proxy(
    oracle: BudgetedOracle,
    op_id: int,
    outages: Iterable[Iterable[Any]],
    CV: Any,
    pc: Any,
    n_anchors: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    controls = _control_grid(CV)
    posterior = _probability_vector(pc, expected_len=len(controls))
    op = _canonical_nonnegative_int(op_id, "op_id")
    outage_rows = [_canonical_outage(outage) for outage in outages]
    if not outage_rows:
        raise ValueError("outages must be nonempty")
    for outage in outage_rows:
        if len(outage) != 3:
            raise ValueError("g2_proxy expects triple outages")

    if n_anchors is None:
        anchors = np.arange(len(controls), dtype=int)
    else:
        n = _canonical_nonnegative_int(n_anchors, "n_anchors")
        if n == 0 or n > len(controls):
            raise ValueError("n_anchors must be between 1 and len(CV)")
        selected = np.argsort(-posterior, kind="stable")[:n]
        anchors = np.array(sorted(int(idx) for idx in selected), dtype=int)

    anchor_grid = np.empty((len(outage_rows), len(anchors)), dtype=float)
    for row_idx, outage in enumerate(outage_rows):
        for anchor_col, control_idx in enumerate(anchors):
            anchor_grid[row_idx, anchor_col] = _g2_value(oracle, op, outage, int(control_idx))

    if len(anchors) == len(controls) and np.array_equal(anchors, np.arange(len(controls))):
        return anchor_grid, anchors

    grid = np.empty((len(outage_rows), len(controls)), dtype=float)
    anchor_controls = controls[anchors]
    for control_idx, control in enumerate(controls):
        distances = np.abs(anchor_controls - control).sum(axis=1)
        nearest_anchor_col = int(np.argmin(distances))
        grid[:, control_idx] = anchor_grid[:, nearest_anchor_col]
    return grid, anchors


def estimate_probability(
    oracle: BudgetedOracle,
    op_id: int,
    outage: Iterable[Any],
    pc: Any,
    n_draws: int,
    rng: Any,
    proxy: Any | None = None,
    beta: float = 1.0,
    tau: float = 0.01,
) -> float:
    op = _canonical_nonnegative_int(op_id, "op_id")
    canonical_outage = _canonical_outage(outage)
    posterior = _probability_vector(pc)
    draws_count = _canonical_nonnegative_int(n_draws, "n_draws")
    if draws_count == 0:
        raise ValueError("n_draws must be positive")
    beta_value = _finite_float(beta, "beta")
    tau_value = _finite_float(tau, "tau")

    proxy_indicators = None
    proxy_expectation = None
    if proxy is not None:
        proxy_values = np.asarray(proxy, dtype=float)
        if proxy_values.shape != posterior.shape or not np.all(np.isfinite(proxy_values)):
            raise ValueError("proxy must be a finite vector with the same shape as pc")
        proxy_indicators = (proxy_values > tau_value).astype(float)
        proxy_expectation = float(np.dot(posterior, proxy_indicators))

    if not hasattr(rng, "choice"):
        raise ValueError("rng must provide choice")
    draws = np.asarray(rng.choice(len(posterior), size=draws_count, replace=True, p=posterior), dtype=int)
    if draws.shape != (draws_count,) or np.any(draws < 0) or np.any(draws >= len(posterior)):
        raise ValueError("rng.choice returned invalid draw indices")

    unique_draws, inverse = np.unique(draws, return_inverse=True)
    unique_truth = np.array(
        [oracle.query(op, canonical_outage, int(control_idx)) > tau_value for control_idx in unique_draws],
        dtype=float,
    )
    truth = unique_truth[inverse]
    if proxy_indicators is None:
        return float(truth.mean())

    proxy_at_draws = proxy_indicators[draws]
    return float(beta_value * proxy_expectation + np.mean(truth - beta_value * proxy_at_draws))


def _midpoint_probability(posterior: np.ndarray, L: np.ndarray, U: np.ndarray, tau: float) -> float:
    qL, qU = posterior_mass_bounds(posterior, L, U, tau)
    return float((qL + qU) / 2.0)


def _choose_random_index(candidates: np.ndarray, rng: Any) -> int:
    if not hasattr(rng, "choice") and not hasattr(rng, "integers"):
        raise ValueError("rng must provide choice or integers")
    if hasattr(rng, "choice"):
        choice = rng.choice(len(candidates), size=1, replace=True, p=np.full(len(candidates), 1.0 / len(candidates)))
        return int(candidates[int(np.asarray(choice).reshape(-1)[0])])
    return int(candidates[int(rng.integers(len(candidates)))])


def bound_probability(
    oracle: BudgetedOracle,
    op_id: int,
    outage: Iterable[Any],
    CV: Any,
    pc: Any,
    max_queries: int,
    *,
    floor: float = 0.0,
    proxy: Any | None = None,
    rng: Any,
    tau: float = 0.01,
) -> float:
    controls = _control_grid(CV)
    posterior = _probability_vector(pc, expected_len=len(controls))
    op = _canonical_nonnegative_int(op_id, "op_id")
    canonical_outage = _canonical_outage(outage)
    query_cap = _canonical_nonnegative_int(max_queries, "max_queries")
    floor_value = _finite_float(floor, "floor")
    tau_value = _finite_float(tau, "tau")

    proxy_values = None
    if proxy is not None:
        proxy_values = np.asarray(proxy, dtype=float)
        if proxy_values.shape != posterior.shape or not np.all(np.isfinite(proxy_values)):
            raise ValueError("proxy must be a finite vector with the same shape as pc")

    queried_idx: list[int] = []
    queried_val: list[float] = []
    L, U = bounds(controls, np.array([], dtype=int), np.array([], dtype=float), floor_value)
    if query_cap == 0:
        return _midpoint_probability(posterior, L, U, tau_value)

    extremes: list[int] = []
    for idx in (int(np.argmax(controls.sum(axis=1))), int(np.argmin(controls.sum(axis=1)))):
        if idx not in extremes:
            extremes.append(idx)

    for idx in extremes:
        if len(queried_idx) >= query_cap:
            return _midpoint_probability(posterior, L, U, tau_value)
        queried_idx.append(idx)
        queried_val.append(oracle.query(op, canonical_outage, idx))
        L, U = bounds(controls, np.asarray(queried_idx, dtype=int), np.asarray(queried_val, dtype=float), floor_value)
        qL, qU = posterior_mass_bounds(posterior, L, U, tau_value)
        if qL > 0.5 or qU <= 0.5:
            return float((qL + qU) / 2.0)

    while len(queried_idx) < query_cap:
        support = np.flatnonzero(posterior > 0.0)
        unresolved = support[(L[support] <= tau_value) & (U[support] > tau_value)]
        candidates = np.array([int(idx) for idx in unresolved if int(idx) not in queried_idx], dtype=int)
        if len(candidates) == 0:
            break

        if proxy_values is not None:
            distances = np.abs(proxy_values[candidates] - tau_value)
            idx = int(candidates[int(np.argmin(distances))])
        else:
            idx = _choose_random_index(candidates, rng)

        queried_idx.append(idx)
        queried_val.append(oracle.query(op, canonical_outage, idx))
        L, U = bounds(controls, np.asarray(queried_idx, dtype=int), np.asarray(queried_val, dtype=float), floor_value)
        qL, qU = posterior_mass_bounds(posterior, L, U, tau_value)
        if qL > 0.5 or qU <= 0.5:
            return float((qL + qU) / 2.0)

    return _midpoint_probability(posterior, L, U, tau_value)
