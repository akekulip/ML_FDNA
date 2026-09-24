"""Shared evaluation utilities: stable scenario hashing, model-independent tie-breaking, metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from . import spec

_GOLD = np.uint64(0x9E3779B97F4A7C15)


def _mix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def scenario_uniform(op, b1, b2, fs, salt: int) -> np.ndarray:
    """Deterministic U[0,1) per scenario id (op, b1, b2, fs); independent of row order."""
    op, b1, b2, fs = (np.asarray(a, dtype=np.int64) for a in (op, b1, b2, fs))
    h = np.full(op.shape, (salt * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for a in (op, b1 + 1, b2 + 1, fs):
            h = _mix(h ^ (a.astype(np.uint64) + _GOLD))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


SALT_SUBSAMPLE, SALT_TIEBREAK = 1, 2


def in_subsample(lab, frac: float) -> np.ndarray:
    return scenario_uniform(lab.op.values, lab.b1.values, lab.b2.values, lab.fs.values, SALT_SUBSAMPLE) < frac


def tiebreak_key(lab) -> np.ndarray:
    return scenario_uniform(lab.op.values, lab.b1.values, lab.b2.values, lab.fs.values, SALT_TIEBREAK)


def rank_order(yp: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Indices sorted by descending prediction; ties broken by the model-independent key."""
    return np.lexsort((key, -np.asarray(yp)))


def rprec(yt, yp, key, thr=None):
    thr = spec.SEVERE if thr is None else thr
    sev = yt > thr
    k = int(sev.sum())
    return np.nan if k == 0 else float(sev[rank_order(yp, key)[:k]].mean())


def recall_at(yt, yp, key, frac, thr=None):
    thr = spec.SEVERE if thr is None else thr
    sev = yt > thr
    if sev.sum() == 0:
        return np.nan
    k = max(1, int(round(frac * len(yt))))
    return float(sev[rank_order(yp, key)[:k]].sum() / sev.sum())


def op_metrics(yt, yp, key) -> dict:
    sev = yt > spec.SEVERE
    return dict(
        rprec=rprec(yt, yp, key),
        r10=recall_at(yt, yp, key, 0.10), r20=recall_at(yt, yp, key, 0.20), r40=recall_at(yt, yp, key, 0.40),
        mae=float(np.abs(yt - yp).mean()), prev=float(sev.mean()),
        ap=float(average_precision_score(sev, yp)) if sev.any() else np.nan,
        rho=float(spearmanr(yt, yp)[0]) if yp.std() > 0 and yt.std() > 0 else np.nan,
    )


def check_spec_hash(data_dir) -> str:
    """Fail loudly if the data were generated under a different label-defining spec."""
    from pathlib import Path

    stored = (Path(data_dir) / "spec_hash.txt").read_text().strip()
    cur = spec.spec_hash()
    if stored != cur:
        raise RuntimeError(f"spec hash mismatch: data={stored} current={cur}; regenerate data or revert spec")
    return cur
