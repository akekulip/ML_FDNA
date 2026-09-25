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


def precision_at(yt, yp, key, frac, thr=None):
    """Precision within the top `frac` of the ranked shortlist (sibling of `recall_at`; needed for the
    adaptive-query registry's declared "shortlist recall/precision at 10/20/40% budgets" metric, which the
    original experiment never computed)."""
    thr = spec.SEVERE if thr is None else thr
    sev = yt > thr
    k = max(1, int(round(frac * len(yt))))
    return float(sev[rank_order(yp, key)[:k]].mean())


def sample_posterior_and_truth(w, q, s, cov, op_id, outage, seed, K_DRAWS=50):
    """One P1/P2-style posterior sampling + observation-conditioning draw for one (op_id, outage), shared
    across every arm's predictions -- factored out of p5_matched_recurrent_eval_v2.py (repair round 2) so a
    training-time diagnostic and the real eval script use the identical, already-tested composition logic
    instead of two copies that could silently drift apart. Returns (pc, true_c_idx, key); `key` is the same
    model-independent tie-break key `rprec` expects."""
    from . import v2, v2data
    rng = np.random.default_rng([op_id, *outage, seed])
    st = v2.sample_states(w, rng, K_DRAWS)
    obs = v2.emit(w, st, q, s, rng, cov)
    pc = v2data.oracle_features(w, obs, s)[0]
    true_c_idx = w.cidx[st]
    key = scenario_uniform(np.full(K_DRAWS, op_id), np.full(K_DRAWS, outage[0]),
                            np.arange(K_DRAWS) + hash(outage) % 100000, np.arange(K_DRAWS), salt=13)
    return pc, true_c_idx, key


def evaluate_rprec_for_preds(w, q, s, cov, op_id, outage, y_true_grid, preds_grid, seed, K_DRAWS=50):
    """Posterior-compose ONE arm's `preds_grid` (predictions over the full CV control grid for this
    (op_id,outage)) with the cell's own sampling, returning (composed_pred, y_true, key) rows ready to be
    concatenated across (op_id,outage) pairs and passed to `rprec`/`op_metrics`."""
    pc, true_c_idx, key = sample_posterior_and_truth(w, q, s, cov, op_id, outage, seed, K_DRAWS)
    pred = (pc * preds_grid).sum(1)
    y_true = y_true_grid[true_c_idx]
    return pred, y_true, key


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
