"""Evaluation-only endpoints and a fixed development selection rule.

Policies do not import this module: realized labels and severe counts belong to
the evaluator. Development selection is exploratory, not a confirmation test.
"""
from __future__ import annotations

import numpy as np

from .evalutil import rprec, recall_at, precision_at


def screening_metrics(labels, scores, keys, exact_probability):
    labels = np.asarray(labels, bool)
    scores = np.asarray(scores, float)
    exact_probability = np.asarray(exact_probability, float)
    if not (labels.shape == scores.shape == exact_probability.shape == np.asarray(keys).shape):
        raise ValueError('candidate arrays must have identical shapes')
    if not np.isfinite(scores).all() or not np.isfinite(exact_probability).all():
        raise ValueError('nonfinite predictions or probability truth')
    out = dict(n_candidates=len(labels), n_severe=int(labels.sum()),
               rprecision=float(rprec(labels, scores, keys, thr=.5)) if labels.any() else None,
               probability_mse=float(np.mean((scores - exact_probability) ** 2)),
               clipped_probability_mse=float(np.mean((np.clip(scores, 0, 1) - exact_probability) ** 2)))
    for pct in (10, 20, 40):
        out[f'recall_{pct}'] = float(recall_at(labels, scores, keys, pct / 100, thr=.5)) if labels.any() else None
        out[f'precision_{pct}'] = float(precision_at(labels, scores, keys, pct / 100, thr=.5))
    return out


def paired_summary(differences, *, n_boot=4000, seed=9025):
    """Resample op clusters and paired MC replicate columns, never candidate rows."""
    d = np.asarray(differences, float)
    if d.ndim == 1:
        d = d[:, None]
    if d.ndim != 2 or 0 in d.shape or not np.isfinite(d).all():
        raise ValueError('finite nonempty op-by-replicate matrix required')
    rng = np.random.default_rng(seed)
    n_op, n_rep = d.shape
    io = rng.integers(n_op, size=(n_boot, n_op))
    ir = rng.integers(n_rep, size=(n_boot, n_rep))
    means = d[io[:, :, None], ir[:, None, :]].mean(axis=(1, 2))
    return dict(mean=float(d.mean()), lo95=float(np.quantile(means, .05)),
                hi95=float(np.quantile(means, .95)), n_ops=n_op, n_replicates=n_rep,
                bootstrap_draws=n_boot)


def select_gate(comparisons, margin=.01):
    """Smallest budget whose paired lower bound exceeds margin in BOTH cells,
    against MC AND the same proxy without target queries. Ties use method name.
    """
    result = {}
    for regime in ('cold', 'warm'):
        rows = [r for r in comparisons if r['regime'] == regime]
        candidates = []
        for budget, method in sorted({(r['budget'], r['method']) for r in rows}):
            proxy = method.replace('cv_', 'proxy_', 1)
            needed = {(cell, baseline) for cell in ('P1_v2b', 'P2_v2c') for baseline in ('mc', proxy)}
            evidence = [r for r in rows if r['budget'] == budget and r['method'] == method]
            by_key = {(r['cell'], r['comparator']): r for r in evidence}
            if needed <= by_key.keys() and all(by_key[k]['lo95'] > margin and by_key[k]['n_ops'] > 1 for k in needed):
                candidates.append(dict(budget=budget, method=method, evidence=[by_key[k] for k in sorted(needed)]))
        result[regime] = dict(status='candidate' if candidates else 'no_go', margin=margin,
                              selected=candidates[0] if candidates else None,
                              passing_candidates=len(candidates))
    return result
