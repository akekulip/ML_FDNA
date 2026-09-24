"""Pre-declared confirmatory decision rules. Statistics fixed after the H5 audit: margin tests (H0: effect <= margin), p-value floor
(1+count)/(B+1), two-way bootstrap over test operating points AND training replicates (training noise is not ignored), NaNs counted
not silently dropped, explicit family size."""
from __future__ import annotations

import numpy as np

SESOI = 0.05
NEGLIGIBLE = 0.02
ALPHA = 0.05


def _as_matrix(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[:, None] if x.ndim == 1 else x


def boot(diff, margin: float = 0.0, n_boot: int = 20000, seed: int = 0) -> dict:
    """diff: (operating points x replicates) paired differences (a vector is treated as one replicate).
    Returns mean, one-sided p for H0: mean <= margin, one-sided 95% lower bound and 90% two-sided interval."""
    d = _as_matrix(diff)
    n_nan = int(np.isnan(d).sum())
    if n_nan:
        raise ValueError(f"{n_nan} NaN paired differences: refusing to drop them silently")
    rng = np.random.default_rng(seed)
    n_op, n_rep = d.shape
    io = rng.integers(0, n_op, (n_boot, n_op))
    ir = rng.integers(0, n_rep, (n_boot, n_rep))
    m = np.empty(n_boot)
    for b in range(n_boot):
        m[b] = d[np.ix_(io[b], ir[b])].mean()
    p = (1 + int((m <= margin).sum())) / (n_boot + 1)
    return dict(mean=float(d.mean()), p=p, lo95_one_sided=float(np.percentile(m, 5)), lo90=float(np.percentile(m, 5)), hi90=float(np.percentile(m, 95)))


def holm(pvals: dict, alpha: float = ALPHA, expected_family: int = None) -> dict:
    if expected_family is not None and len(pvals) != expected_family:
        raise ValueError(f"family size {len(pvals)} != declared {expected_family}")
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def intersection_union(components: dict, margin_per_component: dict = None, **kw) -> dict:
    """Every component must reject its own H0 (mean <= margin); the joint p is the maximum."""
    margin_per_component = margin_per_component or {}
    stats = {k: boot(v, margin=margin_per_component.get(k, 0.0), **kw) for k, v in components.items()}
    return dict(stats=stats, p=max(s["p"] for s in stats.values()), all_positive=all(s["mean"] > 0 for s in stats.values()))


def tost_negligible(diff, bound: float = NEGLIGIBLE, **kw) -> dict:
    """Equivalence: 90% two-sided interval inside (-bound, +bound)."""
    s = boot(diff, **kw)
    return dict(mean=s["mean"], lo90=s["lo90"], hi90=s["hi90"], negligible=bool(s["lo90"] > -bound and s["hi90"] < bound))
