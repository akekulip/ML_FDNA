"""Pre-declared confirmatory decision rules (registry/addendum_H4.yaml): point estimate >= SESOI AND Holm-adjusted one-sided
bootstrap p < 0.05 over the declared family; structure claims are intersection-unions over their components."""
from __future__ import annotations

import numpy as np

SESOI = 0.05
ALPHA = 0.05


def boot_p(diff: np.ndarray, n_boot: int = 20000, seed: int = 0) -> tuple[float, float, float, float]:
    """diff: per-operating-point paired differences. Returns (mean, p one-sided H0: mean <= 0, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    diff = diff[~np.isnan(diff)]
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float((m <= 0).mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone)."""
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def claim(components: dict[str, np.ndarray], effect_key: str) -> dict:
    """components: name -> per-op paired differences. The claim needs every component to have a positive mean and the
    effect component to reach SESOI; its p-value is the maximum over components (intersection-union)."""
    stats = {k: boot_p(v) for k, v in components.items()}
    p = max(s[1] for s in stats.values())
    ok_mean = all(s[0] > 0 for s in stats.values()) and stats[effect_key][0] >= SESOI
    return dict(stats=stats, p=p, effect_ok=ok_mean)


def verdicts(claims: dict[str, dict]) -> dict[str, dict]:
    padj = holm({k: c["p"] for k, c in claims.items()})
    return {k: dict(padj=padj[k], supported=bool(c["effect_ok"] and padj[k] < ALPHA), **c) for k, c in claims.items()}
