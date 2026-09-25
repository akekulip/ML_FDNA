from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import hashlib

import numpy as np

from fdna.rules import boot


def seeded_arm(prefix: str, seed: int) -> str:
    return f"{prefix}_seed{int(seed)}"


def seed_arms(prefix: str, seeds: Iterable[int]) -> list[str]:
    return [seeded_arm(prefix, seed) for seed in seeds]


def summarize_seeded_family(
    prefix: str,
    seeds: Iterable[int],
    per_op_rprec: Mapping[int, Mapping[str, float]],
    references: Iterable[str],
    bootstrap_kwargs: dict | None = None,
) -> dict:
    """Summarize one seeded arm family without collapsing training-replicate noise.

    The bootstrap input is operating points x seeds, matching fdna.rules.boot's two-way
    resampling contract.
    """
    seeds = tuple(int(seed) for seed in seeds)
    arms = seed_arms(prefix, seeds)
    op_ids = sorted(per_op_rprec)
    scores = np.array([[per_op_rprec[op][arm] for arm in arms] for op in op_ids], dtype=float)
    per_seed_mean = {f"seed{seed}": float(scores[:, i].mean()) for i, seed in enumerate(seeds)}
    bootstrap_kwargs = dict(bootstrap_kwargs or {})

    boot_vs = {}
    for reference in references:
        reference_scores = _reference_matrix(reference, seeds, op_ids, per_op_rprec)
        stats = boot(scores - reference_scores, **bootstrap_kwargs)
        stats["matrix_shape"] = list(scores.shape)
        boot_vs[reference] = stats

    return {
        "arms": arms,
        "op_ids": [int(op) for op in op_ids],
        "per_seed_mean": per_seed_mean,
        "mean": float(scores.mean()),
        "std": float(np.std(list(per_seed_mean.values()))),
        "boot_vs": boot_vs,
    }


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference_matrix(reference: str, seeds: tuple[int, ...], op_ids, per_op_rprec) -> np.ndarray:
    seeded = [seeded_arm(reference, seed) for seed in seeds]
    has_seeded = all(all(arm in per_op_rprec[op] for arm in seeded) for op in op_ids)
    if has_seeded:
        return np.array([[per_op_rprec[op][arm] for arm in seeded] for op in op_ids], dtype=float)
    return np.array([[per_op_rprec[op][reference]] * len(seeds) for op in op_ids], dtype=float)
