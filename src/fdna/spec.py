"""Frozen specification: parameters and failure-set splits (see prereg/SPEC.md)."""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from . import comm
from .lp import Params
from .opgen import BASE_COST, SLACK_MIN_MW, RatingSpec

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = json.loads((ROOT / "configs/spec.json").read_text())

PARAMS = Params(ramp_frac=SPEC["corrective"]["ramp_frac"], local_mw=SPEC["corrective"]["local_mw"])
RATING = RatingSpec(kappa=SPEC["rating"]["kappa"], floor_mw=SPEC["rating"]["floor_mw"])
assert list(BASE_COST) == SPEC["dispatch"]["base_cost"] and SLACK_MIN_MW == SPEC["dispatch"]["slack_must_run_mw"]
assert list(comm.SHARES) == SPEC["corrective"]["unit_shares"]

SEVERE = SPEC["severe_threshold"]
TAU = SPEC["sens_tau"]
FAILURE_SETS = comm.failure_sets(SPEC["failure_sets"]["max_size"])


def spec_hash() -> str:
    h = hashlib.sha256()
    for f in ("configs/spec.json", "prereg/SPEC.md"):
        h.update((ROOT / f).read_bytes())
    return h.hexdigest()[:16]


def fs_split(fs: tuple[int, ...]) -> str:
    """'train' | 'val' | 'test' for a failure set (deterministic hash split)."""
    if len(fs) <= 1:
        return "train"
    if len(fs) >= 3:
        return "test"
    m = int(hashlib.sha256(str(fs).encode()).hexdigest(), 16) % 100
    cuts = SPEC["failure_sets"]["pair_split_hash_mod100"]
    for name, (lo, hi) in cuts.items():
        if lo <= m < hi:
            return name
    raise AssertionError


FS_SPLIT = np.array([fs_split(f) for f in FAILURE_SETS])


def fs_ids(kind: str) -> np.ndarray:
    return np.flatnonzero(FS_SPLIT == kind)
