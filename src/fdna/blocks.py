"""Fresh-seed block lock: each (block, hypothesis) may be opened once; every access is logged."""
from __future__ import annotations

import json
import time
from pathlib import Path

LOCK_DIR = Path(__file__).resolve().parents[2] / "registry" / "locks"
BLOCKS = {"confirm": (400, 479), "replicate": (500, 579), "reserve": (600, 699)}


def open_block(block: str, hypothesis: str, commit: str = "") -> range:
    if block not in BLOCKS:
        raise KeyError(f"unknown block {block!r}")
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOCK_DIR / f"{block}__{hypothesis}.lock"
    if lock.exists():
        raise RuntimeError(f"block {block} already opened for {hypothesis}: {lock.read_text()}")
    lock.write_text(json.dumps(dict(block=block, hypothesis=hypothesis, commit=commit, time=time.strftime("%Y-%m-%dT%H:%M:%S"))))
    with open(LOCK_DIR / "access.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} open {block} {hypothesis} {commit}\n")
    lo, hi = BLOCKS[block]
    return range(lo, hi + 1)
