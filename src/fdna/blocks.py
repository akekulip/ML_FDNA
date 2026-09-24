"""Fresh-seed block lock. A (block, slot) may be opened once; the slot must be declared in registry/slots.yaml and match the
caller's (branch, variant, cell); the git tree (tracked files) must be clean; the commit hash is recorded; the lock file and the
output hash are committed by the operator after the run."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
LOCK_DIR = ROOT / "registry" / "locks"
SLOTS_FILE = ROOT / "registry" / "slots.yaml"
BLOCKS = {"confirm": (400, 479), "replicate": (500, 579), "reserve": (600, 699)}


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def open_block(block: str, slot: str, branch: str = None, variant: str = None, cell: int = None, require_clean: bool = True) -> range:
    if block not in BLOCKS:
        raise KeyError(f"unknown block {block!r}")
    slots = yaml.safe_load(SLOTS_FILE.read_text())
    if slot not in slots:
        raise PermissionError(f"slot {slot!r} is not declared in registry/slots.yaml")
    want = slots[slot]
    got = dict(branch=branch, variant=variant, cell=cell)
    if any(got[k] is not None and got[k] != want[k] for k in want):
        raise PermissionError(f"slot {slot} is bound to {want}, caller passed {got}")
    if require_clean and _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked files have uncommitted changes; commit before opening a fresh block")
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock = LOCK_DIR / f"{block}__{slot}.lock"
    if lock.exists():
        raise RuntimeError(f"block {block} already opened for {slot}: {lock.read_text()}")
    commit = _git("rev-parse", "HEAD")
    lock.write_text(json.dumps(dict(block=block, slot=slot, commit=commit, time=time.strftime("%Y-%m-%dT%H:%M:%S"))))
    with open(LOCK_DIR / "access.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} open {block} {slot} {commit}\n")
    lo, hi = BLOCKS[block]
    return range(lo, hi + 1)


def close_block(block: str, slot: str, output: str) -> str:
    """Record the sha256 of the result file in the lock (operator commits lock + result)."""
    lock = LOCK_DIR / f"{block}__{slot}.lock"
    meta = json.loads(lock.read_text())
    meta["output"] = output
    meta["output_sha256"] = hashlib.sha256(Path(output).read_bytes()).hexdigest()
    lock.write_text(json.dumps(meta))
    return meta["output_sha256"]


def block_seed(block: str) -> int:
    """Independent draw of hidden states/observations per block (the exploratory block uses 13)."""
    return 13 if block == "test" else 13 + BLOCKS[block][0]
