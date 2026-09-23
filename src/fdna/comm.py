"""Synthetic communication/control layer (documented as synthetic in prereg/SPEC.md).

Service graph: control centre (CC) -> gateways (GW) -> unit controllers (RTU) -> units.
Every remote generator has two independently commanded units with capacity shares
(0.6, 0.4). A unit is commandable iff CC is up, its RTU is up, and at least one parent
gateway is up (redundant parents are alternatives, not joint prerequisites).
Binary availability only; no bandwidth-to-MW scaling in v1.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

N_REMOTE_GEN = 5
UNITS_PER_GEN = 2
SHARES = np.array([0.6, 0.4])
N_UNIT = N_REMOTE_GEN * UNITS_PER_GEN  # 10
N_GW = 3

# component index layout: 0 = CC, 1..3 = GW0..GW2, 4..13 = RTU of unit 0..9
N_COMP = 1 + N_GW + N_UNIT  # 14
CC = 0


def gw(i: int) -> int:
    return 1 + i


def rtu(u: int) -> int:
    return 1 + N_GW + u


# parent gateways per remote generator (0-based among remote gens)
PARENT_GW = {
    0: (0,),
    1: (0,),
    2: (1, 2),  # dual-homed: redundant path
    3: (1, 2),  # dual-homed: redundant path
    4: (2,),
}


def unit_gen(u: int) -> int:
    return u // UNITS_PER_GEN


def unit_reachable(up: np.ndarray) -> np.ndarray:
    """up: (N_COMP,) bool -> (N_UNIT,) bool commandable units."""
    out = np.zeros(N_UNIT, dtype=bool)
    for u in range(N_UNIT):
        parents = PARENT_GW[unit_gen(u)]
        out[u] = up[CC] and up[rtu(u)] and any(up[gw(p)] for p in parents)
    return out


def control_fraction(up: np.ndarray) -> np.ndarray:
    """(N_REMOTE_GEN,) fraction of each remote generator's corrective range that is commandable."""
    r = unit_reachable(up).reshape(N_REMOTE_GEN, UNITS_PER_GEN)
    return (r * SHARES).sum(axis=1)


def failure_sets(max_size: int = 4) -> list[tuple[int, ...]]:
    sets: list[tuple[int, ...]] = []
    for k in range(max_size + 1):
        sets.extend(combinations(range(N_COMP), k))
    return sets


def state_of(failed: tuple[int, ...]) -> np.ndarray:
    up = np.ones(N_COMP, dtype=bool)
    up[list(failed)] = False
    return up
