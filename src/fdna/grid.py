"""IEEE 30-bus data (pandapower/MATPOWER case30) as plain arrays.

Ratings in the shipped case are placeholders (1e5+), so branch ratings are
derived from nominal flows in `opgen` under a frozen calibration (see prereg/SPEC.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASE_MVA = 100.0


@dataclass(frozen=True)
class Grid:
    n_bus: int
    frm: np.ndarray  # (L,) from-bus index
    to: np.ndarray  # (L,) to-bus index
    x: np.ndarray  # (L,) reactance p.u. on BASE_MVA
    gen_bus: np.ndarray  # (G,) bus of each generator; index 0 is the local balancer (slack)
    gen_pmax: np.ndarray  # (G,) MW
    load: np.ndarray  # (n_bus,) nominal MW

    @property
    def n_branch(self) -> int:
        return len(self.frm)

    @property
    def n_gen(self) -> int:
        return len(self.gen_bus)


def validate_branch_ids(n_branch: int | None, branches: tuple[int, ...]) -> tuple[int, ...]:
    """Return canonical sorted branch ids, rejecting malformed outage sets.

    `n_branch=None` validates type, sign, and uniqueness only for callers such as `hik.outage_key` that do not
    have a grid. Callers at public grid boundaries should pass `grid.n_branch` to reject out-of-range ids too.
    """
    ids = []
    for raw in branches:
        if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
            raise ValueError(f"branch id must be an integer, got {raw!r}")
        b = int(raw)
        if b < 0:
            raise ValueError(f"branch id must be nonnegative, got {b}")
        if n_branch is not None and b >= n_branch:
            raise ValueError(f"branch id {b} out of range for {n_branch} branches")
        ids.append(b)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate branch id in outage set {tuple(ids)!r}")
    return tuple(sorted(ids))


def validate_outage(grid: Grid, branches: tuple[int, ...]) -> tuple[int, ...]:
    return validate_branch_ids(grid.n_branch, branches)


def load_case30() -> Grid:
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case_ieee30()
    pp.rundcpp(net)
    ppc = net._ppc
    br = ppc["branch"]
    gen = ppc["gen"]
    bus = ppc["bus"]
    return Grid(
        n_bus=bus.shape[0],
        frm=br[:, 0].real.astype(int),
        to=br[:, 1].real.astype(int),
        x=br[:, 3].real.astype(float),
        gen_bus=gen[:, 0].real.astype(int),
        # generator limits of the case30 data (pandapower case_ieee30 keeps 360.2 on the ext_grid via max_p_mw; the ppc
        # slack row carries a 1e9 placeholder, so the limit is set explicitly here)
        gen_pmax=np.array([360.2, 140.0, 100.0, 100.0, 100.0, 100.0]),
        load=bus[:, 2].real.astype(float),
    )
