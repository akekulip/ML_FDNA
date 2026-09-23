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
        # MATPOWER case30 limits; the slack row in pandapower is 1e9, use MATPOWER's 360.2
        gen_pmax=np.array([360.2, 140.0, 100.0, 100.0, 100.0, 100.0]),
        load=bus[:, 2].real.astype(float),
    )
