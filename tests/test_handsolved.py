"""Hand-solvable LP cases (tiny grids with closed-form answers)."""
import numpy as np
import pytest

from fdna.grid import Grid
from fdna.lp import OperatingPoint, Params, ScenarioLP


def grid(n_bus, frm, to, gen_bus, load):
    return Grid(n_bus=n_bus, frm=np.array(frm), to=np.array(to), x=np.full(len(frm), 0.1),
                gen_bus=np.array(gen_bus), gen_pmax=np.full(len(gen_bus), 200.0), load=np.array(load, float))


def y(g, op, removed, a, params=Params(0.3, 40.0)):
    return ScenarioLP(g, op, removed, params).y(np.array([a], float))


G2 = grid(2, [0], [1], [0, 1], [0, 100])


def op2(rating):
    return OperatingPoint(demand=np.array([0.0, 100.0]), p0=np.array([60.0, 40.0]), rating=np.array([rating]))


def test_no_stress_no_shed():
    assert y(G2, op2(1000.0), (), 0.0) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("a,expected", [(1.0, 0.0), (0.0, 0.20), (0.1, 0.14), (0.4, 0.0)])
def test_overload_relieved_only_through_remote_redispatch(a, expected):
    # line limit 40 < flow 60: G0 (local) must drop 20 MW and remote G1 must pick it up (range 0.3*200*a)
    assert y(G2, op2(40.0), (), a) == pytest.approx(expected, abs=1e-7)


def test_local_range_binding_forces_shed_and_trip():
    # local range only 5 MW: trip (kappa) must be matched by shed => 15 MW shed even with full remote control
    assert y(G2, op2(40.0), (), 1.0, Params(0.3, 5.0)) == pytest.approx(0.15, abs=1e-7)


@pytest.mark.parametrize("a,expected", [(1.0, 0.0), (0.0, 0.40)])
def test_island_deficit_needs_remote_control(a, expected):
    # line removed: island0 has surplus 40 (trips freely), island1 is short by 40 and only G1 can help
    op = OperatingPoint(demand=np.array([20.0, 80.0]), p0=np.array([60.0, 40.0]), rating=np.array([1000.0]))
    assert y(G2, op, (0,), a) == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("a", [0.0, 1.0])
def test_gen_less_island_is_structural_shed(a):
    g = grid(3, [0, 1], [1, 2], [0, 0], [0, 20, 80])
    op = OperatingPoint(demand=np.array([0.0, 20.0, 80.0]), p0=np.array([60.0, 40.0]), rating=np.array([1e3, 1e3]))
    lp = ScenarioLP(g, op, (1,), Params(0.3, 40.0))
    assert lp.struct_mw == pytest.approx(80.0)
    assert lp.y(np.array([a])) == pytest.approx(0.80, abs=1e-7)
