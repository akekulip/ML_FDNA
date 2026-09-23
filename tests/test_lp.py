import cvxpy as cp
import numpy as np
import pytest

from fdna import comm
from fdna.grid import BASE_MVA, load_case30
from fdna.lp import Params, ScenarioLP
from fdna.opgen import RatingSpec, nominal_rating, sample_op

PRM = Params(ramp_frac=0.5, local_mw=30.0)


@pytest.fixture(scope="module")
def world():
    g = load_case30()
    rating = nominal_rating(g, RatingSpec(1.3, 20.0))
    rng = np.random.default_rng(123)
    ops = [sample_op(g, rating, rng) for _ in range(3)]
    return g, ops


def cvx_shed(g, op, removed, c, prm):
    """Independent cvxpy formulation (explicit branch-flow variables)."""
    n, G = g.n_bus, g.n_gen
    alive = [k for k in range(g.n_branch) if k not in removed]
    th = cp.Variable(n)
    s = cp.Variable(n)
    dl = cp.Variable(G)
    kp = cp.Variable(G)
    f = {k: (BASE_MVA / g.x[k]) * (th[g.frm[k]] - th[g.to[k]]) for k in alive}
    # islands via union-find
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for k in alive:
        par[find(g.frm[k])] = find(g.to[k])
    roots = sorted({find(i) for i in range(n)})
    cons = [s >= 0, s <= op.demand, kp >= 0, kp <= op.p0]
    rng_ = np.r_[prm.local_mw, prm.ramp_frac * g.gen_pmax[1:] * np.asarray(c)]
    cons += [dl >= -np.minimum(rng_, op.p0), dl <= np.minimum(rng_, g.gen_pmax - op.p0)]
    P = op.p0 + dl - kp
    cons += [P >= 0, P <= g.gen_pmax]
    for b in range(n):
        inj = sum(P[i] for i in range(G) if g.gen_bus[i] == b) - (op.demand[b] - s[b])
        out = sum(f[k] for k in alive if g.frm[k] == b) - sum(f[k] for k in alive if g.to[k] == b)
        cons.append(inj == out)
    for k in alive:
        cons += [f[k] <= op.rating[k], f[k] >= -op.rating[k]]
    for r in roots:
        members = [i for i in range(n) if find(i) == r]
        gens = [i for i in range(G) if find(g.gen_bus[i]) == r]
        surplus = max(sum(op.p0[i] for i in gens) - sum(op.demand[i] for i in members), 0.0)
        cons.append(sum(kp[i] for i in gens) - sum(s[i] for i in members) <= surplus)
        cons.append(th[members[0]] == 0)
    prob = cp.Problem(cp.Minimize(cp.sum(s)), cons)
    prob.solve(solver=cp.HIGHS)
    return prob.value


def test_n0_needs_no_shed(world):
    g, ops = world
    for op in ops:
        assert ScenarioLP(g, op, (), PRM).y(np.ones(5)) == pytest.approx(0.0, abs=1e-9)


def test_control_monotonicity(world):
    g, ops = world
    rng = np.random.default_rng(1)
    for op in ops:
        for _ in range(40):
            removed = tuple(rng.choice(g.n_branch, size=rng.integers(1, 3), replace=False))
            lp = ScenarioLP(g, op, removed, PRM)
            lo = rng.choice([0.0, 0.4, 0.6, 1.0], size=5)
            hi = np.maximum(lo, rng.choice([0.0, 0.4, 0.6, 1.0], size=5))
            assert lp.y(hi) <= lp.y(lo) + 1e-9


def test_shed_at_least_structural(world):
    g, ops = world
    rng = np.random.default_rng(2)
    for op in ops:
        for _ in range(60):
            removed = tuple(rng.choice(g.n_branch, size=2, replace=False))
            lp = ScenarioLP(g, op, removed, PRM)
            assert lp.y(np.ones(5)) >= lp.struct_mw / lp.total - 1e-9


def test_matches_independent_cvxpy(world):
    g, ops = world
    rng = np.random.default_rng(3)
    checked = 0
    for op in ops:
        for _ in range(12):
            removed = tuple(rng.choice(g.n_branch, size=rng.integers(1, 3), replace=False))
            c = rng.choice([0.0, 0.4, 0.6, 1.0], size=5)
            a = ScenarioLP(g, op, removed, PRM).shed_mw(c)
            b = cvx_shed(g, op, removed, c, PRM)
            assert a == pytest.approx(b, abs=1e-5)
            checked += 1
    assert checked == 36


def test_comm_reachability():
    up = np.ones(comm.N_COMP, dtype=bool)
    assert comm.control_fraction(up).tolist() == [1.0] * 5
    up[comm.CC] = False
    assert comm.control_fraction(up).sum() == 0
    up = np.ones(comm.N_COMP, dtype=bool)
    up[comm.gw(1)] = False  # dual-homed gens 2,3 keep control via GW2
    assert comm.control_fraction(up).tolist() == [1.0] * 5
    up[comm.gw(2)] = False  # now gens 2,3,4 lost
    assert comm.control_fraction(up).tolist() == [1.0, 1.0, 0.0, 0.0, 0.0]
    up = np.ones(comm.N_COMP, dtype=bool)
    up[comm.rtu(0)] = False
    assert comm.control_fraction(up)[0] == pytest.approx(0.4)
