"""Phase 5 Stage R5: gate `physical_correction.capacity_floor` (the external review's proposed mechanism)
before it is used for any conclusion, the same way `island_floor` was gated against `ScenarioLP.struct_mw`."""
import numpy as np
import pytest

from fdna import physical_correction as pc
from fdna.grid import Grid
from fdna.lp import OperatingPoint, Params, ScenarioLP


def test_reviewer_toy_example_matches_exact_lp_at_20_percent():
    """Two islands after outage, each with one generator; loads 20/80 MW, base generation 50/50 MW, but the
    island-2 generator is capped at 60 MW reachable output. generator-less floor gives 0% (both islands HAVE a
    generator); the capacity floor and the repo's own exact LP both give 20% -- reproduces the review's own
    independently-verified numbers exactly, now against the actual `capacity_floor` implementation."""
    # bus 0-1 in island A (gen 0, load 20), bus 2-3 in island B (gen 1, load 80); no branch between the islands
    # (outage already removed it) -- so frm/to below is the POST-outage topology: branch 0 wires bus0-bus1,
    # branch 1 wires bus2-bus3; branch 2 (bus1-bus2, the tie) is the one that gets outaged.
    grid = Grid(
        n_bus=4, frm=np.array([0, 2, 1]), to=np.array([1, 3, 2]), x=np.array([1.0, 1.0, 1.0]),
        gen_bus=np.array([0, 3]), gen_pmax=np.array([100.0, 60.0]),
        load=np.array([20.0, 0.0, 0.0, 80.0]),
    )
    op = OperatingPoint(demand=grid.load.copy(), p0=np.array([50.0, 50.0]), rating=np.full(3, 1000.0))
    # gen 0 (local balancer, island A) needs no corrective range: p0=50 already exceeds island A's 20 MW demand.
    # gen 1 (island B) is commandable at full ramp (c=1) up to its own pmax=60 -- "capped at 60 MW reachable" --
    # so it can raise output from p0=50 to at most 60, 20 MW short of island B's 80 MW demand.
    params = Params(ramp_frac=1.0, local_mw=0.0)
    outage = (2,)  # the tie branch
    c = np.ones(1)

    f_island = pc.island_floor(grid, op.demand, outage)
    f_capacity = pc.capacity_floor(grid, op, params, outage, c)
    lp = ScenarioLP(grid, op, outage, params)
    exact = lp.y(c)

    assert f_island == pytest.approx(0.0), "both islands have a generator, so the generator-less floor is 0%"
    assert f_capacity == pytest.approx(0.20, abs=1e-9)
    assert exact == pytest.approx(0.20, abs=1e-6)


def test_capacity_floor_never_exceeds_the_exact_lp_shed(request):
    """capacity_floor ignores intra-island congestion, so it must be a valid LOWER bound on the true LP shed --
    checked on real grid data (not the toy example) across several outages and control levels, exactly as
    island_floor is already gated (F <= ScenarioLP.struct_mw exactly; capacity_floor <= the full corrective LP)."""
    from fdna.dataset import G, RATING
    from fdna.opgen import sample_op
    from fdna import spec

    op = sample_op(G, RATING, np.random.default_rng(0))
    outages = [(3,), (7, 15), (2, 9, 21)]
    controls = [np.zeros(G.n_gen - 1), np.full(G.n_gen - 1, 0.5), np.ones(G.n_gen - 1)]
    for outage in outages:
        for c in controls:
            lp = ScenarioLP(G, op, outage, spec.PARAMS)
            exact = lp.y(c)
            f_cap = pc.capacity_floor(G, op, spec.PARAMS, outage, c)
            f_isl = pc.island_floor(G, op.demand, outage)
            assert f_cap <= exact + 1e-9, (outage, c, f_cap, exact)
            assert f_cap >= f_isl - 1e-12, (outage, c, f_cap, f_isl)
