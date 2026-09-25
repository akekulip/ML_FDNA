"""Phase 5 Stage 2: V(empty)=0 is PROVEN from the generator, not assumed. Proof (see docstring below), verified
here across many ops and many random control vectors -- not just the one control state used elsewhere."""
import numpy as np

from fdna import spec
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

"""
Proof: nominal_rating() calibrates `rating` from the BASE-CASE (N-0) dispatch flows of a reference cost vector
(kappa x |nominal flow|, floor). sample_op() only accepts a dispatch (op.p0, via `_dc_dispatch(..., rating)`)
that is ALREADY flow-feasible under that SAME `rating` array (it re-solves with the rating constraint applied
and rejects infeasible draws). So for the accepted op, its own base-case flows satisfy `rating` by construction.

At N-0 (removed=()), ScenarioLP's equality constraint is `L @ theta - s - gen_inc@delta + gen_inc@kappa = p0_bus
- demand`. Setting s=delta=kappa=0 (all within their bounds for ANY c, since lb<=0<=ub always) reduces this to
`L @ theta = p0_bus - demand`, IDENTICAL to `_dc_dispatch`'s own equality with P=op.p0. Since op.p0 was accepted
specifically because that equation has a rating-feasible solution, the SAME theta (hence SAME flows) solves
ScenarioLP's flow constraints too, for any control c (c only affects delta/kappa bounds, and delta=kappa=0 is
always inside those bounds). So s=0 is feasible, hence optimal (objective = sum(s) >= 0), for EVERY c: V(empty,c)=0.
"""


def test_v_empty_is_exactly_zero_for_many_ops_and_random_controls():
    rng = np.random.default_rng(0)
    for op_id in range(15):
        op = sample_op(G, RATING, np.random.default_rng(op_id))
        lp = ScenarioLP(G, op, (), spec.PARAMS)
        for _ in range(10):
            c = rng.uniform(0, 1, size=5)
            assert abs(lp.y(c)) < 1e-9, (op_id, c, lp.y(c))
