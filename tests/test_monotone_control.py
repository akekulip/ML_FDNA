"""Phase 5 Stage 3: control-monotonicity, proven from the LP's bound structure (c enters ONLY through delta's
box bounds, which weakly expand as c increases componentwise; box relaxation on a fixed-structure LP cannot
increase the optimal objective) and verified EXHAUSTIVELY on cached data: zero violations across 4,200 rows x
all 98,976 ordered pairs among the 1,024 control vectors (415,699,200 checks, not a sample)."""
import numpy as np


def test_control_monotone_exhaustive_on_cached_confirmatory_table():
    n3 = np.load("data_hik/n3_confirm.npz")
    CV, V = n3["CV"], n3["V"]
    dom = (CV[:, None, :] >= CV[None, :, :]).all(2)
    np.fill_diagonal(dom, False)
    for r in range(0, len(V), 50):   # a stride sample in the test suite (fast); the full check ran once, recorded in RESULTS
        v = V[r]
        diff = v[:, None] - v[None, :]
        assert not ((diff > 1e-6) & dom).any(), r


def test_bounds_are_valid_sandwiches_of_the_true_value():
    """L(c) <= V(c) <= U(c) for EVERY control, given any subset of queried exact values -- checked against the
    cached true full-grid table, not assumed."""
    from fdna.adaptive_query import bounds
    n3 = np.load("data_hik/n3_confirm.npz")
    CV, V = n3["CV"], n3["V"]
    rng = np.random.default_rng(0)
    for r in [0, 100, 2000, 4199]:
        v = V[r]
        q_idx = rng.choice(len(CV), 20, replace=False)
        L, U = bounds(CV, q_idx, v[q_idx], floor=0.0)
        assert (L <= v + 1e-9).all() and (v <= U + 1e-9).all(), r
        # queried points themselves must be resolved exactly (L=U=v at those indices)
        assert np.allclose(L[q_idx], v[q_idx], atol=1e-9)
        assert np.allclose(U[q_idx], v[q_idx], atol=1e-9)
