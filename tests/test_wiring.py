import numpy as np
import torch

from fdna import comm, spec, wiring
from fdna.nn.service import ServiceGraph, ServiceValueNet


def test_true_wiring_reproduces_control_fractions():
    C = wiring.control_vectors_under(wiring.TRUE_PARENTS, wiring.TRUE_UNIT_GEN)
    ref = np.array([comm.control_fraction(comm.state_of(f)) for f in spec.FAILURE_SETS])
    np.testing.assert_allclose(C, ref, atol=1e-9)


def test_corruption_changes_requested_edge_count_and_is_deterministic():
    for rho in (0.1, 0.2, 0.3):
        p1, u1 = wiring.corrupt(rho, 5)
        p2, u2 = wiring.corrupt(rho, 5)
        assert p1 == p2 and (u1 == u2).all()
        changed = sum(a != b for g in p1 for a, b in zip(p1[g], wiring.TRUE_PARENTS[g])) + int((u1 != wiring.TRUE_UNIT_GEN).sum())
        assert changed == round(rho * wiring.N_EDGES)
    p, u = wiring.corrupt(0.0, 1)
    assert p == wiring.TRUE_PARENTS and (u == wiring.TRUE_UNIT_GEN).all()


def test_service_graph_at_true_wiring_matches_exact_calculation():
    g = ServiceGraph(wiring.TRUE_PARENTS, wiring.TRUE_UNIT_GEN, learn_edges=False, init=12.0)
    flags = torch.tensor([comm.state_of(f).astype(np.float32) for f in spec.FAILURE_SETS[:200]])
    ref = np.array([comm.control_fraction(comm.state_of(f)) for f in spec.FAILURE_SETS[:200]])
    np.testing.assert_allclose(g(flags).detach().numpy(), ref, atol=1e-3)
    assert g.edge_recovery() == (1.0, 1.0)


def test_service_value_net_runs_and_edges_learnable():
    p, u = wiring.corrupt(0.3, 2)
    net = ServiceValueNet(p, u, d_x=6)
    x, flags = torch.randn(16, 6), torch.randint(0, 2, (16, 14)).float()
    net(x, flags).sum().backward()
    assert net.svc.a.grad is not None and net.svc.w.grad is not None
