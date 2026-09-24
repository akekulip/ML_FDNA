import itertools
import numpy as np
import torch

from fdna import comm, v2
from fdna.nn import inference, repair
from fdna.nn.layers import FDNALayer

W = v2.build_world(); LV = inference.level_index(W.CV)


def _pin(layer: FDNALayer):
    """alpha=1 strength irrelevant for min; beta=0 (no criticality slack), u=1: output = min over masked inputs."""
    with torch.no_grad():
        layer.alpha.fill_(20.0); layer.beta.fill_(-20.0); layer.u.fill_(20.0)


def test_or_aware_layer_represents_redundant_gateway_exactly_at_tau0():
    m = repair.ReprInf(W, LV, head="meanfield", or_aware=True, tau=0.0); _pin(m.fdna)
    X = torch.tensor(list(itertools.product([0., 1.], repeat=14)))
    o = m.unit_operability(X)
    for u in (4, 5, 6, 7):                               # dual-homed units: CC & RTU & (GW1 | GW2)
        want = X[:, comm.CC] * X[:, comm.rtu(u)] * torch.maximum(X[:, comm.gw(1)], X[:, comm.gw(2)])
        assert torch.allclose(o[:, u], want, atol=1e-3)
    for u in (0, 1, 8, 9):                               # single-homed: plain AND
        g = comm.unit_gen(u); p = comm.PARENT_GW[g][0]
        want = X[:, comm.CC] * X[:, comm.rtu(u)] * X[:, comm.gw(p)]
        assert torch.allclose(o[:, u], want, atol=1e-3)


def test_or_combine_default_max_independent_of_tau_on_fractional_inputs():
    """Phase 4 correction 4.3: binary inputs can't distinguish max from noisy-OR (both give 0/1 on {0,1} gateways).
    On FRACTIONAL inputs they differ (max(.3,.3)=.3, noisy-OR(.3,.3)=.51): the default or_combine='max' must give the
    SAME gateway-group value at tau=0 and tau=0.05, i.e. the ablation changes only softmin temperature, not OR semantics."""
    x = torch.full((1, 14), 0.7); x[0, comm.gw(1)] = 0.3; x[0, comm.gw(2)] = 0.3   # both parent gateways of gens 2,3 at 0.3
    for tau in (0.0, 0.05):
        m = repair.ReprInf(W, LV, head="meanfield", or_aware=True, tau=tau)        # default or_combine="max"
        gws = torch.stack([x[:, comm.gw(g)] for g in range(comm.N_GW)], 1)
        grp2 = gws[:, list(comm.PARENT_GW[2])].max(1).values                       # generator 2's gateway group, expected 0.3
        assert torch.allclose(grp2, torch.tensor([0.3]), atol=1e-6)
    m_noisy = repair.ReprInf(W, LV, head="meanfield", or_aware=True, tau=0.0, or_combine="noisy_or")
    assert m_noisy.or_combine == "noisy_or" and m_noisy.fdna.tau == 0.0            # opt-in only, never the ablation default


def test_flat_layer_cannot_represent_redundant_gateway():
    m = repair.ReprInf(W, LV, head="meanfield", or_aware=False, tau=0.0); _pin(m.fdna)
    x = torch.ones(1, 14); x[0, comm.gw(1)] = 0                     # one redundant gateway down, other up: truth says unit 4 is up
    assert m.unit_operability(x)[0, 4] < 0.01                        # flat min wrongly reports it down


def test_joint_head_keeps_shared_cause_correlation_meanfield_does_not():
    m = repair.ReprInf(W, LV, head="logic", prior="indep")
    up_all = int(np.flatnonzero(W.X.all(1))[0])
    cc_down = int(np.flatnonzero((~W.X[:, comm.CC]) & W.X[:, 1:].all(1))[0])
    post = torch.zeros(1, len(W.X)); post[0, up_all] = 0.5; post[0, cc_down] = 0.5
    pc = torch.zeros(1, len(W.CV)).index_add_(1, m.cidx, post)[0]
    full, none = int(np.argmax(W.CV.sum(1))), int(np.argmin(W.CV.sum(1)))
    assert abs(pc[full] - 0.5) < 1e-6 and abs(pc[none] - 0.5) < 1e-6          # joint: 50% all controls, 50% none, nothing in between
    mfc = inference.MeanFieldControls(LV)
    lp = mfc(torch.full((1, 10), 0.5))[0].exp()                               # mean-field with the same 0.5 marginals
    assert lp[full] < 0.01 and lp[none] < 0.01                                # 0.5^10 each: correlation lost


def test_meanfield_flat_repr_matches_existing_infdna():
    torch.manual_seed(0)
    old = inference.InfFDNA(LV, "fdna"); new = repair.ReprInf(W, LV, head="meanfield", or_aware=False, tau=0.05)
    new.prior.data.copy_(old.b.prior); new.w1.data.copy_(old.b.w1); new.w0.data.copy_(old.b.w0)
    new.fdna.load_state_dict(old.b.fdna.state_dict())
    e = (torch.rand(16, 13, 2) < 0.4).float()
    assert torch.allclose(old(e), new(e), atol=1e-5)


def test_logic_and_joint_heads_normalised_and_differentiable():
    for head in ("logic", "joint"):
        m = repair.ReprInf(W, LV, head=head, or_aware=True, tau=0.05, prior="cc")
        e = (torch.rand(4, 13, 2) < 0.4).float()
        lp = m(e)
        assert torch.allclose(lp.exp().sum(1), torch.ones(4), atol=1e-4)
        lp[:, 0].sum().backward()
        assert m.w1.grad is not None and torch.isfinite(m.w1.grad).all()
