import numpy as np
import torch

from fdna.nn.layers import FDNALayer, MonotoneValue, softmin


def fdna_ref(o, m, a, b, u):
    out = np.zeros((len(o), m.shape[0]))
    for j in range(m.shape[0]):
        idx = np.flatnonzero(m[j])
        s = 1 - np.mean(a[j, idx] * (1 - o[:, idx]), axis=1)
        bt = np.minimum(1, np.min(o[:, idx] + b[j, idx], axis=1))
        out[:, j] = np.minimum(np.minimum(u[j], s), bt)
    return out


def test_fdna_layer_matches_reference_at_zero_temperature():
    torch.manual_seed(0)
    m = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    L = FDNALayer(m, tau=0.0)
    o = torch.rand(6, 3)
    ref = fdna_ref(o.numpy(), m.numpy(), torch.sigmoid(L.alpha).detach().numpy(), torch.sigmoid(L.beta).detach().numpy(), torch.sigmoid(L.u).detach().numpy())
    np.testing.assert_allclose(L(o).detach().numpy(), np.clip(ref, 0, 1), atol=1e-5)


def test_fdna_layer_soft_min_gradients_flow_and_bounded():
    m = torch.ones(4, 5)
    L = FDNALayer(m, tau=0.05)
    o = torch.rand(8, 5, requires_grad=True)
    out = L(o)
    assert ((out >= 0) & (out <= 1)).all()
    out.sum().backward()
    assert o.grad.abs().sum() > 0 and L.alpha.grad.abs().sum() > 0 and L.beta.grad.abs().sum() > 0


def test_softmin_approaches_min():
    x = torch.tensor([[0.3, 0.7, 0.5]])
    assert abs(softmin(x, 1, 0.0001).item() - 0.3) < 1e-3 and abs(softmin(x, 1, 0.0).item() - 0.3) < 1e-6


def test_monotone_value_is_nonincreasing_in_control():
    torch.manual_seed(1)
    net = MonotoneValue(d_x=10, n_c=5)
    x = torch.randn(64, 10)
    c = torch.rand(64, 5)
    for k in range(5):
        c2 = c.clone(); c2[:, k] = c[:, k] + torch.rand(64) * 0.5
        assert (net(x, c2) <= net(x, c) + 1e-6).all()


def test_fdna_layer_reaches_one_when_everything_is_healthy():
    m = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]])
    L = FDNALayer(m, tau=0.05)
    L.u.data.fill_(20.0); L.alpha.data.fill_(-20.0); L.beta.data.fill_(20.0)
    assert (L(torch.ones(4, 4)) > 0.999).all()          # un-normalised softmin capped this at ~0.95
    L.alpha.data.fill_(20.0); L.beta.data.fill_(-20.0)   # fully strict, critical dependencies
    o = torch.ones(1, 4); o[0, 1] = 0.0                  # a shared critical parent is down
    assert (L(o) < 0.2).all()
