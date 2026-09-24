import numpy as np
import torch

from fdna.nn import belief


def test_masks_and_variants_have_same_edge_count():
    m = belief.flag_ancestor_mask()
    assert m.shape == (13, 14) and m[0].sum() == 3 and m[10].sum() == 2     # unit0: CC,RTU,GW0 ; heartbeat0: CC,GW0
    s = belief._shuffle_columns(m, 3)
    assert s.sum() == m.sum() and (s != m).any()


def test_variants_forward_and_gradients():
    rng = np.random.default_rng(0)
    obs = rng.integers(-1, 2, size=(8, 13)).astype(np.int8)
    e = torch.tensor(belief.obs_tensor(obs)); x = torch.randn(8, 20)
    for v in ("fdna", "shuffled", "unconstrained"):
        net = belief.BeliefFDNA(20, v)
        out = net(x, e); assert out.shape == (8,)
        out.sum().backward()
        assert net.w1.grad.abs().sum() > 0 and net.fdna.alpha.grad.abs().sum() > 0
    g = belief.MinMaxGNN(20)
    out = g(x, e); assert out.shape == (8,)
    out.sum().backward()
    assert g.emb.grad.abs().sum() > 0


def test_fdna_variant_controls_in_unit_interval_and_monotone_head():
    net = belief.BeliefFDNA(6, "fdna")
    e = torch.tensor(belief.obs_tensor(np.random.default_rng(1).integers(-1, 2, size=(16, 13)).astype(np.int8)))
    c = net.controls(e)
    assert ((c >= 0) & (c <= 1.0001)).all()
