import numpy as np
import torch

from fdna import v2
from fdna.nn import belief, inference


def test_level_index_and_mean_field_is_a_distribution():
    w = v2.build_world()
    lv = inference.level_index(w.CV)
    assert lv.shape == (1024, 5) and len({tuple(r) for r in lv}) == 1024
    mf = inference.MeanFieldControls(lv)
    o = torch.rand(6, 10)
    lp = mf(o)
    np.testing.assert_allclose(torch.logsumexp(lp, dim=1).detach().numpy(), 0.0, atol=1e-4)   # normalised over all 1024 vectors


def test_mean_field_matches_exact_when_units_certain():
    w = v2.build_world()
    mf = inference.MeanFieldControls(inference.level_index(w.CV))
    o = torch.ones(1, 10); o[0, 2] = 0.0                     # unit 2 (share 0.6 of generator 1) down, all else up
    p = torch.exp(mf(o))[0].numpy()
    k = int(np.argmax(p))
    np.testing.assert_allclose(w.CV[k], [1.0, 0.4, 1.0, 1.0, 1.0], atol=1e-6)
    assert p[k] > 0.99


def test_inference_models_forward_and_backward():
    w = v2.build_world()
    lv = inference.level_index(w.CV)
    obs = np.random.default_rng(0).integers(-1, 2, size=(32, 13)).astype(np.int8)
    E = torch.tensor(belief.obs_tensor(obs))
    for m in (inference.InfMLP(1024), inference.InfGNN(1024), inference.InfFDNA(lv, "fdna"), inference.InfFDNA(lv, "shuffled"), inference.InfFDNA(lv, "unconstrained")):
        lp = m(E)
        assert lp.shape == (32, 1024)
        np.testing.assert_allclose(torch.logsumexp(lp, dim=1).detach().numpy(), 0.0, atol=1e-3)
        lp[:, 3].sum().backward()
