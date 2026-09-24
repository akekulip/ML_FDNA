"""Inference-only modules: partial observations -> log-distribution over the 1024 control vectors."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .. import comm, v2
from .belief import BeliefFDNA, MinMaxGNN, N_F, N_U
from .layers import MLP

LEVELS = np.array([0.0, 0.4, 0.6, 1.0])


def level_index(CV: np.ndarray) -> np.ndarray:
    """(n_cv, 5) commandable fractions -> (n_cv, 5) level index in {0,1,2,3}."""
    idx = np.abs(CV[:, :, None] - LEVELS[None, None, :]).argmin(2)
    assert np.allclose(LEVELS[idx], CV, atol=1e-4)
    return idx


class MeanFieldControls(nn.Module):
    """Per-generator level distribution from unit operabilities with independent unit Bernoullis:
    units (2g, 2g+1) have shares (0.6, 0.4) -> P(c_g) over {0, .4, .6, 1}."""

    def __init__(self, level_idx: np.ndarray):
        super().__init__()
        self.register_buffer("lv", torch.tensor(level_idx, dtype=torch.long))  # (1024, 5)

    def forward(self, o_unit: torch.Tensor) -> torch.Tensor:
        o = o_unit.clamp(1e-4, 1 - 1e-4).view(-1, comm.N_REMOTE_GEN, comm.UNITS_PER_GEN)
        a, b = o[:, :, 0], o[:, :, 1]            # a: share 0.6, b: share 0.4
        p = torch.stack([(1 - a) * (1 - b), (1 - a) * b, a * (1 - b), a * b], dim=2)   # (B,5,4) levels 0,.4,.6,1
        lp = torch.log(p)
        return sum(lp[:, g, :][:, self.lv[:, g]] for g in range(comm.N_REMOTE_GEN))


class InfMLP(nn.Module):
    def __init__(self, n_cv: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_F * 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_cv))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.net(e.reshape(len(e), -1)), dim=1)


class InfGNN(nn.Module):
    def __init__(self, n_cv: int):
        super().__init__()
        self.g = MinMaxGNN(1)
        self.head = nn.Linear(N_F * 16, n_cv)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.head(self.g.states(e)), dim=1)


class InfFDNA(nn.Module):
    """variant: 'fdna' | 'shuffled' | 'unconstrained' (same masks as BeliefFDNA)."""

    def __init__(self, level_idx: np.ndarray, variant: str = "fdna", seed: int = 0):
        super().__init__()
        self.b = BeliefFDNA(1, variant, seed=seed)
        self.mf = MeanFieldControls(level_idx)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        b = self.b
        logit = b.prior[None] + (e[:, :, 0:1] * (b.mf * b.w1)[None]).sum(1) + (e[:, :, 1:2] * (b.mf * b.w0)[None]).sum(1)
        o_unit = b.fdna(torch.sigmoid(logit))
        return torch.log_softmax(self.mf(o_unit), dim=1)


def fit_ce(model, E_tr, y_tr, E_va, y_va, epochs=40, lr=2e-3, wd=0.0, bs=2048, patience=6, seed=0, dev="cuda"):
    import copy
    torch.manual_seed(seed)
    model.to(dev)
    Xtr = torch.as_tensor(E_tr, dtype=torch.float32, device=dev); ytr = torch.as_tensor(y_tr, dtype=torch.long, device=dev)
    Xva = torch.as_tensor(E_va, dtype=torch.float32, device=dev); yva = torch.as_tensor(y_va, dtype=torch.long, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, state, bad = 1e30, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), device=dev)
        for a in range(0, len(ytr), bs):
            idx = perm[a:a + bs]
            loss = torch.nn.functional.nll_loss(model(Xtr[idx]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = sum(torch.nn.functional.nll_loss(model(Xva[a:a + 8192]), yva[a:a + 8192], reduction="sum").item() for a in range(0, len(yva), 8192)) / len(yva)
        if v < best - 1e-6:
            best, state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return model, best


def predict_logp(model, E, dev="cuda", bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for a in range(0, len(E), bs):
            out.append(model(torch.as_tensor(E[a:a + bs], dtype=torch.float32, device=dev)).cpu().numpy())
    return np.concatenate(out)
