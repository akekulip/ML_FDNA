"""Small full-batch-ish trainer for the Phase-2 torch models (MSE, Adam, early stopping on validation)."""
from __future__ import annotations

import copy

import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def to_t(*arrs):
    return [torch.as_tensor(np.asarray(a), dtype=torch.float32, device=DEV) for a in arrs]


Y_SCALE = 100.0  # targets are shed fractions ~1e-2; unscaled MSE starves gradients (Phase-2 diagnosis, scripts/nn_diagnose.py)


def fit(model, inputs_tr, y_tr, inputs_va, y_va, epochs=60, lr=2e-3, bs=2048, patience=8, seed=0, wd=0.0):
    torch.manual_seed(seed)
    model.to(DEV)
    model.y_scale = Y_SCALE
    y_tr, y_va = np.asarray(y_tr) * Y_SCALE, np.asarray(y_va) * Y_SCALE
    Xtr, Xva = to_t(*inputs_tr), to_t(*inputs_va)
    ytr, yva = to_t(y_tr)[0], to_t(y_va)[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best, best_state, bad = 1e30, None, 0
    n = len(ytr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEV)
        for a in range(0, n, bs):
            idx = perm[a:a + bs]
            loss = torch.nn.functional.mse_loss(model(*[x[idx] for x in Xtr]), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            v = float(torch.nn.functional.mse_loss(model(*Xva), yva))
        if v < best - 1e-9:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best


def predict(model, inputs, bs=8192):
    model.eval()
    X = to_t(*inputs)
    out = []
    with torch.no_grad():
        for a in range(0, len(X[0]), bs):
            out.append(model(*[x[a:a + bs] for x in X]).cpu().numpy())
    return np.concatenate(out) / getattr(model, "y_scale", 1.0)
