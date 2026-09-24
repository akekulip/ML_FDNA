"""Diagnosis of why the Branch-3 neural arms lag trees by ~0.3 R-precision. Uses TRAIN/VALIDATION operating points only."""
import time
import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer

from fdna import baseline_data
from fdna.evalutil import op_metrics
from fdna.nn.layers import MLP, MonotoneValue
from fdna.nn.train import fit, predict, DEV
import fdna.nn.train as T

d = baseline_data.load("data")
X = d.F["elec+phys+ctrl"]; NX = 123; C = X[:, 123:128]
mu, sd = X[d.tr].mean(0), X[d.tr].std(0) + 1e-6
Zs = ((X - mu) / sd).astype(np.float32)
qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200_000, random_state=0).fit(X[d.tr])
Zq = qt.transform(X).astype(np.float32)
val_n2 = d.va & (d.lab.cell.values == "n2_unseen")
ops_tr = np.unique(d.lab.op.values[d.tr])


def val_rprec(pred):
    vals = []
    for o in np.unique(d.lab.op.values[val_n2]):
        m = val_n2 & (d.lab.op.values == o)
        vals.append(op_metrics(d.y[m], pred[m], d.key[m])["rprec"])
    return float(np.nanmean(vals))


def run(name, model_fn, Z, transform, n, epochs=60, lr=2e-3, wd=0.0, bs=2048, mono=False, huber=False):
    sub = ops_tr if n == 100 else np.random.default_rng(1000 * n).choice(ops_tr, n, replace=False)
    mtr = d.tr & np.isin(d.lab.op.values, sub)
    ytr, yva = transform(d.y[mtr]), transform(d.y[d.va])
    model = model_fn()
    inp = lambda m: (Z[m][:, :NX], C[m]) if mono else (Z[m],)
    t0 = time.time()
    model, best = fit(model, inp(mtr), ytr, inp(d.va), yva, epochs=epochs, lr=lr, bs=bs, wd=wd, patience=12)
    pred = np.full(len(d.y), np.nan); pred[val_n2] = predict(model, inp(val_n2))
    print(f"{name:34s} n={n:3d} val-MSE={best:.5f} val-N2 R-prec={val_rprec(pred):.3f} ({time.time()-t0:.0f}s)", flush=True)


ident, x100, l1p = (lambda y: y), (lambda y: y * 100), (lambda y: np.log1p(100 * y))
RUN_OLD = False
if RUN_OLD:
    for n in (25, 100):
        run("R0 MLP, raw y (as in Branch 3)", lambda: MLP(X.shape[1]), Zs, ident, n)
        run("R1 MLP, y*100", lambda: MLP(X.shape[1]), Zs, x100, n)
        run("R2 MLP, log1p(100y)", lambda: MLP(X.shape[1]), Zs, l1p, n)
        run("R3 R2 + quantile inputs", lambda: MLP(X.shape[1]), Zq, l1p, n)
        run("R4 R3 + wide 256x3, 150 ep, wd", lambda: MLP(X.shape[1], 256, 3), Zq, l1p, n, epochs=150, wd=1e-5)
        run("M1 monotone, raw y", lambda: MonotoneValue(NX, 5), Zs, ident, n, mono=True)
        run("M2 monotone, log1p(100y) + quantile", lambda: MonotoneValue(NX, 5), Zq, l1p, n, mono=True)
        run("M3 monotone big, 150 ep", lambda: MonotoneValue(NX, 5, groups=16, planes=16, hidden=256), Zq, l1p, n, mono=True, epochs=150, wd=1e-5)
    

print("---- monotone head with corrected target scaling (validation only) ----")
for n in (25, 100):
    run("M4 monotone, y*100", lambda: MonotoneValue(NX, 5), Zs, x100, n, mono=True)
    run("M5 monotone big, y*100, 100 ep", lambda: MonotoneValue(NX, 5, groups=16, planes=16, hidden=256), Zs, x100, n, mono=True, epochs=100)
    run("M6 monotone, y*100, lr 5e-3", lambda: MonotoneValue(NX, 5), Zs, x100, n, mono=True, lr=5e-3)
    run("R5 MLP wide, y*100, 100 ep", lambda: MLP(X.shape[1], 256, 3), Zs, x100, n, epochs=100)
