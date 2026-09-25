"""Phase 4: linear baseline, v2 -- fixes issues an external review found in v1 (kept, not deleted):
  1. KFold split by ROW (op,triple pairs), not by TRIPLE -- a triple's rows from other ops could leak into
     training. Fixed: GroupKFold grouped by triple id, so an entire triple's rows are held out together,
     genuinely testing generalisation to unseen triples.
  2. Only a squared-error-trained model was fit, but only MAE reported -- 'loses to g2 on MAE' does not
     demonstrate overfitting from a model trained on squared error. Fixed: also fit an MAE-trained
     QuantileRegressor(quantile=0.5), report held-out MAE AND MSE for both.
  3. The claim 'coefficients land at 0.92-1.02 for every ingredient' was false at cv_idx=0 (one coefficient,
     I_bc, was 0.6432). Not repeated here; per-coefficient values reported plainly, in and out of range noted.
"""
import json
import numpy as np
from sklearn.linear_model import LinearRegression, QuantileRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from fdna import spec
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]
outage_to_group = {o: i for i, o in enumerate(sorted(set(OUTAGES)))}  # group id = distinct triple, stable across cv_idx
needed_singles = sorted({b for o in OUTAGES for b in o})
needed_pairs = sorted({(min(o[i], o[j]), max(o[i], o[j])) for o in OUTAGES for i, j in [(0, 1), (0, 2), (1, 2)]})

CV_IDXS = [int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))]
results = {}
for cv_idx in CV_IDXS:
    y_single, y_pair = {}, {}
    for op_id in OP_IDS:
        op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
        for b in needed_singles: y_single[(op_id, b)] = ScenarioLP(G, op, (b,), spec.PARAMS).y(CV[cv_idx])
        for a, b in needed_pairs: y_pair[(op_id, a, b)] = ScenarioLP(G, op, (a, b), spec.PARAMS).y(CV[cv_idx])
    feats, g2s, ys, groups = [], [], [], []
    for op_id, outage, y_true in zip(k3["op_ids"], [tuple(int(x) for x in o) for o in k3["outages"]], k3["V"][:, cv_idx]):
        op_id = int(op_id); a, b, c = outage
        ya, yb, yc = y_single[(op_id, a)], y_single[(op_id, b)], y_single[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y_pair[(op_id, p, q)] - y_single[(op_id, p)] - y_single[(op_id, q)] for p, q in pairs]
        feats.append([ya, yb, yc, *Ivals]); g2s.append(ya + yb + yc + sum(Ivals)); ys.append(float(y_true)); groups.append(outage_to_group[outage])
    X, g2, y, groups = np.array(feats), np.array(g2s), np.array(ys), np.array(groups)
    gkf = GroupKFold(5)
    pred_ols, pred_l1 = np.zeros_like(y), np.zeros_like(y)
    for tr_i, te_i in gkf.split(X, y, groups):
        m1 = LinearRegression(); m1.fit(X[tr_i], y[tr_i]); pred_ols[te_i] = m1.predict(X[te_i])
        m2 = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs"); m2.fit(X[tr_i], y[tr_i]); pred_l1[te_i] = m2.predict(X[te_i])
    mae = lambda p: float(np.abs(y - p).mean()); mse = lambda p: float(((y - p) ** 2).mean())
    rho = lambda p: float(spearmanr(p, y)[0]) if p.std() > 0 else None
    m_ols_full = LinearRegression().fit(X, y); m_l1_full = QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs").fit(X, y)
    results[cv_idx] = {
        "n_groups_triples": len(set(groups)),
        "g2_fixed_coef1": {"mae": mae(g2), "mse": mse(g2), "rho": rho(g2)},
        "linear_ols_groupcv": {"mae": mae(pred_ols), "mse": mse(pred_ols)},
        "linear_l1_mae_trained_groupcv": {"mae": mae(pred_l1), "mse": mse(pred_l1)},
        "fitted_coefficients_ols_full_data": m_ols_full.coef_.round(4).tolist(),
        "fitted_coefficients_l1_full_data": m_l1_full.coef_.round(4).tolist(),
        "g2_beats_ols": bool(mae(g2) < mae(pred_ols)), "g2_beats_l1": bool(mae(g2) < mae(pred_l1)),
    }
    print(cv_idx, json.dumps(results[cv_idx], indent=1))
json.dump(results, open("results/phase4/linear_baseline_v2.json", "w"), indent=1)
