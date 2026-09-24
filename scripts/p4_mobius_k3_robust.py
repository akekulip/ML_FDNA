"""Phase 4 Stage 3: robustness of the Mobius-truncation N-3 result across control states, plus the REQUIRED
equal-information baseline (GBM given the same y_single/y_pair numbers, no imposed additive structure, 5-fold CV
to avoid overfitting-inflated comparison on 1200 rows)."""
import json
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from fdna import spec
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]
needed_singles = sorted({b for o in OUTAGES for b in o})
needed_pairs = sorted({(min(o[i], o[j]), max(o[i], o[j])) for o in OUTAGES for i, j in [(0, 1), (0, 2), (1, 2)]})

CV_IDXS = [int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1))), 512, 100]  # full, none, mid, another
results = {}
for cv_idx in CV_IDXS:
    y_single, y_pair = {}, {}
    for op_id in OP_IDS:
        op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
        for b in needed_singles:
            y_single[(op_id, b)] = ScenarioLP(G, op, (b,), spec.PARAMS).y(CV[cv_idx])
        for a, b in needed_pairs:
            y_pair[(op_id, a, b)] = ScenarioLP(G, op, (a, b), spec.PARAMS).y(CV[cv_idx])
    feats, g1s, g2s, ys = [], [], [], []
    for op_id, outage, y_true in zip(k3["op_ids"], [tuple(int(x) for x in o) for o in k3["outages"]], k3["V"][:, cv_idx]):
        op_id = int(op_id)
        ya, yb, yc = (y_single[(op_id, b)] for b in outage)
        subpairs = [(min(outage[0], outage[1]), max(outage[0], outage[1])), (min(outage[0], outage[2]), max(outage[0], outage[2])), (min(outage[1], outage[2]), max(outage[1], outage[2]))]
        Iab, Iac, Ibc = (y_pair[(op_id, a, b)] - y_single[(op_id, a)] - y_single[(op_id, b)] for a, b in subpairs)
        g1, g2 = ya + yb + yc, ya + yb + yc + Iab + Iac + Ibc
        feats.append([ya, yb, yc, Iab, Iac, Ibc]); g1s.append(g1); g2s.append(g2); ys.append(float(y_true))
    X, g1, g2, y = np.array(feats), np.array(g1s), np.array(g2s), np.array(ys)
    if y.std() < 1e-9:
        results[cv_idx] = {"note": "degenerate (all-zero labels at this control), skipped"}
        continue
    kf = KFold(5, shuffle=True, random_state=0); pred_gbm = np.zeros_like(y)
    for tr_i, te_i in kf.split(X):
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=20, verbose=-1, random_state=0)
        m.fit(X[tr_i], y[tr_i]); pred_gbm[te_i] = m.predict(X[te_i])
    mae = lambda p: float(np.abs(y - p).mean()); rmse = lambda p: float(np.sqrt(((y - p) ** 2).mean()))
    rho = lambda p: float(spearmanr(p, y)[0]) if p.std() > 0 else None
    results[cv_idx] = {"cv_sum": float(CV[cv_idx].sum()), "n": len(y), "mean_y": float(y.mean()),
                        "g1": {"mae": mae(g1), "rmse": rmse(g1), "rho": rho(g1)},
                        "g2_mobius": {"mae": mae(g2), "rmse": rmse(g2), "rho": rho(g2)},
                        "gbm_equal_info_5fold": {"mae": mae(pred_gbm), "rmse": rmse(pred_gbm), "rho": rho(pred_gbm)},
                        "g2_beats_gbm_mae": bool(mae(g2) < mae(pred_gbm))}
    print(cv_idx, results[cv_idx])
json.dump(results, open("results/phase4/mobius_k3_robust.json", "w"), indent=1)
