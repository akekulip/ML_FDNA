"""Phase 4 correction: a plain LINEAR regression on the identical 6 ingredients g2 uses
([y_a,y_b,y_c,I_ab,I_ac,I_bc]), 5-fold CV, exact-control screen (same setup as p4_mobius_k3_test.py /
p4_mobius_k3_richbaseline.py). g2's coefficients are all fixed at 1 (a pure sum); this tests whether FITTING
those coefficients from data helps or hurts, and gives GBM's earlier underperformance a simpler point of
comparison (external review: 'include a simple linear composition baseline too')."""
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from fdna import spec
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]
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
    feats, g2s, ys = [], [], []
    for op_id, outage, y_true in zip(k3["op_ids"], [tuple(int(x) for x in o) for o in k3["outages"]], k3["V"][:, cv_idx]):
        op_id = int(op_id); a, b, c = outage
        ya, yb, yc = y_single[(op_id, a)], y_single[(op_id, b)], y_single[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y_pair[(op_id, p, q)] - y_single[(op_id, p)] - y_single[(op_id, q)] for p, q in pairs]
        feats.append([ya, yb, yc, *Ivals]); g2s.append(ya + yb + yc + sum(Ivals)); ys.append(float(y_true))
    X, g2, y = np.array(feats), np.array(g2s), np.array(ys)
    kf = KFold(5, shuffle=True, random_state=0); pred_lin = np.zeros_like(y)
    for tr_i, te_i in kf.split(X):
        m = LinearRegression(); m.fit(X[tr_i], y[tr_i]); pred_lin[te_i] = m.predict(X[te_i])
    mae = lambda p: float(np.abs(y - p).mean()); rho = lambda p: float(spearmanr(p, y)[0]) if p.std() > 0 else None
    m_full = LinearRegression().fit(X, y)
    results[cv_idx] = {"g2_fixed_coef1": {"mae": mae(g2), "rho": rho(g2)},
                        "linear_fitted_5fold": {"mae": mae(pred_lin), "rho": rho(pred_lin)},
                        "fitted_coefficients": m_full.coef_.round(4).tolist(), "fitted_intercept": round(float(m_full.intercept_), 5),
                        "g2_beats_linear": bool(mae(g2) < mae(pred_lin))}
    print(cv_idx, json.dumps(results[cv_idx], indent=1))
json.dump(results, open("results/phase4/linear_baseline.json", "w"), indent=1)
