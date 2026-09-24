"""Phase 4 Stage 3: RICHER equal-information baseline for the Mobius-truncation test (brief 6.3's staged
comparator set wants a tuned tree on FULL descriptors, not just the 6 Mobius numbers). Adds, per triple: the
three sub-pair LODF risk scores (hik_diag.generalized_det), the k=3 minimal-cut/struct_mw floor and island count
(hik_diag.minimal_cut_struct), and whether the triple is a genuinely new k=3 cut (hik_diag.is_new_cut) -- all free,
exact, LP-solve-free, already correctness-gated. This is the SAME information g2 uses (y_single, y_pair) PLUS the
structural descriptors the island-decomposition design already established are informative, fed to a generic
learner with no imposed additive form -- the fairest baseline g2 must beat."""
import json
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from fdna import spec, hik_diag
from fdna.dataset import G, RATING
from fdna.lodf import ptdf_lodf
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

k3 = np.load("data_hik/k3_screen.npz"); CV = k3["CV"]
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]
needed_singles = sorted({b for o in OUTAGES for b in o})
needed_pairs = sorted({(min(o[i], o[j]), max(o[i], o[j])) for o in OUTAGES for i, j in [(0, 1), (0, 2), (1, 2)]})
_, LODF = ptdf_lodf(G)

CV_IDXS = [int(np.argmax(CV.sum(1))), int(np.argmin(CV.sum(1)))]
results = {}
for cv_idx in CV_IDXS:
    y_single, y_pair, struct_op = {}, {}, {}
    for op_id in OP_IDS:
        op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
        struct_op[op_id] = op.demand
        for b in needed_singles:
            y_single[(op_id, b)] = ScenarioLP(G, op, (b,), spec.PARAMS).y(CV[cv_idx])
        for a, b in needed_pairs:
            y_pair[(op_id, a, b)] = ScenarioLP(G, op, (a, b), spec.PARAMS).y(CV[cv_idx])

    feats_rich, feats_plain, g1s, g2s, ys = [], [], [], [], []
    for op_id, outage, y_true in zip(k3["op_ids"], [tuple(int(x) for x in o) for o in k3["outages"]], k3["V"][:, cv_idx]):
        op_id = int(op_id)
        a, b, c = outage
        ya, yb, yc = y_single[(op_id, a)], y_single[(op_id, b)], y_single[(op_id, c)]
        pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
        Ivals = [y_pair[(op_id, p, q)] - y_single[(op_id, p)] - y_single[(op_id, q)] for p, q in pairs]
        g1, g2 = ya + yb + yc, ya + yb + yc + sum(Ivals)
        risks = [1.0 / max(abs(hik_diag.generalized_det(LODF, (p, q))), 1e-9) for p, q in pairs]
        struct_mw, n_isl = hik_diag.minimal_cut_struct(G, outage, demand=struct_op[op_id])
        new_cut = float(hik_diag.is_new_cut(G, outage))
        feats_plain.append([ya, yb, yc, *Ivals])
        feats_rich.append([ya, yb, yc, *Ivals, *risks, struct_mw / struct_op[op_id].sum(), n_isl, new_cut])
        g1s.append(g1); g2s.append(g2); ys.append(float(y_true))

    Xp, Xr, g1, g2, y = np.array(feats_plain), np.array(feats_rich), np.array(g1s), np.array(g2s), np.array(ys)
    if y.std() < 1e-9:
        results[cv_idx] = {"note": "degenerate, skipped"}; continue
    kf = KFold(5, shuffle=True, random_state=0)
    pred_plain, pred_rich = np.zeros_like(y), np.zeros_like(y)
    for tr_i, te_i in kf.split(Xp):
        m1 = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=20, verbose=-1, random_state=0)
        m1.fit(Xp[tr_i], y[tr_i]); pred_plain[te_i] = m1.predict(Xp[te_i])
        m2 = lgb.LGBMRegressor(n_estimators=200, num_leaves=15, min_child_samples=20, verbose=-1, random_state=0)
        m2.fit(Xr[tr_i], y[tr_i]); pred_rich[te_i] = m2.predict(Xr[te_i])
    mae = lambda p: float(np.abs(y - p).mean()); rho = lambda p: float(spearmanr(p, y)[0]) if p.std() > 0 else None
    results[cv_idx] = {
        "n": len(y),
        "g2_mobius": {"mae": mae(g2), "rho": rho(g2)},
        "gbm_6feat_plain": {"mae": mae(pred_plain), "rho": rho(pred_plain)},
        "gbm_rich_islanddescriptors": {"mae": mae(pred_rich), "rho": rho(pred_rich)},
        "g2_beats_rich_gbm_mae": bool(mae(g2) < mae(pred_rich)),
    }
    print(cv_idx, results[cv_idx])
json.dump(results, open("results/phase4/mobius_k3_richbaseline.json", "w"), indent=1)
