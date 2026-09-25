"""Phase 5 Stage 4: matched recurrent/learned-correction experiment (registry/phase5_matched_recurrent.yaml).
Uses ONLY cached n1/n2/n3_confirm.npz tables (0 new LP solves). All trained arms get IDENTICAL features and are
composed with the SAME P1/P2 posterior on the SAME R-precision endpoint, on 25 held-out test triples never
touched during training. Identifiability check (registry doc): the residual-correction arm trains ONLY on the
N-3 rows (never N-1/N-2, where the residual is trivially zero)."""
import json
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import Ridge

from fdna import spec, v2, v2data
from fdna.adaptive_query import bounds as _unused  # not used here; import kept minimal
from fdna.dataset import G, RATING
from fdna.evalutil import rprec
from fdna.lodf import ptdf_lodf, compensation_det
from fdna.opgen import sample_op
from fdna.physical_correction import island_floor
from fdna.rules import boot

torch.manual_seed(0)
n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]; ALL_OUTAGES = sorted(set(tuple(o) for o in manifest["outage_sets"]))

# frozen 45/25 split, fixed seed, before fitting
split_rng = np.random.default_rng(4004)
perm = split_rng.permutation(len(ALL_OUTAGES))
TRAIN_OUTAGES = {ALL_OUTAGES[i] for i in perm[:45]}
TEST_OUTAGES = {ALL_OUTAGES[i] for i in perm[45:]}
assert TRAIN_OUTAGES.isdisjoint(TEST_OUTAGES) and len(TRAIN_OUTAGES) == 45 and len(TEST_OUTAGES) == 25
print(f"train triples: {len(TRAIN_OUTAGES)}, test triples: {len(TEST_OUTAGES)}, disjoint: {TRAIN_OUTAGES.isdisjoint(TEST_OUTAGES)}")

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]

demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in OP_IDS}
_, LODF = ptdf_lodf(G)


def physical_feats(op_id, outage):
    a, b, c = outage
    F = island_floor(G, demands[op_id], outage)
    pairs = [(a, b), (a, c), (b, c)]
    risks = [1.0 / max(abs(compensation_det(LODF, p[0], p[1])), 1e-9) for p in pairs]
    risks = [min(r, 1e4) for r in risks]  # cap for numeric stability
    return F, risks


PHYS = {(op, o): physical_feats(op, o) for op in OP_IDS for o in ALL_OUTAGES}


def build_features(op_id, outage, cv_idx_arr):
    """Vectorised over a batch of control indices for ONE (op,outage)."""
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)][cv_idx_arr], y1[(op_id, b)][cv_idx_arr], y1[(op_id, c)][cv_idx_arr]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iab, Iac, Ibc = (y2[(op_id, p)][cv_idx_arr] - y1[(op_id, p[0])][cv_idx_arr] - y1[(op_id, p[1])][cv_idx_arr] for p in pairs)
    F, risks = PHYS[(op_id, outage)]
    n = len(cv_idx_arr)
    phys_rep = np.tile([F, *risks], (n, 1))
    cvec = CV[cv_idx_arr]
    X = np.column_stack([ya, yb, yc, Iab, Iac, Ibc, phys_rep, cvec]).astype(np.float32)
    g2 = (ya + yb + yc + Iab + Iac + Ibc).astype(np.float32)
    return X, g2


# ---- build training data: 45 train triples x 60 ops x 128 sampled controls ----
train_rng = np.random.default_rng(5005)
Xtr_list, ytr_list, g2tr_list = [], [], []
outage_row = {o: i for i, o in zip(range(len(outages_sorted)), outages_sorted)}
row_by_outage_op = {(int(op_ids_sorted[i]), outages_sorted[i]): i for i in range(len(outages_sorted))}
for outage in TRAIN_OUTAGES:
    for op_id in OP_IDS:
        i = row_by_outage_op[(op_id, outage)]
        cv_idx = train_rng.choice(len(CV), 128, replace=False)
        X, g2 = build_features(op_id, outage, cv_idx)
        y = V_sorted[i][cv_idx].astype(np.float32)
        Xtr_list.append(X); ytr_list.append(y); g2tr_list.append(g2)
Xtr = np.concatenate(Xtr_list); ytr = np.concatenate(ytr_list); g2tr = np.concatenate(g2tr_list)
print(f"train rows: {len(Xtr)}, target std: {ytr.std():.4f}")

val_rng = np.random.default_rng(6006)
val_outages = list(TRAIN_OUTAGES)[:10]
Xva_list, yva_list = [], []
for outage in val_outages:
    for op_id in OP_IDS[:15]:
        i = row_by_outage_op[(op_id, outage)]
        cv_idx = val_rng.choice(len(CV), 32, replace=False)
        X, _ = build_features(op_id, outage, cv_idx)
        Xva_list.append(X); yva_list.append(V_sorted[i][cv_idx].astype(np.float32))
Xva, yva = np.concatenate(Xva_list), np.concatenate(yva_list)
print(f"val rows: {len(Xva)}, target std: {yva.std():.4f}")

# ---- fit ridge ----
ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
print("ridge val MSE:", float(((ridge.predict(Xva) - yva) ** 2).mean()))

# ---- fit GBM (tuned, small grid, screen-level) ----
best = None
rng0 = np.random.default_rng(1)
grid = [dict(num_leaves=int(a), min_child_samples=int(b)) for a, b in
        zip(rng0.choice([15, 31, 63], 8), rng0.choice([20, 50, 100], 8))]
for p in grid:
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, verbose=-1, random_state=0, n_jobs=8, **p)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(20, verbose=False)])
    e = float(((m.predict(Xva) - yva) ** 2).mean())
    if best is None or e < best[0]: best = (e, p, m)
gbm = best[2]
print("gbm val MSE:", best[0])

# ---- fit DeepSets (pair-aware, commutative) ----
class DeepSetsPairAware(nn.Module):
    def __init__(self, d_in=15, h=64):
        super().__init__()
        self.elem_enc = nn.Sequential(nn.Linear(2, h), nn.ReLU(), nn.Linear(h, h))   # (singleton, one interaction) per element-slot, pooled
        self.readout = nn.Sequential(nn.Linear(h + 4 + 5, h), nn.ReLU(), nn.Linear(h, 1))  # + phys(4) + c(5)

    def forward(self, X):
        # X columns: [ya,yb,yc, Iab,Iac,Ibc, F,rab,rac,rbc, c1..c5]
        elems = torch.stack([
            torch.stack([X[:, 0], X[:, 3]], 1),   # (a, ab-interaction)
            torch.stack([X[:, 1], X[:, 4]], 1),   # (b, ac-interaction)  -- pairing is a convention, pool removes order sensitivity
            torch.stack([X[:, 2], X[:, 5]], 1),   # (c, bc-interaction)
        ], 1)  # (B,3,2)
        e = self.elem_enc(elems).sum(1)  # commutative sum-pool over the 3 element-slots
        rest = X[:, 6:]
        return self.readout(torch.cat([e, rest], 1)).squeeze(-1)


def train_torch(model, Xtr, ytr, Xva, yva, epochs=80, lr=2e-3, dev="cuda"):
    model.to(dev)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(yva, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    best_val, best_state, bad = 1e30, None, 0
    losses = []
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(Xt), device=dev)
        for a in range(0, len(idx), 2048):
            b = idx[a:a + 2048]
            opt.zero_grad(); pred = model(Xt[b]); loss = ((pred - yt[b]) ** 2).mean(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vloss = ((model(Xv) - yv) ** 2).mean().item()
        losses.append(vloss)
        if vloss < best_val - 1e-8:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 12: break
    model.load_state_dict(best_state)
    return model, best_val, losses


ds_model, ds_val, ds_losses = train_torch(DeepSetsPairAware(), Xtr, ytr, Xva, yva)
print("deepsets val MSE:", ds_val, "loss decreased:", ds_losses[0] > ds_losses[-1], "n_epochs:", len(ds_losses))

# ---- fit residual-correction network: target = y - g2 (ONLY on N-3 train rows, per the identifiability check) ----
class ResidualNet(nn.Module):
    def __init__(self, d_in=15, h=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X):
        return self.net(X).squeeze(-1)


rtr, rva = (ytr - g2tr).astype(np.float32), None
Xva_g2 = None
Xva_list2, g2va_list = [], []
for outage in val_outages:
    for op_id in OP_IDS[:15]:
        i = row_by_outage_op[(op_id, outage)]
        cv_idx = val_rng.choice(len(CV), 32, replace=False)
        _, g2v = build_features(op_id, outage, cv_idx)
        g2va_list.append(g2v)
g2va = np.concatenate(g2va_list)
rva = (yva - g2va).astype(np.float32)
res_model, res_val_mse_of_residual, res_losses = train_torch(ResidualNet(), Xtr, rtr, Xva, rva)
print("residual-net val MSE (of residual target):", res_val_mse_of_residual, "loss decreased:", res_losses[0] > res_losses[-1])

# ---- pre-rejection checklist ----
checklist = {
    "target_variation_train_std": float(ytr.std()), "target_variation_val_std": float(yva.std()),
    "ridge_beats_predict_mean_baseline": bool(((ridge.predict(Xva) - yva) ** 2).mean() < yva.var()),
    "deepsets_loss_decreased": bool(ds_losses[0] > ds_losses[-1]),
    "residual_loss_decreased": bool(res_losses[0] > res_losses[-1]),
    "split_disjoint_verified": bool(TRAIN_OUTAGES.isdisjoint(TEST_OUTAGES)),
}
print("PRE-REJECTION CHECKLIST", json.dumps(checklist, indent=1))
json.dump({"split": {"train": [list(o) for o in TRAIN_OUTAGES], "test": [list(o) for o in TEST_OUTAGES]},
           "checklist": checklist, "val_mse": {"ridge": float(((ridge.predict(Xva) - yva) ** 2).mean()),
           "gbm": best[0], "deepsets": ds_val, "residual_net_on_residual_target": res_val_mse_of_residual}},
          open("results/phase5/matched_recurrent_training.json", "w"), indent=1)
print("TRAINING PHASE DONE", flush=True)

import pickle
with open("data_hik/matched_recurrent_models.pkl", "wb") as f:
    pickle.dump({"ridge": ridge, "gbm": gbm}, f)
torch.save(ds_model.state_dict(), "data_hik/deepsets_model.pt")
torch.save(res_model.state_dict(), "data_hik/residual_model.pt")
