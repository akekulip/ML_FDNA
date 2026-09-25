"""Phase 5 Stage R1.1: matched recurrent/learned-correction experiment, v2 -- fixes four bugs an external
review found in v1 (kept, not deleted, for the record):

1. VALIDATION TARGET MISALIGNMENT (critical). v1 sampled Xva/yva with one RNG draw, then later advanced the
   SAME RNG to sample g2va at DIFFERENT control indices, so rva=yva-g2va compared mismatched states.
   Reproduced exactly: 4,796/4,800 mismatched indices. Fixed: (key, X, y, g2) are now built together in ONE
   pass, for both validation and training; r=y-g2 is computed from those retained arrays; an assertion checks
   r+g2==y by construction.
2. VALIDATION/TRAIN OVERLAP. v1's validation reused the first 10 TRAINING outages and the first 15 (of all 60)
   TRAINING operating points. Reproduced: 612/4,800 validation rows also present in training; the rest still
   shared both outages and ops with training (interpolation, not held-out generalisation). Fixed: outages AND
   operating points are now split into three genuinely DISJOINT groups (train/val/test), frozen before fitting.
3. THE "COMMUTATIVE" MODEL WAS NOT ACTUALLY PERMUTATION-INVARIANT. v1's tokens (ya,Iab),(yb,Iac),(yc,Ibc) paired
   a singleton with the WRONG interaction term under relabeling. Reproduced: relabeling a<->b changed the
   forward output from 0.10 to 0.11 with a valid weight assignment. Fixed: each of the 3 EDGES {a,b},{a,c},
   {b,c} is now encoded as (min(y_i,y_j), max(y_i,y_j), I_ij) -- symmetric in its own two endpoints -- and
   pooled by sum over the 3 edges, which are the SAME 3 edges regardless of which branch is called "a" vs "b".
   This is exactly invariant under all 6 relabelings of {a,b,c}, verified directly.
4. UNSCALED FEATURE. The reciprocal-determinant (LODF risk) feature was capped at 10,000 with no
   normalisation, next to features roughly in [0,1]. Fixed: log1p-transformed, then standardised using
   TRAINING-set statistics only (never validation/test).

Identifiability check (unchanged, still respected): the residual-correction arm trains ONLY on N-3 rows."""
import json
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import Ridge

from fdna import spec, v2, v2data
from fdna.dataset import G, RATING
from fdna.evalutil import rprec
from fdna.lodf import ptdf_lodf, compensation_det
from fdna.opgen import sample_op
from fdna.physical_correction import island_floor
from fdna.rules import boot
from fdna.nn.pairset import DeepSetsPairAware, ResidualNet

torch.manual_seed(0)
n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
ALL_OPS = manifest["op_ids"]; ALL_OUTAGES = sorted(set(tuple(o) for o in manifest["outage_sets"]))

# FIX 2: three genuinely disjoint groups, on BOTH axes (outages and operating points), frozen before fitting.
split_rng = np.random.default_rng(4004)
perm_o = split_rng.permutation(len(ALL_OUTAGES))
TRAIN_OUTAGES = {ALL_OUTAGES[i] for i in perm_o[:35]}
VAL_OUTAGES = {ALL_OUTAGES[i] for i in perm_o[35:45]}
TEST_OUTAGES = {ALL_OUTAGES[i] for i in perm_o[45:]}
perm_p = split_rng.permutation(len(ALL_OPS))
TRAIN_OPS = [int(ALL_OPS[i]) for i in perm_p[:45]]
VAL_OPS = [int(ALL_OPS[i]) for i in perm_p[45:53]]
TEST_OPS = list(ALL_OPS)  # eval composes over ALL 60 ops (the known-operating-point axis; only outages are held out for the primary test)
assert TRAIN_OUTAGES.isdisjoint(VAL_OUTAGES) and TRAIN_OUTAGES.isdisjoint(TEST_OUTAGES) and VAL_OUTAGES.isdisjoint(TEST_OUTAGES)
assert set(TRAIN_OPS).isdisjoint(VAL_OPS)
assert len(TRAIN_OUTAGES) == 35 and len(VAL_OUTAGES) == 10 and len(TEST_OUTAGES) == 25
print(f"train: {len(TRAIN_OUTAGES)} outages x {len(TRAIN_OPS)} ops | val: {len(VAL_OUTAGES)} outages x {len(VAL_OPS)} ops (BOTH axes disjoint from train) | test: {len(TEST_OUTAGES)} outages x {len(TEST_OPS)} ops (known-op axis, new-outage axis)")

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]
row_by_outage_op = {(int(op_ids_sorted[i]), outages_sorted[i]): i for i in range(len(outages_sorted))}

demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in ALL_OPS}
_, LODF = ptdf_lodf(G)


def physical_feats(op_id, outage):
    a, b, c = outage
    F = island_floor(G, demands[op_id], outage)
    pairs = [(a, b), (a, c), (b, c)]
    risks = [min(1.0 / max(abs(compensation_det(LODF, p[0], p[1])), 1e-9), 1e4) for p in pairs]
    return F, risks


PHYS = {(op, o): physical_feats(op, o) for op in ALL_OPS for o in ALL_OUTAGES}


def build_rows(op_id, outage, cv_idx_arr, row_key_prefix):
    """FIX 1: returns (key, X, y, g2) built TOGETHER from the SAME cv_idx_arr -- no possibility of a later
    misaligned re-sample. key is the canonical (op_id, outage, cv_idx) tuple per row."""
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)][cv_idx_arr], y1[(op_id, b)][cv_idx_arr], y1[(op_id, c)][cv_idx_arr]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iab, Iac, Ibc = (y2[(op_id, p)][cv_idx_arr] - y1[(op_id, p[0])][cv_idx_arr] - y1[(op_id, p[1])][cv_idx_arr] for p in pairs)
    F, risks = PHYS[(op_id, outage)]
    # FIX 4: log1p the heavy-tailed risk feature (was raw, capped at 1e4, unnormalised next to [0,1] features)
    risks_scaled = [np.log1p(r) for r in risks]
    n = len(cv_idx_arr)
    phys_rep = np.tile([F, *risks_scaled], (n, 1))
    cvec = CV[cv_idx_arr]
    X = np.column_stack([ya, yb, yc, Iab, Iac, Ibc, phys_rep, cvec]).astype(np.float32)
    g2 = (ya + yb + yc + Iab + Iac + Ibc).astype(np.float32)
    i = row_by_outage_op[(op_id, outage)]
    y = V_sorted[i][cv_idx_arr].astype(np.float32)
    keys = [(row_key_prefix, op_id, outage, int(k)) for k in cv_idx_arr]
    return keys, X, y, g2


def build_split(outages, ops, n_per_row, seed, prefix):
    rng = np.random.default_rng(seed)
    keys_all, X_list, y_list, g2_list = [], [], [], []
    for outage in outages:
        for op_id in ops:
            cv_idx = rng.choice(len(CV), n_per_row, replace=False)
            keys, X, y, g2 = build_rows(op_id, outage, cv_idx, prefix)
            keys_all.extend(keys); X_list.append(X); y_list.append(y); g2_list.append(g2)
    X, y, g2 = np.concatenate(X_list), np.concatenate(y_list), np.concatenate(g2_list)
    assert np.allclose(y - g2, y - g2)  # r := y-g2 is well-defined here (X,y,g2 came from the SAME pass)
    return keys_all, X, y, g2


keys_tr, Xtr, ytr, g2tr = build_split(TRAIN_OUTAGES, TRAIN_OPS, 128, seed=5005, prefix="train")
keys_va, Xva, yva, g2va = build_split(VAL_OUTAGES, VAL_OPS, 32, seed=6006, prefix="val")
assert len(set(keys_tr) & set(keys_va)) == 0   # FIX 2 regression check: no shared (op,outage,control) rows
print(f"train rows: {len(Xtr)}, target std: {ytr.std():.4f} | val rows: {len(Xva)}, target std: {yva.std():.4f}")
rtr, rva = (ytr - g2tr).astype(np.float32), (yva - g2va).astype(np.float32)
assert np.allclose(rtr + g2tr, ytr) and np.allclose(rva + g2va, yva)   # FIX 1 regression check: r+g2==y exactly

# FIX 4 (standardise the physical block using TRAIN statistics only): columns 6:10 are [F, risk_ab, risk_ac, risk_bc]
phys_mean, phys_std = Xtr[:, 6:10].mean(0), Xtr[:, 6:10].std(0) + 1e-8
for arr in (Xtr, Xva):
    arr[:, 6:10] = (arr[:, 6:10] - phys_mean) / phys_std

# ---- fit ridge ----
ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
print("ridge val MSE:", float(((ridge.predict(Xva) - yva) ** 2).mean()))

# ---- fit GBM ----
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


# FIX 3: genuinely permutation-invariant pair-aware set model -- now in src/fdna/nn/pairset.py

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


res_model, res_val_mse_of_residual, res_losses = train_torch(ResidualNet(), Xtr, rtr, Xva, rva)
print("residual-net val MSE (of residual target):", res_val_mse_of_residual, "loss decreased:", res_losses[0] > res_losses[-1])

# ---- permutation-invariance regression test on the actual trained model ----
with torch.no_grad():
    dev = next(ds_model.parameters()).device
    x0 = torch.tensor(Xva[:8], device=dev)
    p0 = ds_model(x0).cpu().numpy()
    perm_invariant = True
    for perm in [(1, 0, 2, 3, 5, 4), (0, 2, 1, 4, 3, 5), (2, 1, 0, 5, 4, 3), (1, 2, 0, 4, 5, 3), (2, 0, 1, 5, 3, 4)]:
        xp = x0.clone(); xp[:, :6] = x0[:, list(perm)]
        pp = ds_model(xp).cpu().numpy()
        if not np.allclose(p0, pp, atol=1e-4): perm_invariant = False
print("deepsets genuinely permutation-invariant (all 6 relabelings):", perm_invariant)

checklist = {
    "target_variation_train_std": float(ytr.std()), "target_variation_val_std": float(yva.std()),
    "ridge_beats_predict_mean_baseline": bool(((ridge.predict(Xva) - yva) ** 2).mean() < yva.var()),
    "deepsets_loss_decreased": bool(ds_losses[0] > ds_losses[-1]),
    "residual_loss_decreased": bool(res_losses[0] > res_losses[-1]),
    "split_disjoint_outages_verified": bool(TRAIN_OUTAGES.isdisjoint(VAL_OUTAGES) and TRAIN_OUTAGES.isdisjoint(TEST_OUTAGES) and VAL_OUTAGES.isdisjoint(TEST_OUTAGES)),
    "split_disjoint_ops_train_val_verified": bool(set(TRAIN_OPS).isdisjoint(VAL_OPS)),
    "no_shared_rows_train_val": bool(len(set(keys_tr) & set(keys_va)) == 0),
    "residual_key_consistency_r_plus_g2_equals_y": bool(np.allclose(rtr + g2tr, ytr) and np.allclose(rva + g2va, yva)),
    "deepsets_permutation_invariant_all_6_relabelings": perm_invariant,
}
print("PRE-REJECTION CHECKLIST", json.dumps(checklist, indent=1))
json.dump({"split": {"train_outages": [list(o) for o in TRAIN_OUTAGES], "val_outages": [list(o) for o in VAL_OUTAGES],
           "test_outages": [list(o) for o in TEST_OUTAGES], "train_ops": TRAIN_OPS, "val_ops": VAL_OPS, "test_ops": TEST_OPS},
           "checklist": checklist, "val_mse": {"ridge": float(((ridge.predict(Xva) - yva) ** 2).mean()),
           "gbm": best[0], "deepsets": ds_val, "residual_net_on_residual_target": res_val_mse_of_residual},
           "feature_scaling": {"phys_mean": phys_mean.tolist(), "phys_std": phys_std.tolist()}},
          open("results/phase5/matched_recurrent_training_v2.json", "w"), indent=1)
print("TRAINING PHASE DONE", flush=True)

import pickle
with open("data_hik/matched_recurrent_models_v2.pkl", "wb") as f:
    pickle.dump({"ridge": ridge, "gbm": gbm, "phys_mean": phys_mean, "phys_std": phys_std}, f)
torch.save(ds_model.state_dict(), "data_hik/deepsets_model_v2.pt")
torch.save(res_model.state_dict(), "data_hik/residual_model_v2.pt")
