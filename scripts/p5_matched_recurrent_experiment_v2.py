"""Phase 5 Stage R1.1: matched recurrent/learned-correction experiment, v2 -- fixes four bugs an external
review found in v1 (repair round 1; kept, not deleted, for the record):

1. VALIDATION TARGET MISALIGNMENT (critical). v1 sampled Xva/yva with one RNG draw, then later advanced the
   SAME RNG to sample g2va at DIFFERENT control indices, so rva=yva-g2va compared mismatched states.
   Reproduced exactly: 4,796/4,800 mismatched indices. Fixed: (key, X, y, g2) are now built together in ONE
   pass, for both validation and training.
2. VALIDATION/TRAIN OVERLAP. v1's validation reused the first 10 TRAINING outages and the first 15 (of all 60)
   TRAINING operating points. Fixed: outages AND operating points are split into three groups, TRAIN/VAL
   genuinely disjoint on both axes; TEST reuses all 60 ops against 25 held-out outages by design (the known-
   operating-point axis, see the split print below) -- never claimed disjoint from train/val on the op axis.
3. THE "COMMUTATIVE" MODEL WAS NOT ACTUALLY PERMUTATION-INVARIANT. Fixed in two rounds: repair round 1 fixed
   the edge/interaction pairing (each of the 3 edges now a symmetric (min,max,I) token, sum-pooled); repair
   round 2 (this version) fixed a SECOND bug an independent review found -- the readout concatenated the raw,
   per-edge risk features AFTER pooling in a fixed order, so relabeling still changed the output. Now each
   risk travels WITH its own edge token (src/fdna/nn/pairset.py), and preprocessing uses ONE shared scaler
   across all 3 risk slots (src/fdna/nn/pairset_features.fit_shared_risk_scaler), not 3 separate per-slot ones.
4. UNSCALED FEATURE. The reciprocal-determinant (LODF risk) feature was capped at 10,000 with no
   normalisation, next to features roughly in [0,1]. Fixed: log1p-transformed, then standardised using
   TRAINING-set statistics only (never validation/test).

Repair round 2 also fixes two VACUOUS regression guards an independent review found: v1's
`assert np.allclose(y-g2, y-g2)` compared an array to itself (always true, tests nothing), and
`assert np.allclose(rtr+g2tr, ytr)` was an algebraic tautology given `rtr := ytr-g2tr` two lines above (cannot
detect a misaligned g2). Both replaced with a genuine independent reconstruction from canonical keys, re-derived
directly from the raw n1/n2/n3 tables, not from the already-built arrays. The key-overlap check also used to
embed the split name in the key (`(prefix, op_id, outage, cv_idx)`), making the intersection empty by
construction regardless of real overlap; keys are now canonical `(op_id, outage, cv_idx)` with no prefix.

Repair round 2 also adds a residual-model diagnostic protocol (external review: the residual arm's failure was
undiagnosed -- no explicit zero-correction incumbent, checkpoint selection used the wrong metric, no multi-seed
check, no tiny-batch overfit sanity check). See train_residual_with_diagnostics below.

Repair round 3 (a THIRD independent review) corrects two overclaims from round 2 and adds a bounded ablation:
- The residual diagnostic (`_diag_rprec`) previously pooled a random 12-pair subset of VAL_OUTAGES x VAL_OPS
  into ONE ranking, P1-only -- coarse (only 2 distinct achievable values across 17 checked epochs) and a
  protocol mismatch with final eval (which ranks per-op then averages). Fixed: uses the FULL VAL_OUTAGES x
  VAL_OPS cross-product, ranked per-op then averaged, for BOTH cells (scalar selection = mean of the two).
- `lam` is trained JOINTLY with the final linear layer, so its value is not independently identifiable as a
  validated shrinkage factor. Fixed: `select_frozen_shrinkage` freezes the trained correction and sweeps an
  independent, prespecified `SHRINK_GRID` on validation using the same per-op/dual-cell diagnostic.
- DeepSets ablation (external review section 4): "the earlier model benefited from forbidden information" was
  withdrawn as overclaimed (the pair-risk features are legitimate inputs available to every arm) -- but WHETHER
  the score change was caused by the pooling architecture itself, vs. the simultaneous normalization/
  optimization changes, was genuinely unresolved. `OrderedMLP` (a plain, deliberately non-invariant MLP on the
  same raw features) and `PermAveragedModel` (wraps any trained model, averaging its output over all 6
  relabelings -- exactly invariant by construction) isolate this, holding data/features/split/seeds fixed.

Identifiability check (unchanged, still respected): the residual-correction arm trains ONLY on N-3 rows."""
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import Ridge

from fdna import spec, v2, v2data
from fdna.dataset import G, RATING
from fdna.evalutil import rprec, evaluate_rprec_for_preds
from fdna.lodf import ptdf_lodf
from fdna.opgen import sample_op
from fdna.rules import boot
from fdna.nn.pairset import DeepSetsPairAware, ResidualNet, OrderedMLP, PermAveragedModel
from fdna.nn import pairset_features as pf

torch.manual_seed(0)
n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
ALL_OPS = manifest["op_ids"]; ALL_OUTAGES = sorted(set(tuple(o) for o in manifest["outage_sets"]))

# three groups, TRAIN/VAL genuinely disjoint on BOTH axes (outages and operating points), frozen before fitting
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
print(f"train: {len(TRAIN_OUTAGES)} outages x {len(TRAIN_OPS)} ops | val: {len(VAL_OUTAGES)} outages x {len(VAL_OPS)} ops (BOTH axes disjoint from train) | test: {len(TEST_OUTAGES)} outages x {len(TEST_OPS)} ops (known-op axis, new-outage axis -- NOT disjoint from train/val on the op axis, by design)")

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]
row_by_outage_op = {(int(op_ids_sorted[i]), outages_sorted[i]): i for i in range(len(outages_sorted))}

demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in ALL_OPS}
_, LODF = ptdf_lodf(G)
PHYS = {(op, o): pf.physical_feats(G, LODF, demands[op], o) for op in ALL_OPS for o in ALL_OUTAGES}


def build_split(outages, ops, n_per_row, seed):
    """Canonical keys (op_id, outage, cv_idx), no split-name prefix -- overlap checks below are genuine."""
    rng = np.random.default_rng(seed)
    keys_all, X_list, y_list, g2_list = [], [], [], []
    for outage in outages:
        for op_id in ops:
            cv_idx = rng.choice(len(CV), n_per_row, replace=False)
            i = row_by_outage_op[(op_id, outage)]
            y_true = V_sorted[i][cv_idx]
            X, y, g2 = pf.build_row(op_id, outage, cv_idx, y1=y1, y2=y2, phys=PHYS[(op_id, outage)], CV=CV, y_true=y_true)
            keys_all.extend((op_id, outage, int(k)) for k in cv_idx)
            X_list.append(X); y_list.append(y); g2_list.append(g2)
    return keys_all, np.concatenate(X_list), np.concatenate(y_list), np.concatenate(g2_list)


def reconstruct_y_g2(key):
    """Independent reconstruction directly from the raw n1/n2/n3 tables, given ONLY the canonical key -- unlike
    the old `assert allclose(rtr+g2tr, ytr)` (a tautology given rtr:=ytr-g2tr), this recomputes y and g2 from
    scratch and would catch a misaligned g2 (repair round 2, replacing a vacuous regression guard)."""
    op_id, outage, cv_idx = key
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)][cv_idx], y1[(op_id, b)][cv_idx], y1[(op_id, c)][cv_idx]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iab, Iac, Ibc = (y2[(op_id, p)][cv_idx] - y1[(op_id, p[0])][cv_idx] - y1[(op_id, p[1])][cv_idx] for p in pairs)
    g2 = float(ya + yb + yc + Iab + Iac + Ibc)
    y = float(V_sorted[row_by_outage_op[(op_id, outage)]][cv_idx])
    return y, g2


keys_tr, Xtr, ytr, g2tr = build_split(TRAIN_OUTAGES, TRAIN_OPS, 128, seed=5005)
keys_va, Xva, yva, g2va = build_split(VAL_OUTAGES, VAL_OPS, 32, seed=6006)
assert len(set(keys_tr) & set(keys_va)) == 0   # genuine now: canonical keys, no split-prefix tainting
print(f"train rows: {len(Xtr)}, target std: {ytr.std():.4f} | val rows: {len(Xva)}, target std: {yva.std():.4f}")

# genuine independent reconstruction check (repair round 2): sample 300 keys from each split, re-derive y/g2
# from the raw tables via the canonical key alone, compare to what build_row actually produced
_check_rng = np.random.default_rng(9009)
_recon_ok = True
for keys, X, y, g2 in ((keys_tr, Xtr, ytr, g2tr), (keys_va, Xva, yva, g2va)):
    sample_idx = _check_rng.choice(len(keys), min(300, len(keys)), replace=False)
    for i in sample_idx:
        y_recon, g2_recon = reconstruct_y_g2(keys[i])
        if not (np.isclose(y_recon, y[i], atol=1e-5) and np.isclose(g2_recon, g2[i], atol=1e-5)):
            _recon_ok = False
            break
print("independent key-based reconstruction matches built arrays:", _recon_ok)
assert _recon_ok

rtr, rva = (ytr - g2tr).astype(np.float32), (yva - g2va).astype(np.float32)

# shared risk-slot scaler (repair round 2 fix: was 3 SEPARATE per-slot means/stds, which alone breaks
# invariance even with a fixed architecture, since a risk value's normalized value depended on which slot --
# ab, ac, or bc -- it happened to land in)
phys_mean, phys_std = pf.fit_shared_risk_scaler(Xtr)
for arr in (Xtr, Xva):
    pf.apply_scaler(arr, phys_mean, phys_std)

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


## ---- repair round 3: DeepSets/ordered/perm-averaged ablation, 3 predeclared seeds each ----
# Isolates architecture (pooling vs. ordered+6-way averaging) while holding data/features/split/seeds fixed.
# Both DeepSetsPairAware and OrderedMLP use the SAME train_torch selection protocol (plain val-MSE), so neither
# gets an advantage from a different checkpoint-selection rule. PermAveragedModel is a prediction-time-only
# wrapper (not trained) around the already-trained OrderedMLP -- 6 forward passes, charged explicitly below.
ABLATION_SEEDS = [0, 1, 2]
COLUMN_PERMS = [pf.column_perm_for_vertex_perm(sigma) for sigma in pf.ALL_VERTEX_PERMS]
ds_models, ordered_models = {}, {}
deepsets_seed_results, ordered_seed_results = [], []
ds_losses = None
for seed in ABLATION_SEEDS:
    torch.manual_seed(seed)
    m, val, losses = train_torch(DeepSetsPairAware(), Xtr, ytr, Xva, yva)
    ds_models[seed] = m
    deepsets_seed_results.append({"seed": seed, "val_mse": val, "loss_decreased": bool(losses[0] > losses[-1])})
    torch.save(m.state_dict(), "data_hik/deepsets_model_v2.pt" if seed == 0 else f"data_hik/deepsets_model_v2_seed{seed}.pt")
    if seed == 0:
        ds_losses = losses  # kept for the existing downstream checklist entry (deepsets_loss_decreased)

    torch.manual_seed(seed)
    mo, valo, losseso = train_torch(OrderedMLP(), Xtr, ytr, Xva, yva)
    ordered_models[seed] = mo
    ordered_seed_results.append({"seed": seed, "val_mse": valo, "loss_decreased": bool(losseso[0] > losseso[-1])})
    torch.save(mo.state_dict(), f"data_hik/ordered_model_v2_seed{seed}.pt")
ds_model, ds_val = ds_models[0], deepsets_seed_results[0]["val_mse"]  # keep existing downstream names
print("deepsets val MSE (seed 0):", ds_val, "multi-seed:", deepsets_seed_results)
print("ordered MLP val MSE, multi-seed:", ordered_seed_results)

# Sanity check (NOT a strict negative control -- see finding below): OrderedMLP is architecturally CAPABLE of
# non-invariance (confirmed on a fresh random init in tests/test_ordered_and_perm_averaged.py, spread>1e-3), but
# whether a TRAINED checkpoint actually uses that capability on this data is a separate, empirical question.
# Checked across a larger sample (64 rows, not 8) and all 3 seeds, not just one -- a genuine finding, not a fluke.
ordered_max_spread_by_seed = {}
with torch.no_grad():
    for seed in ABLATION_SEEDS:
        dev_o = next(ordered_models[seed].parameters()).device
        x0o = torch.tensor(Xva[:64], device=dev_o)
        po0 = ordered_models[seed](x0o).cpu().numpy()
        ordered_max_spread_by_seed[seed] = max(
            float(np.abs(po0 - ordered_models[seed](x0o[:, perm]).cpu().numpy()).max()) for perm in COLUMN_PERMS)
ordered_max_spread = ordered_max_spread_by_seed[0]
print(f"ordered MLP trained-checkpoint permutation spread, all seeds (64 rows): {ordered_max_spread_by_seed}")
# NOTE on sample size: an earlier pass of this check used only 8 Xva rows and found a near-zero spread
# (~3e-6), which looked like a real finding (the trained model converging to an approximately symmetric
# solution unforced) but turned out to be a small-sample artifact -- with 64 rows and all 3 seeds checked here,
# the spread is 0.012-0.022, comparable in scale to the target's own std (0.033-0.037), i.e. genuinely
# substantial. Caught before it was written up as a finding anywhere: the trained OrderedMLP checkpoints DO
# meaningfully exploit branch-position ordering, confirming the ablation's negative control holds as expected,
# not the "gradient descent finds symmetry unforced" story a smaller sample suggested.

# permutation-averaged val MSE per seed -- diagnostic only, no training, just wraps the already-trained OrderedMLP
perm_avg_seed_results = []
for seed in ABLATION_SEEDS:
    pa = PermAveragedModel(ordered_models[seed], COLUMN_PERMS).eval()
    dev_pa = next(ordered_models[seed].parameters()).device
    with torch.no_grad():
        pa_val = float(((pa(torch.tensor(Xva, device=dev_pa)) - torch.tensor(yva, device=dev_pa)) ** 2).mean().item())
    perm_avg_seed_results.append({"seed": seed, "val_mse": pa_val})
print("ordered perm-averaged val MSE, multi-seed:", perm_avg_seed_results)

# ---- permutation-invariance regression check on the actual trained model, FULL pipeline ----
# repair round 2: uses pf.column_perm_for_vertex_perm (programmatically generated, all 6 relabelings, validated
# against the docstring's own worked example) instead of manually-maintained tuples, on Xva rows that carry
# REAL (unequal) risk values -- not the fabricated equal-risk rows that made the old dedicated test blind to
# this exact bug. Tolerance is calibrated per-run against measured fp32-vs-fp64 rounding on the SAME input,
# not an ad hoc constant.
with torch.no_grad():
    dev = next(ds_model.parameters()).device
    x0 = torch.tensor(Xva[:8], device=dev)
    p0 = ds_model(x0).cpu().numpy()
    ds_model64 = copy.deepcopy(ds_model).double()
    eps0 = float(np.abs(p0 - ds_model64(x0.double()).cpu().numpy()).max())
    tol = max(10 * eps0, 1e-6)
    perm_invariant = True
    max_spread = 0.0
    for sigma in pf.ALL_VERTEX_PERMS:
        perm = pf.column_perm_for_vertex_perm(sigma)
        xp = x0[:, perm]
        pp = ds_model(xp).cpu().numpy()
        max_spread = max(max_spread, float(np.abs(p0 - pp).max()))
    perm_invariant = max_spread < tol
print(f"deepsets genuinely permutation-invariant (all 6 relabelings, full pipeline): {perm_invariant} (max_spread={max_spread:.2e}, tol={tol:.2e}, fp32/fp64 eps0={eps0:.2e})")


# ---- residual model: diagnostic protocol (repair round 2, corrected round 3) ----
# g2_fixed (already in the final arm comparison, see p5_matched_recurrent_eval_v2.py) IS the residual arm's
# zero-correction (lambda=0) baseline -- ResidualNet's final layer is zero-initialized (src/fdna/nn/pairset.py)
# so that point is a real, reachable step-(-1) state here, not an accident of best_val never being checked
# before the first optimizer step.
w = v2.build_world()
DIAG_CELLS = {"P1": (0.7, 0.3, None), "P2": (0.3, 0.2, v2.COVERAGE_SPARSE)}
RESIDUAL_SEEDS = [0, 1, 2]  # predeclared before any result is examined
SHRINK_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]  # prespecified, for the frozen-shrinkage experiment below

# repair round 3 (external review finding 5): the diagnostic previously sampled a random 12-pair SUBSET of
# VAL_OUTAGES x VAL_OPS and pooled them into ONE ranking -- both a mismatch with final eval (which ranks WITHIN
# each op then averages) and needlessly coarse (12 blocks -> only 2 distinct achievable rprec values across 17
# checked epochs). Fixed: use the FULL VAL_OUTAGES x VAL_OPS cross-product (10x8=80 pairs, still cheap -- no new
# LP solves, just forward passes over cached grids), grouped by operating point, ranked per-op then averaged --
# the same protocol p5_matched_recurrent_eval_v2.py's final table uses. Extended to BOTH cells (was P1-only);
# the scalar selection criterion is the mean of P1 and P2 (both cell values are still recorded individually).
DIAG_VAL_PAIRS = [(o, p) for o in VAL_OUTAGES for p in VAL_OPS]


def _full_grid_features(op_id, outage):
    idx_all = np.arange(len(CV))
    y_full = V_sorted[row_by_outage_op[(op_id, outage)]]
    X, y, g2 = pf.build_row(op_id, outage, idx_all, y1=y1, y2=y2, phys=PHYS[(op_id, outage)], CV=CV, y_true=y_full)
    pf.apply_scaler(X, phys_mean, phys_std)
    return X, g2


def _diag_rprec(predict_fn) -> dict:
    """predict_fn(Xg_tensor) -> np.ndarray of predictions over the full CV grid for one (op_id,outage). Returns
    {"P1": per-op-mean rprec, "P2": per-op-mean rprec} over the full VAL_OUTAGES x VAL_OPS cross-product,
    matching the final eval script's own per-op-then-average protocol."""
    per_cell = {}
    for cell_name, (q, s, cov) in DIAG_CELLS.items():
        per_op = {op: {"p": [], "y": [], "k": []} for op in VAL_OPS}
        for outage, op_id in DIAG_VAL_PAIRS:
            Xg, g2g = _full_grid_features(op_id, outage)
            pred_g = predict_fn(Xg, g2g)
            y_true_grid = V_sorted[row_by_outage_op[(op_id, outage)]].astype(np.float64)
            p, yv_, k = evaluate_rprec_for_preds(w, q, s, cov, op_id, outage, y_true_grid, pred_g, seed=101)
            per_op[op_id]["p"].append(p); per_op[op_id]["y"].append(yv_); per_op[op_id]["k"].append(k)
        per_op_rprec = [rprec(np.concatenate(d["y"]), np.concatenate(d["p"]), np.concatenate(d["k"]))
                        for d in per_op.values() if d["p"]]
        per_cell[cell_name] = float(np.mean(per_op_rprec))
    return per_cell


def _model_predict_fn(model, dev):
    def f(Xg, g2g):
        with torch.no_grad():
            return g2g + model(torch.tensor(Xg, device=dev)).cpu().numpy()
    return f


def train_residual_with_diagnostics(Xtr, rtr, Xva, rva, seed, epochs=80, lr=2e-3, dev="cuda", check_every=5):
    """Selects the final checkpoint by mean(P1,P2) per-op R-precision on the full validation cross-product
    (ties broken by value MSE), not raw value-MSE alone. Seeds best_val/best_state from an epoch-(-1) evaluation
    BEFORE any optimizer step -- the reachable zero-correction point competes in selection, rather than being
    skipped by construction."""
    torch.manual_seed(seed)
    model = ResidualNet(); model.to(dev)
    Xt, yt = torch.tensor(Xtr, device=dev), torch.tensor(rtr, device=dev)
    Xv, yv = torch.tensor(Xva, device=dev), torch.tensor(rva, device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    model.eval()
    with torch.no_grad():
        vloss0 = ((model(Xv) - yv) ** 2).mean().item()
    rprec0 = _diag_rprec(_model_predict_fn(model, dev))
    scalar0 = float(np.mean(list(rprec0.values())))
    history = [{"epoch": -1, "value_mse_val": vloss0, "rprec_val": rprec0, "lam": float(model.lam.item())}]
    best_key = (scalar0, -vloss0)
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    best_epoch = -1
    losses = [vloss0]
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
        rprec_ep = None
        if (ep + 1) % check_every == 0 or ep == epochs - 1:
            rprec_ep = _diag_rprec(_model_predict_fn(model, dev))
            cand = (float(np.mean(list(rprec_ep.values()))), -vloss)
            if cand > best_key:
                best_key = cand
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                best_epoch = ep
        history.append({"epoch": ep, "value_mse_val": vloss, "rprec_val": rprec_ep, "lam": float(model.lam.item())})
    model.load_state_dict(best_state)
    return model, best_key[0], -best_key[1], history, best_epoch, losses


def select_frozen_shrinkage(model, dev) -> dict:
    """External review finding 5: `lam` is trained JOINTLY with the final linear layer, so its learned value is
    not independently identifiable as a validated shrinkage factor -- the layer's own weights can absorb any
    rescaling of lam. Fixed: FREEZE the trained model's raw (unscaled) correction (`raw_correction`, already
    exposed as a separate method precisely for this), and sweep a prespecified, independent SHRINK_GRID on
    VALIDATION using the same per-op/dual-cell diagnostic as checkpoint selection -- selecting a shrink value
    this way is a genuine, properly-separated experiment, not a jointly-optimized scalar."""
    model.eval()
    scored = {}
    for shrink in SHRINK_GRID:
        def f(Xg, g2g, shrink=shrink):
            with torch.no_grad():
                return g2g + shrink * model.raw_correction(torch.tensor(Xg, device=dev)).cpu().numpy()
        r = _diag_rprec(f)
        scored[shrink] = {"P1": r["P1"], "P2": r["P2"], "mean": float(np.mean([r["P1"], r["P2"]]))}
    best_shrink = max(scored, key=lambda k: scored[k]["mean"])
    return {"grid": scored, "selected_shrink": best_shrink, "selected_on": "validation, mean(P1,P2) per-op rprec"}


def tiny_batch_overfit_check(Xtr, rtr, group_ids, n=12, iters=1000, seed=0):
    """Can a fresh ResidualNet memorize 12 fixed points at all? A failure points to an optimization/objective
    bug; a pass alongside poor val/test performance is real evidence of 'no generalizable signal', a stronger
    and more honest conclusion than an undiagnosed 'worst arm'.

    Picks ONE row from each of the n outage/op BLOCKS with the largest single-row |residual| (the row within
    each block with max |r|), not simply the n largest-|r| rows overall: the identifiability check already
    guarantees g2 is exact for size<=2 sets, but even among N-3 rows a whole (outage,op) block can have r==0
    throughout if that triple's third-order Mobius interaction happens to be zero (mechanism-1, see
    STAGE2_PHYSICAL_CORRECTION.md) -- taking literal row 0:12 would then hand the net nothing to overfit
    (confirmed: happened on the first pass). Taking the top-12 |r| rows WITHOUT the block constraint has a
    second, subtler failure mode -- also confirmed by running it -- a single block's residual is roughly
    CONSTANT across nearby controls, so the top-12 by raw |r| cluster within one or two blocks and end up
    with near-zero VARIANCE despite large magnitude, making the pass/fail threshold vacuous in the other
    direction. Forcing one row per distinct block guarantees genuine diversity."""
    uniq_groups = np.unique(group_ids)
    block_best_row, block_best_absr = {}, {}
    for i, g in enumerate(group_ids):
        if g not in block_best_absr or abs(rtr[i]) > block_best_absr[g]:
            block_best_absr[g] = abs(rtr[i]); block_best_row[g] = i
    top_groups = sorted(uniq_groups, key=lambda g: -block_best_absr[g])[:n]
    idx = np.array([block_best_row[g] for g in top_groups])
    torch.manual_seed(seed)
    m = ResidualNet()
    Xs, ys = torch.tensor(Xtr[idx]), torch.tensor(rtr[idx])
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=0.0)
    for _ in range(iters):
        opt.zero_grad(); pred = m(Xs); loss = ((pred - ys) ** 2).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        final_mse = float(((m(Xs) - ys) ** 2).mean().item())
    target_var = float(np.var(rtr[idx]))
    ok = final_mse < max(1e-4 * target_var, 1e-8)
    return bool(ok), final_mse, target_var


residual_seed_results = []
for seed in RESIDUAL_SEEDS:
    res_model_s, best_rprec_s, best_vmse_s, res_history_s, best_epoch_s, res_losses_s = \
        train_residual_with_diagnostics(Xtr, rtr, Xva, rva, seed=seed)
    residual_seed_results.append({"seed": seed, "best_rprec_val": best_rprec_s, "best_value_mse_val": best_vmse_s,
                                   "best_epoch": best_epoch_s, "final_lambda": float(res_model_s.lam.item())})
    print(f"residual seed {seed}: best rprec_val={best_rprec_s:.4f} (epoch {best_epoch_s}), value_mse={best_vmse_s:.3e}, lambda={res_model_s.lam.item():.3f}")
    if seed == RESIDUAL_SEEDS[0]:
        res_model, res_val_mse_of_residual, res_losses, res_history = res_model_s, best_vmse_s, res_losses_s, res_history_s
        torch.save(res_model.state_dict(), "data_hik/residual_model_v2.pt")
    else:
        torch.save(res_model_s.state_dict(), f"data_hik/residual_model_v2_seed{seed}.pt")

# frozen-shrinkage-grid experiment on the canonical seed (external review finding 5) -- freeze res_model's
# raw_correction, sweep SHRINK_GRID on validation, select by the same per-op/dual-cell diagnostic
frozen_shrinkage = select_frozen_shrinkage(res_model, next(res_model.parameters()).device)
print("frozen-shrinkage selection (canonical seed 0):", json.dumps(frozen_shrinkage, indent=1))

_block_id = {}
for k in keys_tr:
    block = (k[0], k[1])  # (op_id, outage), dropping the cv_idx
    if block not in _block_id:
        _block_id[block] = len(_block_id)
group_ids_tr = np.array([_block_id[(k[0], k[1])] for k in keys_tr])
tiny_overfit_ok, tiny_overfit_mse, tiny_overfit_target_var = tiny_batch_overfit_check(Xtr, rtr, group_ids_tr)
print("residual tiny-batch overfit check:", tiny_overfit_ok, "final_mse:", tiny_overfit_mse, "target_var:", tiny_overfit_target_var)

checklist = {
    "target_variation_train_std": float(ytr.std()), "target_variation_val_std": float(yva.std()),
    "ridge_beats_predict_mean_baseline": bool(((ridge.predict(Xva) - yva) ** 2).mean() < yva.var()),
    "deepsets_loss_decreased": bool(ds_losses[0] > ds_losses[-1]),
    "residual_loss_decreased": bool(res_losses[0] > res_losses[-1]),
    "split_disjoint_outages_verified": bool(TRAIN_OUTAGES.isdisjoint(VAL_OUTAGES) and TRAIN_OUTAGES.isdisjoint(TEST_OUTAGES) and VAL_OUTAGES.isdisjoint(TEST_OUTAGES)),
    "split_disjoint_ops_train_val_verified": bool(set(TRAIN_OPS).isdisjoint(VAL_OPS)),
    "no_shared_rows_train_val": bool(len(set(keys_tr) & set(keys_va)) == 0),
    "independent_key_reconstruction_matches": _recon_ok,
    "deepsets_permutation_invariant_all_6_relabelings_full_pipeline": perm_invariant,
    "residual_tiny_batch_overfit_ok": tiny_overfit_ok,
    "ordered_mlp_trained_checkpoint_permutation_spread_by_seed": ordered_max_spread_by_seed,
}
print("PRE-REJECTION CHECKLIST", json.dumps(checklist, indent=1))
json.dump({"split": {"train_outages": [list(o) for o in TRAIN_OUTAGES], "val_outages": [list(o) for o in VAL_OUTAGES],
           "test_outages": [list(o) for o in TEST_OUTAGES], "train_ops": TRAIN_OPS, "val_ops": VAL_OPS, "test_ops": TEST_OPS},
           "checklist": checklist, "val_mse": {"ridge": float(((ridge.predict(Xva) - yva) ** 2).mean()),
           "gbm": best[0], "deepsets": ds_val, "residual_net_on_residual_target": res_val_mse_of_residual},
           "feature_scaling": {"phys_mean": phys_mean.tolist(), "phys_std": phys_std.tolist()},
           "permutation_invariance": {"max_spread": max_spread, "tolerance": tol, "fp32_fp64_eps0": eps0},
           "residual_diagnostics": {"seeds": residual_seed_results, "canonical_seed_history": res_history,
                                     "frozen_shrinkage_selection": frozen_shrinkage,
                                     "tiny_batch_overfit": {"ok": tiny_overfit_ok, "final_mse": tiny_overfit_mse,
                                                             "target_var": tiny_overfit_target_var}},
           "ablation_ordered_vs_pooled": {
               "seeds": ABLATION_SEEDS,
               "deepsets_multiseed": deepsets_seed_results,
               "ordered_mlp_multiseed": ordered_seed_results,
               "ordered_perm_averaged_multiseed": perm_avg_seed_results,
               "ordered_mlp_param_count": sum(p.numel() for p in OrderedMLP().parameters()),
               "deepsets_param_count": sum(p.numel() for p in DeepSetsPairAware().parameters()),
               "scope": "isolates architecture (pooling vs ordered+6-way averaging) holding data/features/split/"
                        "seeds fixed; does NOT establish whether ordering is a real transferable signal on other "
                        "topologies or branch renumberings (external review section 4, repair round 3).",
           }},
          open("results/phase5/matched_recurrent_training_v2.json", "w"), indent=1)
print("TRAINING PHASE DONE", flush=True)

import pickle
with open("data_hik/matched_recurrent_models_v2.pkl", "wb") as f:
    pickle.dump({"ridge": ridge, "gbm": gbm, "phys_mean": phys_mean, "phys_std": phys_std}, f)
torch.save(ds_model.state_dict(), "data_hik/deepsets_model_v2.pt")
