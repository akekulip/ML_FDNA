"""Phase 4 Stage 3: the brief's actual staged comparator set (section 6.1/6.3) -- a tuned tree on permutation-
invariant aggregate descriptors, an invariant DeepSets model, and a GRU (order-sensitive by construction),
ALL trained ONLY on N-1+N-2 (data_hik/train_n1n2.npz), evaluated ZERO-SHOT on the frozen N-3 and N-4 manifests.
Compared against the closed-form g2/g3 Mobius truncation (scripts/p4_mobius_k3_test.py, p4_mobius_k4_test.py).
Properly batched/padded (mask-based), not a per-sample Python loop.

Required checks per brief 6.5: permutation sensitivity (GRU fed sorted vs 5 random orders per eval set)."""
import json
import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(4)
import lightgbm as lgb
from scipy.stats import spearmanr

from fdna.dataset import G, RATING

torch.manual_seed(0)
deg = np.zeros(G.n_bus)
for f, t in zip(G.frm, G.to):
    deg[f] += 1; deg[t] += 1
PHI = np.stack([G.x, RATING, deg[G.frm], deg[G.to]], axis=1).astype(np.float32)
PHI = (PHI - PHI.mean(0)) / (PHI.std(0) + 1e-9)


def pad_batch(seqs, max_len, op_ids=None, y_single=None):
    """y_single: dict (op_id, branch) -> known N-1 shed value, appended as an extra per-element feature (equal-
    information test: give the learned models the SAME solved sub-answers g1/g2/g3 use, not just static descriptors)."""
    B = len(seqs)
    D = PHI.shape[1] + (1 if y_single is not None else 0)
    X = np.zeros((B, max_len, D), np.float32)
    mask = np.zeros((B, max_len), np.float32)
    for i, s in enumerate(seqs):
        for j, b in enumerate(s):
            feat = PHI[b]
            if y_single is not None:
                op = int(op_ids[i]); feat = np.concatenate([feat, [y_single.get((op, b), 0.0)]])
            X[i, j] = feat; mask[i, j] = 1.0
    return torch.tensor(X), torch.tensor(mask)


tr = np.load("data_hik/train_n1n2.npz")
Xtr_seq = [[int(b) for b in row if b >= 0] for row in tr["outages"]]
ytr = tr["y"].astype(np.float32)
rng = np.random.default_rng(0); perm = rng.permutation(len(ytr))
val_n = len(ytr) // 5
val_idx, train_idx = perm[:val_n], perm[val_n:]


def agg_features(seqs):
    feats = []
    for s in seqs:
        f = PHI[list(s)]
        feats.append(np.concatenate([f.sum(0), f.max(0), [len(s)]]))
    return np.array(feats, np.float32)


# build y_single (fresh N-1 solves, same fixed control state) for every (op,branch) that appears anywhere below --
# gives the learned models the SAME first-order solved information g1/g2/g3 use (equal-information comparator).
from fdna import spec
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op
k3d = np.load("data_hik/k3_screen.npz"); CV_IDX_GLOBAL = int(np.argmax(k3d["CV"].sum(1)))
k4_manifest = json.load(open("data_hik/manifest_k4_screen.json"))
all_op_ids = sorted(set(int(x) for x in tr["op_ids"]) | set(int(x) for x in k3d["op_ids"]) | set(int(x) for x in k4_manifest["op_ids"]))
Y_SINGLE = {}
for op_id in all_op_ids:
    op = sample_op(G, RATING, np.random.default_rng(op_id))
    for b in range(G.n_branch):
        Y_SINGLE[(op_id, b)] = ScenarioLP(G, op, (b,), spec.PARAMS).y(k3d["CV"][CV_IDX_GLOBAL])
print(f"Y_SINGLE built for {len(all_op_ids)} ops x {G.n_branch} branches", flush=True)

Xtr_agg = agg_features(Xtr_seq)
gbm = lgb.LGBMRegressor(n_estimators=300, num_leaves=15, min_child_samples=20, verbose=-1, random_state=0)
gbm.fit(Xtr_agg[train_idx], ytr[train_idx], eval_set=[(Xtr_agg[val_idx], ytr[val_idx])], callbacks=[lgb.early_stopping(20, verbose=False)])


class DeepSets(nn.Module):
    def __init__(self, d=4, h=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, h))
        self.readout = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, mask):
        e = self.enc(X) * mask.unsqueeze(-1)
        return self.readout(e.sum(1)).squeeze(-1)


class GRUModel(nn.Module):
    def __init__(self, d=4, h=32):
        super().__init__()
        self.enc = nn.Linear(d, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.readout = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X, mask):
        e = self.enc(X)
        lengths = mask.sum(1).clamp(min=1).long().cpu()
        packed = nn.utils.rnn.pack_padded_sequence(e, lengths, batch_first=True, enforce_sorted=False)
        _, hN = self.gru(packed)
        return self.readout(hN[0]).squeeze(-1)


def train_torch(model, seqs, y, max_len, op_ids, epochs=150, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    yt = torch.tensor(y)
    Xall, Mall = pad_batch(seqs, max_len, op_ids, Y_SINGLE)
    best_val, best_state, bad = 1e30, None, 0
    for ep in range(epochs):
        model.train()
        idx = rng.permutation(len(train_idx))
        for a in range(0, len(idx), 512):
            bidx = train_idx[idx[a:a + 512]]
            opt.zero_grad()
            pred = model(Xall[bidx], Mall[bidx])
            loss = ((pred - yt[bidx]) ** 2).mean()
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vpred = model(Xall[val_idx], Mall[val_idx])
            vloss = ((vpred - yt[val_idx]) ** 2).mean().item()
        if vloss < best_val - 1e-7:
            best_val, best_state, bad = vloss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 15:
                break
    model.load_state_dict(best_state)
    return model


MAX_LEN_TRAIN = 2
ds = train_torch(DeepSets(d=5), Xtr_seq, ytr, MAX_LEN_TRAIN, tr["op_ids"])
gru = train_torch(GRUModel(d=5), Xtr_seq, ytr, MAX_LEN_TRAIN, tr["op_ids"])
print("training done", flush=True)


def evaluate(name, manifest_path, npz_path, max_len):
    manifest = json.load(open(manifest_path))
    d = np.load(npz_path)
    seqs = [tuple(int(x) for x in o) for o in d["outages"]]
    if "V" in d.files:
        CV = d["CV"]; cv_idx = int(np.argmax(CV.sum(1))); y_true = d["V"][:, cv_idx]
    else:
        y_true = d["y"]
    op_ids_eval = d["op_ids"]
    Xagg = agg_features(seqs)
    pred_gbm = gbm.predict(Xagg)
    Xb, Mb = pad_batch(seqs, max_len, op_ids_eval, Y_SINGLE)
    with torch.no_grad():
        pred_ds = ds(Xb, Mb).numpy()
        pred_gru_sorted = gru(*pad_batch([tuple(sorted(s)) for s in seqs], max_len, op_ids_eval, Y_SINGLE)).numpy()
        perm_preds = []
        for k in range(5):
            shuffled = [tuple(np.random.default_rng(k).permutation(list(s))) for s in seqs]
            perm_preds.append(gru(*pad_batch(shuffled, max_len, op_ids_eval, Y_SINGLE)).numpy())
        perm_preds = np.array(perm_preds)
    mae = lambda p: float(np.abs(y_true - p).mean()); rho = lambda p: float(spearmanr(p, y_true)[0]) if p.std() > 0 else None
    out = {"n": len(y_true),
           "gbm_agg_features": {"mae": mae(pred_gbm), "rho": rho(pred_gbm)},
           "deepsets": {"mae": mae(pred_ds), "rho": rho(pred_ds)},
           "gru_sorted": {"mae": mae(pred_gru_sorted), "rho": rho(pred_gru_sorted)},
           "gru_permutation_sensitivity": {"max_abs_diff_vs_sorted": float(np.abs(perm_preds - pred_gru_sorted[None]).max()),
                                            "mean_abs_diff_vs_sorted": float(np.abs(perm_preds - pred_gru_sorted[None]).mean())}}
    print(name, json.dumps(out, indent=1), flush=True)
    return out


out = {"k3": evaluate("k3", "data_hik/manifest_k3_screen.json", "data_hik/k3_screen.npz", 3),
       "k4": evaluate("k4", "data_hik/manifest_k4_screen.json", "data_hik/k4_screen.npz", 4)}
json.dump(out, open("results/phase4/setmodels_k3k4.json", "w"), indent=1)
