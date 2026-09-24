"""Phase 3 Step 1e: can the CURRENT FDNALayer represent known-answer motifs? Fit alpha/beta/u by gradient descent from several inits to the exact truth table.
Motifs: AND3 (CC,RTU,GW) ; AND_OR (CC & RTU & (GW1 | GW2)) ; mandatory-only 2-input AND. Metric: max abs error over all binary inputs (best of restarts)."""
import itertools, json
import torch; torch.set_num_threads(1)
import numpy as np
import torch
from fdna.nn.layers import FDNALayer

def truth(kind):
    X = np.array(list(itertools.product([0, 1], repeat=4 if kind == "and_or" else 3 if kind == "and3" else 2)), np.float32)
    if kind == "and2": y = X[:, 0] * X[:, 1]
    elif kind == "and3": y = X[:, 0] * X[:, 1] * X[:, 2]
    else: y = X[:, 0] * X[:, 1] * np.maximum(X[:, 2], X[:, 3])
    return torch.tensor(X), torch.tensor(y)

out = {}
for kind in ("and2", "and3", "and_or"):
    X, y = truth(kind); n = X.shape[1]
    for tau in (0.05, 0.01, 0.0):
        best = 9.0
        for seed in range(12):
            torch.manual_seed(seed)
            L = FDNALayer(torch.ones(1, n), tau=tau)
            with torch.no_grad():
                L.alpha.normal_(0, 2); L.beta.normal_(-1, 2); L.u.normal_(2, 1)
            opt = torch.optim.Adam(L.parameters(), lr=0.05)
            for _ in range(600):
                opt.zero_grad(); loss = ((L(X).squeeze(-1) - y) ** 2).mean(); loss.backward(); opt.step()
            err = (L(X).squeeze(-1) - y).abs().max().item(); best = min(best, err)
        out[f"{kind}_tau{tau}"] = round(best, 3)
print(json.dumps(out, indent=1)); json.dump(out, open("results/phase3/step1_motifs.json", "w"), indent=1)
