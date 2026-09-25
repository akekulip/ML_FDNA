"""Phase 5: a genuinely permutation-invariant pair-aware set model for a 3-element outage set {a,b,c}.

v1 (external review, repair round 1) had the edge/interaction pairing itself wrong (a singleton paired with the
WRONG interaction term under relabeling) -- fixed by encoding each of the 3 EDGES {a,b},{a,c},{b,c} as
(min(y_i,y_j), max(y_i,y_j), I_ij) and pooling by SUM. v2 (this version; external review, repair round 2) found
that fix was still incomplete: the readout concatenated the raw, per-edge `risk_ab/risk_ac/risk_bc` features
AFTER pooling, in a fixed slot order -- a genuine unordered-edge quantity handled asymmetrically, the exact
same class of bug as the one already fixed in the edge encoder, just one layer downstream of it. Fixed by
folding `risk_ij` into its own edge's token as a 4th component, so it travels through the SUM pool with the
edge it belongs to; only F (a whole-set topological invariant) and c1..c5 (the generator control vector, never
per-edge) remain outside the pooled representation, since neither depends on vertex labeling. Preprocessing
must match: see `pairset_features.fit_shared_risk_scaler` (one pooled scaler over all 3 risk slots, not 3
separate per-slot ones -- a fixed architecture alone does not fix a per-slot-dependent normalisation).
Verified end-to-end (raw features -> scaler -> model) in tests/test_deepsets_invariance.py."""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsPairAware(nn.Module):
    def __init__(self, h: int = 64):
        super().__init__()
        self.edge_enc = nn.Sequential(nn.Linear(4, h), nn.ReLU(), nn.Linear(h, h))
        self.readout = nn.Sequential(nn.Linear(h + 1 + 5, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X columns: [ya, yb, yc, Iab, Iac, Ibc, F, risk_ab, risk_ac, risk_bc, c1..c5]."""
        ya, yb, yc, Iab, Iac, Ibc = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
        F = X[:, 6]
        risk_ab, risk_ac, risk_bc = X[:, 7], X[:, 8], X[:, 9]
        c = X[:, 10:15]
        edges = torch.stack([
            torch.stack([torch.minimum(ya, yb), torch.maximum(ya, yb), Iab, risk_ab], 1),
            torch.stack([torch.minimum(ya, yc), torch.maximum(ya, yc), Iac, risk_ac], 1),
            torch.stack([torch.minimum(yb, yc), torch.maximum(yb, yc), Ibc, risk_bc], 1),
        ], 1)  # (B, 3, 4) -- risk_XX now travels WITH its own edge's (min,max,I) triple
        e = self.edge_enc(edges).sum(1)
        return self.readout(torch.cat([e, F.unsqueeze(1), c], 1)).squeeze(-1)


class ResidualNet(nn.Module):
    """Retains the exact g2 term separately (added by the caller); learns ONLY the correction.
    prediction = g2 + lam * r_theta(X). The final layer is zero-initialized AND `lam` is a separate free
    scalar, so r_theta(X)=0 for every X at initialization regardless of lam's value -- the untrained/zero-
    correction point is a real, reachable point at step 0, not an accident of checkpoint selection never being
    evaluated before the first optimizer step (external review, repair round 2: the residual arm's failure was
    previously undiagnosed -- no explicit zero-correction incumbent existed to compare against)."""
    def __init__(self, d_in: int = 15, h: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.lam = nn.Parameter(torch.tensor(1.0))

    def raw_correction(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.lam * self.raw_correction(X)
