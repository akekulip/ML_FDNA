"""Phase 5: a genuinely permutation-invariant pair-aware set model for a 3-element outage set {a,b,c}, fixing
a real bug an external review found in the earlier inline version (src/fdna/nn was the wrong place for it to
live -- it was embedded directly in a top-level script, which is why the bug survived: importing the class for
testing required running the whole training pipeline). Each of the 3 EDGES {a,b},{a,c},{b,c} is encoded as
(min(y_i,y_j), max(y_i,y_j), I_ij) -- symmetric in its own two endpoints -- then pooled by SUM over the 3
edges, which are the SAME 3 edges regardless of which branch is called "a" vs "b". Exactly invariant under all
6 relabelings of {a,b,c}, verified in tests/test_deepsets_invariance.py."""
from __future__ import annotations

import torch
import torch.nn as nn


class DeepSetsPairAware(nn.Module):
    def __init__(self, h: int = 64):
        super().__init__()
        self.edge_enc = nn.Sequential(nn.Linear(3, h), nn.ReLU(), nn.Linear(h, h))
        self.readout = nn.Sequential(nn.Linear(h + 4 + 5, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X columns: [ya, yb, yc, Iab, Iac, Ibc, F, risk_ab, risk_ac, risk_bc, c1..c5]."""
        ya, yb, yc, Iab, Iac, Ibc = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5]
        edges = torch.stack([
            torch.stack([torch.minimum(ya, yb), torch.maximum(ya, yb), Iab], 1),
            torch.stack([torch.minimum(ya, yc), torch.maximum(ya, yc), Iac], 1),
            torch.stack([torch.minimum(yb, yc), torch.maximum(yb, yc), Ibc], 1),
        ], 1)  # (B, 3, 3)
        e = self.edge_enc(edges).sum(1)
        return self.readout(torch.cat([e, X[:, 6:]], 1)).squeeze(-1)


class ResidualNet(nn.Module):
    """Retains the exact g2 term separately (added by the caller); learns ONLY the correction."""
    def __init__(self, d_in: int = 15, h: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)
