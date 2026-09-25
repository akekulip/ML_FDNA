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


class OrderedMLP(nn.Module):
    """Canonical ordered baseline for repair round 3's ablation (external review section 4): a plain MLP
    directly on the raw 15-column row X = [ya,yb,yc, Iab,Iac,Ibc, F, risk_ab,risk_ac,risk_bc, c1..c5], with NO
    pooling step -- deliberately NOT invariant to relabeling {a,b,c}. Isolates whether the POOLING architecture
    itself (vs. the invariance property per se) explains DeepSets' worse score after the full invariance fix.
    Matched rough capacity to DeepSetsPairAware(h=64) (9,089 params): this net has 9,409 (~3.5% more, not a
    capacity confound in either direction). See tests/test_ordered_and_perm_averaged.py for the required
    negative control (this model must NOT be invariant on a real checkpoint, or the ablation's premise breaks)."""
    def __init__(self, d_in: int = 15, h: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


class PermAveragedModel(nn.Module):
    """Wraps ANY already-trained model taking the 15-column row convention, making it EXACTLY invariant to all
    6 vertex relabelings by construction, at prediction time only -- averaging the whole raw-input-to-prediction
    function over the permutation group (external review section 4: "Averaging the whole raw-input-to-
    prediction function over the permutation group is invariant by construction, even when its base model is
    not; charge its six forward passes."). Not itself trained -- `base` is frozen/eval'd by the caller.

    Safe to apply directly to already-scaled rows: `pairset_features.apply_scaler` uses ONE shared (mean,std)
    across risk columns 7/8/9 (`fit_shared_risk_scaler`), so permuting a scaled row is equivalent to permuting
    then scaling. This wrapper must NOT be used on raw features with a per-slot scaler -- there is no such
    scaler in this repo, but the invariant is stated explicitly since it's the exact bug class this codebase has
    hit twice before (repair rounds 1 and 2)."""
    def __init__(self, base: nn.Module, column_perms: list):
        super().__init__()
        self.base = base
        self.column_perms = column_perms

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.base(X[:, perm]) for perm in self.column_perms], dim=0).mean(0)


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
