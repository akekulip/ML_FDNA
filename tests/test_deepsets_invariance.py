"""Phase 5 repair round 2: the pair-aware set model must be genuinely invariant to relabeling {a,b,c},
INCLUDING the per-edge risk features. Round 1 fixed the edge/interaction encoder but left the readout
concatenating raw, unpermuted risk features after pooling; the round-1 test used EQUAL risk values
(1.0,1.0,1.0) and only permuted columns 0-5, so it could not detect this -- permuting three identical numbers
changes nothing. This test uses UNEQUAL risks and generates all 6 relabelings programmatically via
fdna.nn.pairset_features.column_perm_for_vertex_perm (not hand-written tuples -- an earlier hand-written
version of this test had two of its six tuples wrong, caught by an independent qa-verifier pass)."""
import numpy as np
import torch

from fdna.nn.pairset import DeepSetsPairAware
from fdna.nn.pairset_features import ALL_VERTEX_PERMS, column_perm_for_vertex_perm


def test_deepsets_pair_aware_is_permutation_invariant_all_6_relabelings_unequal_risks():
    torch.manual_seed(0)
    model = DeepSetsPairAware()
    model.eval()
    # UNEQUAL risk_ab/ac/bc (columns 7,8,9) -- the exact condition the round-1 test could not exercise
    x = torch.tensor([[0.02, 0.08, 0.05, 0.01, 0.06, -0.02, 0.0, 0.3, 1.1, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        outputs = {}
        for sigma in ALL_VERTEX_PERMS:
            perm = column_perm_for_vertex_perm(sigma)
            outputs[sigma] = float(model(x[:, perm]).item())
        p0 = outputs[(0, 1, 2)]

        # calibrated tolerance: measured fp32-vs-fp64 rounding on the SAME (unpermuted) input, not an ad hoc
        # constant -- distinguishes genuine roundoff from a real architectural defect
        import copy
        model64 = copy.deepcopy(model).double()
        eps0 = float(abs(p0 - model64(x.double()).item()))
        tol = max(10 * eps0, 1e-6)

    values = list(outputs.values())
    spread = max(values) - min(values)
    assert spread < tol, (outputs, f"spread={spread:.2e} tol={tol:.2e} eps0={eps0:.2e}")


def test_deepsets_readout_is_actually_sensitive_to_the_risk_features():
    """Negative control: confirms the fix is substantive, not merely a no-op reshape. If `risk_ab/ac/bc` were
    accidentally dropped, broadcast away, or zeroed by a shape bug in edge_enc's 4th input, this model would
    ALSO look permutation-invariant (three degenerate zero components can't break invariance) but for the
    wrong reason. Perturbing only the risk values (holding min/max/I fixed) must change the output for at
    least one of a handful of perturbations -- confirms the 4th edge-token component is genuinely wired in."""
    torch.manual_seed(0)
    model = DeepSetsPairAware()
    model.eval()
    x = torch.tensor([[0.02, 0.08, 0.05, 0.01, 0.06, -0.02, 0.0, 0.3, 1.1, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        p0 = model(x).item()
        changed = False
        for delta in (1.0, -1.0, 5.0, -5.0):
            xp = x.clone(); xp[:, 7] = xp[:, 7] + delta  # perturb risk_ab only
            if abs(model(xp).item() - p0) > 1e-4:
                changed = True
                break
        assert changed, "output did not respond to any risk_ab perturbation -- the risk feature may not be wired in"
