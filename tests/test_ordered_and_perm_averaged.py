"""Phase 5 repair round 3: the DeepSets ablation (external review section 4) isolates whether the POOLING
architecture itself explains DeepSets' worse post-fix score, vs. the simultaneous normalization/optimization
changes. `OrderedMLP` (a plain, deliberately non-invariant MLP on the raw 15-column row) is the ablation's
non-invariant baseline; `PermAveragedModel` wraps ANY trained model, averaging its output over all 6
relabelings -- exactly invariant by construction, regardless of the base model's own behavior. Both checked
here on an untrained (freshly initialized) instance, same pattern as tests/test_deepsets_invariance.py:
unequal risk values, all 6 relabelings generated programmatically via pairset_features, calibrated tolerance."""
import numpy as np
import torch

from fdna.nn.pairset import OrderedMLP, PermAveragedModel
from fdna.nn.pairset_features import ALL_VERTEX_PERMS, column_perm_for_vertex_perm

_COLUMN_PERMS = [column_perm_for_vertex_perm(sigma) for sigma in ALL_VERTEX_PERMS]
_X = torch.tensor([[0.02, 0.08, 0.05, 0.01, 0.06, -0.02, 0.0, 0.3, 1.1, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0]])


def test_ordered_mlp_is_not_permutation_invariant_negative_control():
    """Required negative control: if OrderedMLP were somehow invariant, the ablation's premise (that pooling
    architecture is the thing being isolated) would be broken. Confirmed on unequal risks, all 6 relabelings."""
    torch.manual_seed(0)
    model = OrderedMLP()
    model.eval()
    with torch.no_grad():
        outputs = [float(model(_X[:, perm]).item()) for perm in _COLUMN_PERMS]
    spread = max(outputs) - min(outputs)
    assert spread > 1e-3, (outputs, "OrderedMLP should NOT be invariant -- if it is, the ablation is broken")


def test_perm_averaged_model_is_invariant_by_construction_regardless_of_base():
    """PermAveragedModel must be exactly invariant even wrapping a base model that itself is NOT (OrderedMLP,
    confirmed non-invariant above) -- averaging over the whole permutation group is invariant by construction."""
    torch.manual_seed(0)
    base = OrderedMLP()
    base.eval()
    wrapped = PermAveragedModel(base, _COLUMN_PERMS)
    wrapped.eval()
    with torch.no_grad():
        p0 = float(wrapped(_X).item())
        outputs = [float(wrapped(_X[:, perm]).item()) for perm in _COLUMN_PERMS]

        # calibrated tolerance: measured fp32-vs-fp64 rounding on the SAME (unpermuted) input, same approach as
        # tests/test_deepsets_invariance.py -- distinguishes genuine roundoff from a real defect
        import copy
        wrapped64 = copy.deepcopy(wrapped).double()
        eps0 = float(abs(p0 - wrapped64(_X.double()).item()))
        tol = max(10 * eps0, 1e-6)

    spread = max(outputs) - min(outputs)
    assert spread < tol, (outputs, f"spread={spread:.2e} tol={tol:.2e} eps0={eps0:.2e}")
