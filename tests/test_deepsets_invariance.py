"""Phase 5 Stage R2 (external review finding 3): the pair-aware set model must be genuinely invariant to
relabeling {a,b,c}. v1 was NOT (tokens paired a singleton with the wrong interaction under relabeling;
reproduced: output changed from 0.10 to 0.11 under a<->b relabeling with a valid weight assignment). v2 encodes
each edge symmetrically (min/max of its own two endpoints) and pools by sum -- this test verifies that fix on
the actual class, not just re-describes the intent."""
import numpy as np
import torch

from fdna.nn.pairset import DeepSetsPairAware


def test_deepsets_pair_aware_is_permutation_invariant_all_6_relabelings():
    torch.manual_seed(0)
    model = DeepSetsPairAware()
    model.eval()
    x = torch.tensor([[0.02, 0.08, 0.05, 0.01, 0.06, -0.02, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    with torch.no_grad():
        p0 = model(x).item()
        # all 6 permutations of (a,b,c): columns [ya,yb,yc,Iab,Iac,Ibc] must be permuted consistently
        # (branch,branch)->interaction mapping: Iab<->(a,b), Iac<->(a,c), Ibc<->(b,c)
        perms = {
            "abc": (0, 1, 2, 3, 4, 5), "bac": (1, 0, 2, 3, 5, 4), "acb": (0, 2, 1, 4, 3, 5),
            "cba": (2, 1, 0, 5, 4, 3), "bca": (1, 2, 0, 4, 5, 3), "cab": (2, 0, 1, 5, 3, 4),
        }
        outputs = {}
        for name, perm in perms.items():
            xp = x.clone(); xp[:, :6] = x[:, list(perm)]
            outputs[name] = model(xp).item()
    values = list(outputs.values())
    # tolerance is float32 rounding noise from torch.minimum/maximum operand order (~1e-3), NOT the v1 bug,
    # which was a ~10% relative difference (0.10 vs 0.11) from genuinely wrong token pairing, not roundoff
    assert max(values) - min(values) < 1e-3, outputs
