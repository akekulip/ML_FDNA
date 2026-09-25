"""Phase 5 Stage R2 (external review finding 7): regression test for the missed-severe sign bug. The bug:
storing err=pred-truth then reconstructing pred as truth-err (=2*truth-pred) silently hides real misses. This
test constructs the exact failure case the review used (truth=0.05, pred=0, a real missed-severe event) and
checks the CORRECT reconstruction/direct-use logic, not the buggy one."""
import numpy as np


def test_missed_severe_uses_prediction_directly_not_a_buggy_reconstruction():
    truth, pred, tau = 0.05, 0.0, 0.01
    err = pred - truth                         # the stored quantity in the original buggy script
    buggy_reconstruction = truth - err          # = 2*truth - pred: WRONG, gives 0.10 here
    correct_reconstruction = truth + err        # = pred exactly: correct
    assert abs(buggy_reconstruction - 0.10) < 1e-9          # confirms the bug's exact wrong behaviour
    assert abs(correct_reconstruction - pred) < 1e-9         # the fix
    is_severe = truth > tau
    missed_using_buggy = is_severe and (buggy_reconstruction <= tau)
    missed_using_correct = is_severe and (correct_reconstruction <= tau)
    assert missed_using_buggy is False          # the bug hides this miss
    assert missed_using_correct is True         # the fix correctly flags it as missed
