import numpy as np
import pandas as pd

from fdna.baseline_data import Data, evaluate, strata


def test_strata_partition_and_evaluate():
    n = 60
    lab = pd.DataFrame(dict(op=np.repeat([1, 2], 30), cell=["n2_unseen"] * n, b1=np.arange(n) % 41, b2=-1, fs=0,
                            split="test", y=np.tile([0.0, 0.05, 0.0], 20), y_struct=0.0))
    d = Data(lab=lab, F={}, y=lab.y.values, key=np.random.default_rng(0).random(n), tr=np.zeros(n, bool),
             va=np.zeros(n, bool), te=np.ones(n, bool), novel=np.arange(n) % 2 == 0,
             fs_size=np.tile([2, 3, 4], 20), loss=np.zeros(n, bool))
    S = strata(d)
    assert (S["n2_unseen|familiar"] ^ S["n2_unseen|novel"]).sum() == n  # exact partition of the cell
    assert sum(S[f"n2_unseen|size{k}"].sum() for k in (2, 3, 4)) == n
    rows = evaluate(d, "perfect", d.y.copy(), 0, {"n2_unseen": S["n2_unseen"]})
    assert len(rows) == 2 and all(abs(r["rprec"] - 1.0) < 1e-12 for r in rows)
    assert all(abs(r["prev"] - 1 / 3) < 1e-9 for r in rows)
