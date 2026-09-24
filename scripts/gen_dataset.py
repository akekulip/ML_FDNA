"""Generate reference labels. Usage: gen_dataset.py [out_dir] [n_train n_val n_test] [test_op_start]"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fdna import spec
from fdna.dataset import assemble, job

if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    n_tr, n_va, n_te = (int(x) for x in sys.argv[2:5]) if len(sys.argv) > 4 else (100, 20, 80)
    te0 = int(sys.argv[5]) if len(sys.argv) > 5 else 200
    assert n_tr <= 100 and n_va <= 100 and te0 >= 200, "operating-point id ranges must not overlap (train<100, val 100-199, test>=200)"
    out.mkdir(exist_ok=True)
    jobs = [("train", i) for i in range(0, n_tr)] + [("val", i) for i in range(100, 100 + n_va)] \
        + [("test", i) for i in range(te0, te0 + n_te)]
    res = []
    with Pool(30) as pool:
        for k, r in enumerate(pool.imap_unordered(job, jobs)):
            res.append(r)
            if k % 20 == 0:
                print(k, "/", len(jobs), flush=True)
    lab, ops = assemble(res)
    lab.to_parquet(out / "labels.parquet")
    np.savez(out / "ops.npz", **ops)
    (out / "spec_hash.txt").write_text(spec.spec_hash())
    print("spec", spec.spec_hash(), "rows", len(lab))
