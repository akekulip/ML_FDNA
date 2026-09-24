"""LP value table V[op, outage, control vector] for all 1024 control vectors of the v2 world.
Usage: value_table.py OUT_DIR SPLIT:START:STOP[:N_N2] ...   e.g. train:0:100:0 val:100:120:80 test:200:280:150"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from fdna import spec, v2
from fdna.dataset import G, PAIRS, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

WORLD = v2.build_world()
CV = WORLD.CV


def contingencies(op_id: int, n_n2: int, with_n0: bool):
    conts = ([(-1, -1)] if with_n0 else []) + [(b, -1) for b in range(G.n_branch)]
    if n_n2:
        idx = np.random.default_rng(20_000 + op_id).choice(len(PAIRS), n_n2, replace=False)
        conts += [PAIRS[i] for i in sorted(idx)]
    return conts


def job(args):
    op_id, n_n2, with_n0 = args
    op = sample_op(G, RATING, np.random.default_rng(op_id))
    conts = contingencies(op_id, n_n2, with_n0)
    V = np.zeros((len(conts), len(CV)), np.float32)
    for i, (a, b) in enumerate(conts):
        lp = ScenarioLP(G, op, tuple(x for x in (a, b) if x >= 0), spec.PARAMS)
        for k in range(len(CV)):
            V[i, k] = lp.y(CV[k])
    return op_id, np.array(conts, np.int16), V


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(exist_ok=True)
    plans = []
    for a in sys.argv[2:]:
        name, lo, hi, n2 = a.split(":")
        plans.append((name, list(range(int(lo), int(hi))), int(n2), name == "train"))
    t0 = time.time()
    with Pool(24) as pool:
        for name, ops, n2, n0 in plans:
            res = sorted(pool.imap_unordered(job, [(o, n2, n0) for o in ops]), key=lambda r: r[0])
            np.savez(out / f"vtable_{name}.npz", op_ids=np.array([r[0] for r in res]),
                     conts=np.array([r[1] for r in res]), V=np.array([r[2] for r in res]), CV=CV)
            print(name, len(ops), "ops done", round(time.time() - t0), "s", flush=True)
    (out / "spec_hash.txt").write_text(spec.spec_hash())
