"""Phase 4 partial-obs validation: generate y_pair across ALL 1024 control vectors (needed to compose g2(op,S,c)
for every c, not just one fixed control). Reuses the k=3 manifest's 156 needed sub-pairs x 20 ops. y_single is
NOT regenerated -- it is read directly from data_v2/vtable_train.npz's existing dense N-1 rows (free)."""
import json, time
from multiprocessing import Pool
import numpy as np

from fdna import spec, v2
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

WORLD = v2.build_world(); CV = WORLD.CV
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]
NEEDED_PAIRS = sorted({tuple(sorted((o[i], o[j]))) for o in OUTAGES for i, j in [(0, 1), (0, 2), (1, 2)]})
print(f"{len(NEEDED_PAIRS)} pairs x {len(OP_IDS)} ops x 1024 controls = {len(NEEDED_PAIRS)*len(OP_IDS)*1024} solves")


def job(args):
    op_id, pair = args
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    lp = ScenarioLP(G, op, pair, spec.PARAMS)
    y = np.array([lp.y(CV[k]) for k in range(len(CV))], np.float32)
    return op_id, pair, y


if __name__ == "__main__":
    import os
    t0 = time.time()
    tasks = [(o, p) for o in OP_IDS for p in NEEDED_PAIRS]
    with Pool(int(os.environ.get("N_PROC", 24))) as pool:
        results = list(pool.imap_unordered(job, tasks))
    op_ids = np.array([r[0] for r in results])
    pairs = np.array([list(r[1]) for r in results])
    Y = np.array([r[2] for r in results], np.float32)
    np.savez("data_hik/ypair_fullgrid.npz", op_ids=op_ids, pairs=pairs, Y=Y, CV=CV)
    print("done", len(results), "rows,", round(time.time() - t0), "s")
