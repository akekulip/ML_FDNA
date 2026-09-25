"""Phase 4 confirmatory: generate N-1 (all 41 branches, full control grid), N-2 (needed sub-pairs, full grid),
and N-3 (the 70 frozen triples, full grid) labels for the 60 fresh reserve operating points. All infeasible
solves counted, not dropped. Reports total LP-solve cost per label type (brief's cost-accounting requirement)."""
import json, time
from itertools import combinations
from multiprocessing import Pool

import numpy as np

from fdna import spec, v2
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

WORLD = v2.build_world(); CV = WORLD.CV
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]; OUTAGES = [tuple(s) for s in manifest["outage_sets"]]
NEEDED_PAIRS = sorted({tuple(sorted((o[i], o[j]))) for o in OUTAGES for i, j in combinations(range(3), 2)})
print(f"60 ops x (41 singles + {len(NEEDED_PAIRS)} pairs + 70 triples) x 1024 controls")


def job(args):
    op_id, outage = args
    op = sample_op(G, RATING, np.random.default_rng(int(op_id)))
    lp = ScenarioLP(G, op, outage, spec.PARAMS)
    y = np.zeros(len(CV), np.float32); n_fail = 0
    for k in range(len(CV)):
        try:
            y[k] = lp.y(CV[k])
        except RuntimeError:
            y[k] = np.nan; n_fail += 1
    return op_id, outage, y, n_fail


if __name__ == "__main__":
    import os
    t0 = time.time()
    all_outages = [(b,) for b in range(G.n_branch)] + NEEDED_PAIRS + OUTAGES
    tasks = [(o, out) for o in OP_IDS for out in all_outages]
    print(f"total tasks (op,outage) pairs: {len(tasks)}, total LP solves: {len(tasks)*len(CV)}")
    with Pool(int(os.environ.get("N_PROC", 24))) as pool:
        results = list(pool.imap_unordered(job, tasks))
    n_fail_total = sum(r[3] for r in results)
    by_len = {1: [], 2: [], 3: []}
    for op_id, outage, y, nf in results:
        by_len[len(outage)].append((op_id, outage, y))
    def save(name, rows):
        op_ids = np.array([r[0] for r in rows])
        outages = np.array([list(r[1]) + [-1] * (3 - len(r[1])) for r in rows])
        Y = np.array([r[2] for r in rows], np.float32)
        np.savez(f"data_hik/{name}_confirm.npz", op_ids=op_ids, outages=outages, V=Y, CV=CV)
        print(name, len(rows), "rows")
    save("n1", by_len[1]); save("n2", by_len[2]); save("n3", by_len[3])
    manifest_run = {"total_lp_solves": len(tasks) * len(CV), "n_infeasible": int(n_fail_total), "wall_s": round(time.time() - t0, 1),
                    "n1_rows": len(by_len[1]), "n2_rows": len(by_len[2]), "n3_rows": len(by_len[3])}
    json.dump(manifest_run, open("data_hik/confirm_run.json", "w"), indent=1)
    print(manifest_run)
