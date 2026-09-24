"""Phase 4 Stage 3: generate exact LP labels for the frozen k=3 manifest (data_hik/manifest_k3_screen.json).
Mirrors value_table.py's multiprocessing pattern. Infeasible/failed solves are counted, not dropped."""
import json, time
from multiprocessing import Pool

import numpy as np

from fdna import hik, v2, spec
from fdna.dataset import G

WORLD = v2.build_world(); CV = WORLD.CV
manifest = json.load(open("data_hik/manifest_k3_screen.json"))
OP_IDS, OUTAGES = manifest["op_ids"], [tuple(s) for s in manifest["outage_sets"]]


def job(args):
    op_id, outage = args
    y, any_infeasible, note = hik.label_kset(G, op_id, outage, CV)
    return op_id, outage, y, any_infeasible, note


if __name__ == "__main__":
    import os
    t0 = time.time()
    tasks = [(o, s) for o in OP_IDS for s in OUTAGES]
    n_failed_rows = 0
    with Pool(int(os.environ.get("N_PROC", 24))) as pool:
        results = list(pool.imap_unordered(job, tasks))
    op_ids = np.array([r[0] for r in results])
    outages = np.array([list(r[1]) for r in results])
    V = np.array([r[2] for r in results], np.float32)
    infeasible = np.array([r[3] for r in results])
    n_failed_rows = int(infeasible.sum())
    np.savez("data_hik/k3_screen.npz", op_ids=op_ids, outages=outages, V=V, CV=CV, infeasible=infeasible)
    (out := open("data_hik/k3_screen_run.json", "w")).write(json.dumps({
        "spec_hash": spec.spec_hash(), "n_rows": len(results), "n_rows_with_any_infeasible_control": n_failed_rows,
        "wall_s": time.time() - t0, "manifest": "manifest_k3_screen.json"}, indent=1)); out.close()
    print("done", len(results), "rows,", n_failed_rows, "with infeasible controls,", round(time.time() - t0), "s")
