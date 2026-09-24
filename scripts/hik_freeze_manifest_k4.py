"""Phase 4 Stage 3: freeze the k=4 manifest BEFORE any label is generated (brief 6.2/6.3: extend to N-4 only
after the N-3 screen passes -- it did, see results/phase4/RESULTS.md). Smaller than the k=3 manifest since the
combinatorics are larger (C(41,4)=101,270) and the point is a truncation-order test, not exhaustive coverage."""
import json
from fdna import hik, v2data
from fdna.dataset import G

tr = v2data.load_vtable("data_v2", "train")
OP_IDS = [int(x) for x in tr["op_ids"][:20]]     # same 20 ops as the k=3 manifest, for direct comparability
K, N_SETS, SEED = 4, 25, 43

m = hik.sample_kset_manifest(G, k=K, n_sets=N_SETS, seed=SEED, stress_frac=0.3)
manifest = {"k": K, "seed": SEED, "n_sets": N_SETS, "op_ids": OP_IDS,
            "outage_sets": [list(s) for s in m.outage_sets], "strata": m.strata.tolist(),
            "n_possible_ksets": hik.n_possible_ksets(G.n_branch, K),
            "note": "Extends the k=3 screen per brief 6.3's staging rule (N-3 passed). Same 20 ops as k=3 manifest."}
json.dump(manifest, open("data_hik/manifest_k4_screen.json", "w"), indent=1)
print(manifest["n_sets"], "sets,", manifest["n_possible_ksets"], "possible")
