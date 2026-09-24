"""Phase 4 Stage 3 (brief 6.2/6.3): freeze the k=3 sampling manifest BEFORE any label is generated. Small,
exploratory-scale screen (not exhaustive: C(41,3)=10,660 possible triples; we sample 60 + reuse 20 held-out
train-table operating points, well under the 100-op / 191-outage x 1024-control scale of the existing N-1/N-2
value tables). Oracle-query accounting: 20 ops x 60 triples x 1024 controls = 1,228,800 LP solves for the primary
training/screen manifest below."""
import json
from fdna import hik, v2data
from fdna.dataset import G

tr = v2data.load_vtable("data_v2", "train")
OP_IDS = [int(x) for x in tr["op_ids"][:20]]                 # reuse the first 20 train-table operating points (already used for N-1/N-2 labels; no new op sampling needed)
K = 3
N_SETS = 60
SEED = 42

m = hik.sample_kset_manifest(G, k=K, n_sets=N_SETS, seed=SEED, stress_frac=0.3)
manifest = {
    "k": K, "seed": SEED, "n_sets": N_SETS, "op_ids": OP_IDS,
    "outage_sets": [list(s) for s in m.outage_sets], "strata": m.strata.tolist(),
    "n_possible_ksets": hik.n_possible_ksets(G.n_branch, K),
    "oracle_query_plan": {"n_op": len(OP_IDS), "n_outage_sets": N_SETS, "n_controls_per_row": 1024,
                           "total_lp_solves": len(OP_IDS) * N_SETS * 1024},
    "note": "op_ids are reused from data_v2 train table (already labelled for N-0/N-1); these are NEW (op, k=3-outage) labels only. Frozen before any label generated. Distinct from all existing PAIRS-based N-2 manifests."
}
json.dump(manifest, open("data_hik/manifest_k3_screen.json", "w"), indent=1)
print(manifest["oracle_query_plan"])
