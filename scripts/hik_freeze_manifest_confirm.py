"""Phase 4 confirmatory: freeze the fresh 70-triple manifest over 60 fresh reserve operating points (600-659),
BEFORE any label is generated. registry/phase4_mobius_confirm.yaml, block opened via src/fdna/blocks.open_block
(reserve__MOBIUS_CONF_P1/P2.lock)."""
import json
from fdna import hik
from fdna.blocks import block_seed
from fdna.dataset import G

OP_IDS = list(range(600, 660))   # 60 fresh ops; 660-699 left unused/reserved
SEED = block_seed("reserve")     # 613, the block's own deterministic seed (src/fdna/blocks.py convention)
K, N_SETS = 3, 70

m = hik.sample_kset_manifest(G, k=K, n_sets=N_SETS, seed=SEED, stress_frac=0.3)
manifest = {"k": K, "seed": SEED, "n_sets": N_SETS, "op_ids": OP_IDS,
            "outage_sets": [list(s) for s in m.outage_sets], "strata": m.strata.tolist(),
            "note": "Confirmatory manifest, fresh reserve block 600-659, seed=block_seed('reserve')=613. Distinct from the exploratory data_hik/manifest_k3_screen.json (seed 42, ops 0-19)."}
json.dump(manifest, open("data_hik/manifest_k3_confirm.json", "w"), indent=1)
print(f"{N_SETS} triples, {len(OP_IDS)} ops, seed {SEED}")
