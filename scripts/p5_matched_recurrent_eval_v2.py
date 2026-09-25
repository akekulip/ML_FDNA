"""Phase 5 Stage R3: matched recurrent/learned-correction evaluation, v2 -- uses the CORRECTED models
(scripts/p5_matched_recurrent_experiment_v2.py: fixed validation alignment, disjoint train/val groups,
genuinely permutation-invariant DeepSets [repair round 2: full pipeline, including the risk-feature readout
tail], scaled features). Composes every arm with the SAME P1/P2 posterior, scores R-precision on the 25
held-out TEST triples (disjoint from both train and val on the outage axis), all 60 operating points (the
known-operating-point / new-outage-combination axis, as documented in the training split -- TEST intentionally
reuses train/val operating points, never claimed otherwise)."""
from pathlib import Path
import argparse, json, os, pickle, platform, subprocess, sys, time
import numpy as np
import torch

from fdna import v2
from fdna.dataset import G, RATING
from fdna.evalutil import rprec, sample_posterior_and_truth
from fdna.lodf import ptdf_lodf
from fdna.opgen import sample_op
from fdna.physical_correction import g2_clipped
from fdna.rules import boot
from fdna.nn.pairset import DeepSetsPairAware, ResidualNet, OrderedMLP, PermAveragedModel
from fdna.nn import pairset_features as pf
from fdna.nn.evaluation import seed_arms, sha256_file, summarize_seeded_family

torch.set_num_threads(4)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu",
                        help="Torch device for neural arms only; scoring/protocol stay unchanged.")
    parser.add_argument("--output", default="results/phase5/matched_recurrent_eval_v2.json",
                        help="Destination JSON artifact path.")
    return parser.parse_args()


ARGS = parse_args()
if ARGS.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("requested --device cuda but torch.cuda.is_available() is false")
DEVICE = torch.device(ARGS.device)
START_TIME = time.perf_counter()

n1 = np.load("data_hik/n1_confirm.npz"); n2 = np.load("data_hik/n2_confirm.npz"); n3 = np.load("data_hik/n3_confirm.npz")
CV = n1["CV"]
manifest = json.load(open("data_hik/manifest_k3_confirm.json"))
OP_IDS = manifest["op_ids"]
train_info = json.load(open("results/phase5/matched_recurrent_training_v2.json"))
TEST_OUTAGES = [tuple(o) for o in train_info["split"]["test_outages"]]
phys_mean = np.array(train_info["feature_scaling"]["phys_mean"]); phys_std = np.array(train_info["feature_scaling"]["phys_std"])

y1 = {(int(o), int(out[0])): V.astype(np.float64) for o, out, V in zip(n1["op_ids"], n1["outages"], n1["V"])}
y2 = {(int(o), (int(out[0]), int(out[1]))): V.astype(np.float64) for o, out, V in zip(n2["op_ids"], n2["outages"], n2["V"])}
order3 = sorted(range(len(n3["op_ids"])), key=lambda i: (int(n3["op_ids"][i]), tuple(int(x) for x in n3["outages"][i][:3])))
op_ids_sorted = n3["op_ids"][order3]; outages_sorted = [tuple(int(x) for x in n3["outages"][i][:3]) for i in order3]
V_sorted = n3["V"][order3]
row_by_outage_op = {(int(op_ids_sorted[i]), outages_sorted[i]): i for i in range(len(outages_sorted))}

demands = {op: sample_op(G, RATING, np.random.default_rng(int(op))).demand for op in OP_IDS}
_, LODF = ptdf_lodf(G)
PHYS = {(op, o): pf.physical_feats(G, LODF, demands[op], o) for op in OP_IDS for o in TEST_OUTAGES}


def build_features_all(op_id, outage):
    idx_all = np.arange(len(CV))
    y_full = V_sorted[row_by_outage_op[(op_id, outage)]]
    X, y, g2 = pf.build_row(op_id, outage, idx_all, y1=y1, y2=y2, phys=PHYS[(op_id, outage)], CV=CV, y_true=y_full)
    pf.apply_scaler(X, phys_mean, phys_std)
    return X, g2


with open("data_hik/matched_recurrent_models_v2.pkl", "rb") as f:
    models = pickle.load(f)
ridge, gbm = models["ridge"], models["gbm"]
DEEPSETS_SEEDS = (0, 1, 2)
ds_models = {}
for seed in DEEPSETS_SEEDS:
    path = "data_hik/deepsets_model_v2.pt" if seed == 0 else f"data_hik/deepsets_model_v2_seed{seed}.pt"
    m = DeepSetsPairAware(); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.to(DEVICE).eval()
    ds_models[seed] = m
# repair round 3 (external review finding 5): evaluate ALL 3 predeclared residual seeds, not only seed 0 --
# "3 seeds matching validation scores" was never 3 independent test replications when only 1 was ever scored
# on TEST. Also add residual_frozen_shrunk: the trained model's raw_correction (unscaled by its own jointly-
# trained `lam`) times a shrink value SELECTED ON VALIDATION via a prespecified grid (SHRINK_GRID in the
# training script) -- a properly separated shrinkage estimate, since `lam` itself is not independently
# identifiable (the final linear layer's own weights can absorb any rescaling of a jointly-trained scalar).
res_models = {}
for s in (0, 1, 2):
    path = "data_hik/residual_model_v2.pt" if s == 0 else f"data_hik/residual_model_v2_seed{s}.pt"
    m = ResidualNet(); m.load_state_dict(torch.load(path, map_location=DEVICE)); m.to(DEVICE).eval()
    res_models[s] = m
frozen_shrink = train_info["residual_diagnostics"]["frozen_shrinkage_selection"]["selected_shrink"]

# repair round 3 ablation (external review section 4): a canonical ORDERED (non-invariant) baseline, and a
# PERMUTATION-AVERAGED wrapper around it (exactly invariant by construction, 6 forward passes charged) -- both
# trained/evaluated across the same 3 seeds, comparing architectures under matched data/features/split/seeds
# without treating the observed gap as a causal estimate for pooling itself.
COLUMN_PERMS = [pf.column_perm_for_vertex_perm(sigma) for sigma in pf.ALL_VERTEX_PERMS]
ORDERED_SEEDS = (0, 1, 2)
ordered_models, ordered_pa_models = {}, {}
for s in ORDERED_SEEDS:
    mo = OrderedMLP(); mo.load_state_dict(torch.load(f"data_hik/ordered_model_v2_seed{s}.pt", map_location=DEVICE)); mo.to(DEVICE).eval()
    ordered_models[s] = mo
    ordered_pa_models[s] = PermAveragedModel(mo, COLUMN_PERMS).eval()

w = v2.build_world()
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
CELL_SEED = {"P1_v2b": 1, "P2_v2c": 2}
# g2_fixed doubles as the residual arm's zero-correction (lambda=0) baseline: ResidualNet's final layer is
# zero-initialized (src/fdna/nn/pairset.py), so "did the trained correction actually help vs doing nothing"
# is answered directly by comparing the residual row below to this row, not by a separate arm.
DEEPSETS_ARMS = seed_arms("deepsets", DEEPSETS_SEEDS)
ARMS = (["g2_fixed", "g2_clipped", "ridge", "gbm", "deepsets"] + DEEPSETS_ARMS +
        ["residual", "residual_seed1", "residual_seed2", "residual_frozen_shrunk"]
        + [f"ordered_seed{s}" for s in ORDERED_SEEDS]
        + [f"ordered_perm_avg_seed{s}" for s in ORDERED_SEEDS])

USED_FILES = [
    "data_hik/n1_confirm.npz",
    "data_hik/n2_confirm.npz",
    "data_hik/n3_confirm.npz",
    "data_hik/manifest_k3_confirm.json",
    "data_hik/matched_recurrent_models_v2.pkl",
    "data_hik/deepsets_model_v2.pt",
    "data_hik/deepsets_model_v2_seed1.pt",
    "data_hik/deepsets_model_v2_seed2.pt",
    "data_hik/residual_model_v2.pt",
    "data_hik/residual_model_v2_seed1.pt",
    "data_hik/residual_model_v2_seed2.pt",
    "data_hik/ordered_model_v2_seed0.pt",
    "data_hik/ordered_model_v2_seed1.pt",
    "data_hik/ordered_model_v2_seed2.pt",
    "results/phase5/matched_recurrent_training_v2.json",
    "scripts/p5_matched_recurrent_eval_v2.py",
    "src/fdna/nn/evaluation.py",
    "src/fdna/nn/pairset.py",
    "src/fdna/nn/pairset_features.py",
    "src/fdna/rules.py",
]


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def jsonable_cell(cell):
    q, share, coverage = cell
    if coverage is None:
        coverage_out = None
    else:
        coverage_out = np.asarray(coverage).tolist()
    return {"q": q, "s": share, "coverage": coverage_out}


def hardware_metadata(device):
    cuda_name = None
    cuda_capability = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
        cuda_capability = list(torch.cuda.get_device_capability(0))
    smi = None
    try:
        smi = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        smi = f"unavailable: {exc}"
    return {
        "selected_device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "cuda_device_name": cuda_name,
        "cuda_device_capability": cuda_capability,
        "nvidia_smi": smi,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "omp_num_threads_env": os.environ.get("OMP_NUM_THREADS"),
    }

results = {}
for cell, (q, s, cov) in CELLS.items():
    per_op = {op: {a: [] for a in ARMS} for op in OP_IDS}
    per_op_y = {op: [] for op in OP_IDS}
    per_op_key = {op: [] for op in OP_IDS}
    for outage in TEST_OUTAGES:
        for op_id in OP_IDS:
            i = row_by_outage_op[(op_id, outage)]
            y_true_grid = V_sorted[i].astype(np.float64)
            X, g2grid = build_features_all(op_id, outage)
            preds = {"g2_fixed": g2grid, "g2_clipped": np.array([g2_clipped(v, G, demands[op_id], outage) for v in g2grid])}
            preds["ridge"] = ridge.predict(X)
            preds["gbm"] = gbm.predict(X)
            with torch.no_grad():
                Xt = torch.tensor(X, device=DEVICE)
                for seed, model in ds_models.items():
                    preds[f"deepsets_seed{seed}"] = model(Xt).detach().cpu().numpy()
                preds["deepsets"] = preds["deepsets_seed0"]
                preds["residual"] = g2grid + res_models[0](Xt).detach().cpu().numpy()
                preds["residual_seed1"] = g2grid + res_models[1](Xt).detach().cpu().numpy()
                preds["residual_seed2"] = g2grid + res_models[2](Xt).detach().cpu().numpy()
                preds["residual_frozen_shrunk"] = g2grid + frozen_shrink * res_models[0].raw_correction(Xt).detach().cpu().numpy()
                # NOTE: loop variable named os_seed, NOT s -- the outer `for cell, (q, s, cov) in CELLS.items():`
                # already uses `s` for the coverage-share parameter; reusing `s` here silently clobbered it
                # (Python has no block scoping), corrupting sample_posterior_and_truth's `s` argument for
                # every row after the first, identically across every arm -- caught by a suspiciously identical,
                # suspiciously low R-precision across ALL arms on the first real rerun, not assumed correct.
                for os_seed in ORDERED_SEEDS:
                    preds[f"ordered_seed{os_seed}"] = ordered_models[os_seed](Xt).detach().cpu().numpy()
                    preds[f"ordered_perm_avg_seed{os_seed}"] = ordered_pa_models[os_seed](Xt).detach().cpu().numpy()

            # sample the shared posterior/observation ONCE per (op,outage), compose every arm against it --
            # not once per arm (which would redundantly redraw the same deterministic seed 6x)
            pc, true_c_idx, key = sample_posterior_and_truth(w, q, s, cov, op_id, outage, seed=CELL_SEED[cell])
            y_true = y_true_grid[true_c_idx]
            for arm in ARMS:
                per_op[op_id][arm].append((pc * preds[arm]).sum(1))
            per_op_y[op_id].append(y_true); per_op_key[op_id].append(key)

    per_op_rprec = {op: {} for op in OP_IDS}
    for op in OP_IDS:
        yt = np.concatenate(per_op_y[op]); key = np.concatenate(per_op_key[op])
        for arm in ARMS:
            pred = np.concatenate(per_op[op][arm])
            per_op_rprec[op][arm] = rprec(yt, pred, key)
    mean_rprec = {arm: float(np.mean([per_op_rprec[op][arm] for op in OP_IDS])) for arm in ARMS}
    diffs = {arm: np.array([per_op_rprec[op][arm] - per_op_rprec[op]["g2_fixed"] for op in OP_IDS]) for arm in ARMS if arm != "g2_fixed"}
    boots = {arm: boot(d) for arm, d in diffs.items()}
    results[cell] = {"per_op_rprec": per_op_rprec, "mean_rprec": mean_rprec, "boot_vs_g2_fixed": boots}
    results[cell]["ablation_summary"] = {
        "deepsets_mean": float(np.mean([mean_rprec[f"deepsets_seed{s}"] for s in DEEPSETS_SEEDS])),
        "deepsets_std": float(np.std([mean_rprec[f"deepsets_seed{s}"] for s in DEEPSETS_SEEDS])),
        "ordered_mean": float(np.mean([mean_rprec[f"ordered_seed{s}"] for s in ORDERED_SEEDS])),
        "ordered_std": float(np.std([mean_rprec[f"ordered_seed{s}"] for s in ORDERED_SEEDS])),
        "ordered_perm_avg_mean": float(np.mean([mean_rprec[f"ordered_perm_avg_seed{s}"] for s in ORDERED_SEEDS])),
        "ordered_perm_avg_std": float(np.std([mean_rprec[f"ordered_perm_avg_seed{s}"] for s in ORDERED_SEEDS])),
        "deepsets_legacy_alias_seed0": mean_rprec["deepsets"],
        "scope": "compares architectures under matched data/features/split/seeds; does NOT identify pooling "
                 "as a causal mechanism and does NOT establish whether ordering is a real transferable signal "
                 "on other topologies or branch renumberings (out of scope, external review section 4, repair "
                 "round 3).",
    }
    results[cell]["seed_family_summary"] = {
        "deepsets": summarize_seeded_family(
            "deepsets", DEEPSETS_SEEDS, per_op_rprec,
            references=("g2_fixed", "gbm", "ordered_perm_avg"),
        ),
        "ordered": summarize_seeded_family(
            "ordered", ORDERED_SEEDS, per_op_rprec,
            references=("g2_fixed", "gbm", "ordered_perm_avg"),
        ),
        "ordered_perm_avg": summarize_seeded_family(
            "ordered_perm_avg", ORDERED_SEEDS, per_op_rprec,
            references=("g2_fixed", "gbm"),
        ),
    }
    print(cell, json.dumps(results[cell], indent=1))
results["provenance"] = {
    "head": git_head(),
    "torch_threads": torch.get_num_threads(),
    "elapsed_s": time.perf_counter() - START_TIME,
    "command": sys.argv,
    "hardware": hardware_metadata(DEVICE),
    "eval_protocol": {
        "op_count": len(OP_IDS),
        "test_outage_count": len(TEST_OUTAGES),
        "posterior_draw_count": 50,
        "cells": {name: jsonable_cell(cell) for name, cell in CELLS.items()},
        "deep_sets_seeds": list(DEEPSETS_SEEDS),
        "ordered_seeds": list(ORDERED_SEEDS),
        "no_retrain": True,
        "fresh_holdout": False,
    },
    "feature_scaling": {"phys_mean": phys_mean.tolist(), "phys_std": phys_std.tolist()},
    "sha256": {path: sha256_file(path) for path in USED_FILES},
}
out_path = Path(ARGS.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
tmp_path.write_text(json.dumps(results, indent=1, allow_nan=False) + "\n")
tmp_path.replace(out_path)
