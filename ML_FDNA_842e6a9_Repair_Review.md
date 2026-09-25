# ML_FDNA — independent review of the repair push

Reviewed commit: `842e6a9ee3089800c848c847fd046dfc010fc5d6`, compared with `d962880`. Review date: 25 September 2026.

**Verdict: substantial repairs, materially better evidence, but the blanket “all findings fixed / fully invariant / recurrent methods ruled out” conclusion remains too strong.** Fixed composition remains the best reported method on this experiment. The repaired set model is now close enough that the earlier sweeping negative verdict should stay withdrawn. One important architectural defect survives the repair and its acceptance tests.

I inspected the pinned source, registries, training/evaluation artifacts and repair write-ups. I independently executed small witnesses against the committed forward method and capacity-bound/LP implementations, checked split membership, parsed the active registries and recomputed reported arithmetic. I did not rerun the full training or 85-test suite: the public snapshot available here lacks the bulk label caches and trained binary artifacts, and this runtime lacks Torch and pandapower. Reported large-experiment scores below are the committed results, not independent retraining results. The forward witness uses NumPy substitutes and a valid explicit weight assignment; it establishes an architectural defect, not the size of that defect in the saved trained checkpoint.

**1. Repairs that deserve credit**

| Previous defect | Evidence in this push | Assessment |
|---|---|---|
| Residual validation sampled different control indices for features and baseline | `build_rows` now constructs key, X, y and g2 together from one index array | Actual alignment mechanism fixed; regression assertions remain weak |
| Training/validation overlap | Outage and operating-point training/validation memberships are disjoint | Fixed for the current split |
| Hidden-state leakage in adaptive prediction | Prediction uses posterior bounds `(qL,qU)`; hidden control is only used to score | Fixed in inspected production path |
| “Posterior” Monte Carlo sampled the prior | Samples now use `p=pc/pc.sum()` | Fixed |
| Different observations at different budgets | Observation is drawn once per operating point and cell, outside the budget loop | Fixed |
| Wrong absolute missed-severe counts | Evaluation now uses actual predictions directly | Corrected results: 47 misses at full control, 38 at no control |
| Wrong no-new-cut implication | Documentation withdraws the implication; a triangle regression example shows the distinction | Correct mathematical correction |
| 4,200 rows mislabeled as 70 bootstrap clusters | Script aggregates row losses to 70 triple means before bootstrapping | Fixed; significance correctly withdrawn |
| Missing reproducible exhaustive monotonicity check | Full script and count artifact now committed | Traceability repaired; I did not independently rerun the cached 415.7-million-pair check |
| Active registry parse errors | The four inspected current/repaired registry files parse | Fixed for those files; some older files are explicitly recorded as still invalid |

The stronger capacity bound was implemented rather than dismissed. My independent toy run returned generator-less floor 0.0, capacity floor 0.2 and exact repository LP shed 0.2. Its validity also follows from island balance and the actual corrective upper-generation bounds. This is useful engineering, even when its measured benefit is small.

**2. The repaired set model is still not fully permutation-invariant — high priority**

Source: [pairset.py](https://github.com/akekulip/ML_FDNA/blob/842e6a9/src/fdna/nn/pairset.py), [acceptance test](https://github.com/akekulip/ML_FDNA/blob/842e6a9/tests/test_deepsets_invariance.py), [training/scaling code](https://github.com/akekulip/ML_FDNA/blob/842e6a9/scripts/p5_matched_recurrent_experiment_v2.py).

The repair correctly associates each singleton pair with its own interaction, using tokens `(min(y_i,y_j), max(y_i,y_j), I_ij)` and sum pooling. But the final readout concatenates `X[:,6:]`, which still contains **ordered** `risk_ab, risk_ac, risk_bc`. Relabeling a and b swaps the ac and bc edges, so it must swap their risks too. A generic readout of that ordered vector is not invariant.

The committed dedicated test only permutes columns 0–5 and gives all three risks the same value. It therefore cannot detect this defect. The training script also permutes only the first six fields; its last two cyclic permutation mappings still disagree with the corrected mappings in the dedicated test.

Independent witness: set the edge encoder to zero and make the readout select the positive `risk_ac` coordinate. This is a valid assignment of the existing layers' weights. With risks `[1,2,3]`, the actual committed forward function returns 2 before relabeling and 3 after a↔b. All six incomplete permutations used by the present test return 2. The full six permutations return `[2,1,3,1,3,2]`.

There is a second part to the same repair: preprocessing standardizes the three risk slots using separate means and standard deviations. Even an edge-aware architecture would retain a slot-dependent transform if that preprocessing remained unchanged. Fit a common training-only scaler over pooled edge-risk values.

**Concrete fix:** put the corresponding normalized risk inside each edge token; pool the edge encodings; append only actual global features, such as F and the generator control vector, after pooling. Generate all six transformations from vertex/edge mappings rather than manually maintaining tuples. Test the complete raw-feature → preprocessing → model pipeline with unequal risks, nontrivial interactions, random parameters and the trained checkpoint. Keep tolerances tied to measured floating-point error. The claim that `minimum/maximum` themselves introduce approximately 1e-3 roundoff does not justify the current loose acceptance tolerance.

This defect does not prove that its repair will improve R-precision. It does prove that the stated architectural guarantee and its QA PASS are unsupported.

**3. The results changed enough to change the scientific interpretation**

Source: [corrected evaluation JSON](https://github.com/akekulip/ML_FDNA/blob/842e6a9/results/phase5/matched_recurrent_eval_v2.json).

| Method | P1 R-precision | P2 R-precision | Deficit versus fixed composition, percentage points |
|---|---:|---:|---:|
| Fixed composition | 0.8921 | 0.9221 | — |
| Tuned GBM | 0.8895 | 0.9185 | 0.26 / 0.36 |
| Repaired DeepSets | 0.8810 | 0.9152 | 1.11 / 0.69 |
| Ridge | 0.8663 | 0.9065 | 2.59 / 1.56 |
| Learned residual | 0.8377 | 0.8704 | 5.44 / 5.17 |

The set model's former deficit was approximately 11–12 points. Now it is about one point. The repairs changed architecture, feature scaling and data splitting together, so the change cannot be attributed quantitatively to the architecture fix alone without an ablation.

The reported 90% intervals remain below zero for these particular trained arms. They justify retaining fixed composition as the current lead. They do not establish that RNNs, structured learning, or learned corrections cannot work. This arm is a set network, not an RNN; this is one training seed and one held-out outage manifest, with repeated development on the same underlying operating-point pool. A cluster bootstrap over test operating points conditions on the fitted model and the sampled outage set; it does not establish robustness to training seeds or a new outage population.

The residual model deserves diagnosis before another capacity increase. It is randomly initialized, has no explicit zero-correction incumbent, and chooses checkpoints by pointwise validation MSE rather than posterior-composed validation screening performance. Those facts do not prove the reason for its failure, but the current code does not separate optimization error, objective mismatch and absent transferable signal.

Use `prediction = g2 + lambda*r_theta`, initialize the correction to zero, retain lambda=0 as a legitimate candidate and include the untrained g2 baseline in checkpoint selection. Tune lambda and checkpoints using a frozen validation protocol aligned with the deployed endpoint. This protects the validation choice, not an unconditional guarantee of test improvement. Record both value MSE and composed R-precision, complete loss curves, tiny-batch overfit results, and several predeclared seeds before rejecting the mechanism.

**4. The split is legitimate for new outages, but its description and guards need correction**

The saved split has 35 training, 10 validation and 25 test outage triples. Training and validation use disjoint operating points. However, all 60 operating points appear in test: 45 were used in training, 8 in validation and only 7 in neither. Therefore the claims of three disjoint groups “on both axes” are false. This remains a legitimate new-outage experiment if described that way; shared operating points do not themselves invalidate that estimand.

Report the three operating-point strata separately. Any claim about new-operating-point transfer needs a sufficiently sized untouched operating-point test, not a pooled result dominated by previously used points.

Two purported regression checks are ineffective:

- Keys include a `train` or `val` prefix before their intersection is checked. An identical physical scenario in both splits would still appear distinct. Compare canonical `(op_id, outage, control_idx)` keys, storing split labels separately.
- `allclose(y-g2,y-g2)` is tautological. Defining `r=y-g2` and then checking `r+g2=y` cannot detect a g2 array aligned to the wrong controls. Reconstruct selected targets and g2 values independently from the canonical keys, then compare.

The current construction appears correct; these are weaknesses in the safeguards, not evidence that the earlier misalignment persists.

**5. Adaptive querying is repaired, but it still does not implement the full screening question**

Sources: [script](https://github.com/akekulip/ML_FDNA/blob/842e6a9/scripts/p5_adaptive_query_experiment_v2.py), [results](https://github.com/akekulip/ML_FDNA/blob/842e6a9/results/phase5/adaptive_query_experiment_v2.json), [registry](https://github.com/akekulip/ML_FDNA/blob/842e6a9/registry/phase5_adaptive_query.yaml).

The reversal after repairing the baseline is real in the saved point estimates. With only two posterior samples, MC accuracy is 0.9317/0.9290. The best guided accuracy across all tested caps is 0.9290/0.9079, reached at cap 16, with approximately 7.56/7.48 queries actually used. No uncertainty analysis was saved for that comparison, so do not label it a new significance claim.

There are still mismatches between the registered task and the implementation:

- Registry requests cold-start total cost, recall/precision and shortlist quality at 10/20/40% budgets. The output reports classification accuracy and N-3 query counts; lower-order cost is noted separately, not incorporated in a practical cost-quality curve.
- Each candidate receives its own cap. There is no global allocation across a ranked shortlist.
- Acquisition selects the state whose g2 value is closest to the severity threshold, without scoring posterior mass or how much a query could change the decision.
- Stopping waits for every posterior-supported state to be classified. For the implemented binary decision, `qL>0.5` or `qU<=0.5` already certifies the decision. For ranking, stop when candidates' intervals establish the required order, not only after full state classification.
- MC counts repeated draws as fresh oracle solves, although an oracle cache can reuse repeated control-state values. Charge unique queries, and distinguish samples, cache reads, LP solves and wall time.

A revised benchmark needs the zero-new-query g2 scorer, posterior MC, a posterior-oracle ceiling, matched information access, realistic cold/warm cache accounting, per-operating-point outputs and a fixed shortlist metric. The expensive lower-order table is 228,352 scalar solves per operating point on this manifest; its presence must not be treated as free in a cold-start method claim.

**A concrete candidate worth screening:** use g2 as a posterior control variate, rather than only a threshold-distance query heuristic. Define `h(c)=1[V(c)>tau]` and `h0(c)=1[g2(c)>tau]`. For independent posterior draws `c_j`, estimate severity probability by

`p_hat = E_p[h0(c)] + (1/m) * sum_j (h(c_j)-h0(c_j))`.

For fixed h0 and a known posterior this is unbiased before any clipping; variance improves only if the residual has favorable variance, which must be measured. It can fail when approximation errors are poorly correlated or highly concentrated. Start with a cheap cached-data variance/error diagnostic before implementing adaptive allocation. Count all costs, including evaluating or constructing h0. If switching to non-posterior adaptive draws, add correct weighting or derive valid bounds; do not silently retain the unbiasedness claim. This is a candidate to test, not a claim of methodological novelty.

**6. Smaller corrections that matter to the record**

The corrected no-control linear comparison retains a 4.68% MSE point improvement, but the proper 70-triple bootstrap gives p≈0.364 and a 90% interval spanning zero. Withdrawing the significance claim was correct. Triple-only resampling also does not represent every source of joint operating-point/outage uncertainty.

The capacity floor is active above the generator-less floor on 120/4,200 no-control rows. The reported MAEs imply a **0.551%** reduction relative to the island-only correction, not 1.5%. The **1.543%** reduction is relative to uncorrected g2. Misses remain 47 at full control and 38 at no control; the later prose saying 38 in both states is wrong. “Capacity floor does not activate” on full-control rows means it adds nothing beyond the existing island floor, not that all physical capacity constraints or congestion are irrelevant.

The null physical-correction result concerns this sample. Targeted diagnostic manifests should cover nonzero third-order floor interaction, capacity-deficient islands and connected congestion separately, alongside an unchanged representative sample. Do not present an enriched diagnostic population as representative operational performance. The monotonicity proof is useful correctness infrastructure; a proof following directly from nested LP feasible sets is not by itself evidence of research novelty.

**7. Concrete continuation for Claude**

1. Reproduce the remaining full-feature invariance witness first. Repair edge-risk encoding and pooled normalization together; replace manually maintained permutation mappings and ineffective key/alignment checks. Correct split descriptions and arithmetic.
2. Keep current datasets explicitly exploratory. Save per-row predictions and canonical keys, per-operating-point metrics, model/config hashes and full training histories. Do not claim that repair reruns on reused data constitute new independent confirmation.
3. Diagnose the learned residual with a zero-correction incumbent, validated shrinkage, tiny-batch fit and endpoint-aligned checkpoint selection. Run a small prespecified multi-seed comparison. Inspect errors by residual size, threshold proximity and physical mechanism. Do not enlarge the model before locating the failure.
4. Repair the adaptive benchmark contract before inventing a more complex policy. Include the zero-query and posterior-MC baselines, query caching, total costs, shortlist metrics and decision-aware stopping. Screen posterior residual/control-variate variance and posterior-mass-weighted uncertainty reduction as two distinct mechanisms.
5. If the question remains about RNN N+k transfer, run an actual recurrent comparator using matched per-outage and pairwise information, appropriate supervision and charged label budgets. Test order sensitivity or permutation averaging, and charge the latter's cost. A DeepSets loss is not an RNN falsification. Residual supervision from N≤2 remains uninformative for g2's identically zero residual there.
6. Separate mechanism tests from performance claims. Enrich only the diagnostic manifest to expose known failure mechanisms. Promote a repaired method only after a predeclared effect threshold and fresh operating points/outage combinations; check the registry before reserving unused seeds. Have a reviewer who did not implement the change challenge complete input/output contracts, not just rerun author-written assertions.

Fixed composition should remain the lead while this work proceeds. The justified claim is that it is strongest in the current reported comparison. The unjustified claim is that a clean, decisive, independent experiment has already eliminated recurrent or structured alternatives.

**Reproduction appendix**

The following small script reproduces the architecture witness, saved split membership, overlap-guard counterexample, capacity-bound toy LP, active YAML parsing and the arithmetic above. Run from a checkout pinned to 842e6a9, with NumPy, SciPy and PyYAML installed. It does not run neural training or require the bulk label caches.

```python
"""Small independent witnesses for ML_FDNA 842e6a9; no cached labels or Torch required.

The forward-function witness executes the committed method with NumPy operations
and an explicitly realizable weight assignment. It does not load trained weights.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import itertools
import json
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo" if (ROOT / "repo/src/fdna").exists() else Path.cwd()
sys.path.insert(0, str(REPO / "src"))
from fdna.grid import Grid
from fdna.lp import OperatingPoint, Params, ScenarioLP
from fdna.physical_correction import capacity_floor, island_floor

out = {}
tree = ast.parse((REPO / "src/fdna/nn/pairset.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSetsPairAware")
forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
fake_torch = SimpleNamespace(
    Tensor=np.ndarray, stack=lambda a, dim: np.stack(a, axis=dim),
    cat=lambda a, dim: np.concatenate(a, axis=dim),
    minimum=np.minimum, maximum=np.maximum,
)
ns = {"torch": fake_torch}
exec(compile(ast.Module(body=[forward], type_ignores=[]), "committed_forward", "exec"), ns)
# Both edge-encoder linear layers can have all-zero weights. In the readout,
# select risk_ac into the first hidden unit, ReLU, then select that unit.
model = SimpleNamespace(
    edge_enc=lambda a: np.zeros((*a.shape[:-1], 64)),
    readout=lambda a: np.maximum(a[:, 66:67], 0.0),
)
x = np.array([[.02, .08, .05, .01, .06, -.02, 0., 1., 2., 3., 0., 0., 0., 0., 0.]])
edges = [(0, 1), (0, 2), (1, 2)]
edge_index = {e: i for i, e in enumerate(edges)}
full_outputs, partial_outputs = {}, {}
for p in itertools.permutations(range(3)):
    ep = [edge_index[tuple(sorted((p[a], p[b])))] for a, b in edges]
    perm = list(p) + [3 + i for i in ep] + [6] + [7 + i for i in ep] + list(range(10, 15))
    xp = x[:, perm]
    xp_partial = x.copy()
    xp_partial[:, :6] = xp[:, :6]
    full_outputs[str(p)] = float(ns["forward"](model, xp)[0])
    partial_outputs[str(p)] = float(ns["forward"](model, xp_partial)[0])
out["permutation_witness"] = {
    "partial_permutation_used_by_repo_test": partial_outputs,
    "complete_permutation_with_edge_risks": full_outputs,
    "partial_test_passes": len(set(partial_outputs.values())) == 1,
    "full_test_passes": len(set(full_outputs.values())) == 1,
    "scope": "architectural counterexample, not a measurement of the trained checkpoint",
}

training = json.loads((REPO / "results/phase5/matched_recurrent_training_v2.json").read_text())
splits = training["split"]
to_sets = lambda name: {tuple(x) for x in splits[name]}
out["actual_split"] = {
    "train_val_outage_overlap": len(to_sets("train_outages") & to_sets("val_outages")),
    "train_test_outage_overlap": len(to_sets("train_outages") & to_sets("test_outages")),
    "train_val_op_overlap": len(set(splits["train_ops"]) & set(splits["val_ops"])),
    "train_test_op_overlap": len(set(splits["train_ops"]) & set(splits["test_ops"])),
    "val_test_op_overlap": len(set(splits["val_ops"]) & set(splits["test_ops"])),
    "entirely_new_test_ops": len(set(splits["test_ops"]) - set(splits["train_ops"]) - set(splits["val_ops"])),
}
k = (600, (0, 1, 2), 42)
out["overlap_guard_counterexample"] = {
    "same_physical_row": k,
    "prefixed_intersection_count": len({("train", *k)} & {("val", *k)}),
    "canonical_intersection_count": len({k} & {k}),
}

g = Grid(n_bus=4, frm=np.array([0, 2, 1]), to=np.array([1, 3, 2]), x=np.ones(3),
         gen_bus=np.array([0, 3]), gen_pmax=np.array([100., 60.]), load=np.array([20., 0., 0., 80.]))
op = OperatingPoint(g.load.copy(), np.array([50., 50.]), np.full(3, 1000.))
params = Params(ramp_frac=1., local_mw=0.)
c, outage = np.ones(1), (2,)
out["new_capacity_function_toy"] = {
    "generatorless_floor": island_floor(g, op.demand, outage),
    "capacity_floor": capacity_floor(g, op, params, outage, c),
    "actual_ScenarioLP_shed": ScenarioLP(g, op, outage, params).y(c),
}

physical = json.loads((REPO / "results/phase5/capacity_floor_eval.json").read_text())["no_control"]
out["capacity_MAE_arithmetic"] = {
    "improvement_over_island_percent": 100 * (1 - physical["clip_capacity"]["mae"] / physical["clip_island"]["mae"]),
    "improvement_over_plain_percent": 100 * (1 - physical["clip_capacity"]["mae"] / physical["plain"]["mae"]),
}

out["yaml"] = {}
for name in ["phase3_step1.yaml", "phase3_step2.yaml", "phase5_adaptive_query.yaml", "phase5_matched_recurrent.yaml"]:
    try:
        yaml.safe_load((REPO / "registry" / name).read_text())
        out["yaml"][name] = "parses"
    except yaml.YAMLError as exc:
        out["yaml"][name] = str(exc)

aq = json.loads((REPO / "results/phase5/adaptive_query_experiment_v2.json").read_text())
out["adaptive_from_committed_results"] = {}
for cell in ("P1_v2b", "P2_v2c"):
    curves = aq[cell]
    out["adaptive_from_committed_results"][cell] = {
        "posterior_MC_at_2_accuracy": curves["2"]["mc_acc"],
        "best_guided_accuracy_any_budget": max(v["guided_acc"] for v in curves.values()),
        "best_guided_budget": max(curves, key=lambda b: curves[b]["guided_acc"]),
        "MC_strictly_better_than_both_at_every_equal_cap": all(v["mc_acc"] > max(v["guided_acc"], v["random_acc"]) for v in curves.values()),
    }

(ROOT / "reproduced_repair_review.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
```
