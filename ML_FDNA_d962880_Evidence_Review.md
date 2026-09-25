# ML_FDNA evidence review at d962880

25 September 2026. Audited commit: `d96288059fe0e8a83c62ab17b3cfcf8bc20bb336`; eight new commits after `7b7cb3e`.

## Verdict

The push contains useful implementation work, but it does not establish the sweeping negative conclusions in the final report. Several reported failures are contaminated by concrete bugs; the recurrent question was only partially addressed; and the prescribed brainstorming, alternative-solution work, and independent final review were not completed.

Keep g2 as the current working baseline. Withdraw the claims that recurrence has been decisively or twice independently ruled out, that the adaptive experiment is a faithful matched posterior-query comparison, and that the physical correction has passed a valid missed-severe rejection test. Correct and rerun those evaluations before increasing model size or buying more labels.

The earlier g2 confirmation is a separate result. The new 68-triple sensitivity summary supports its survival, with gains over g1 of approximately +0.254 and +0.273. The bugs below do not by themselves invalidate that earlier result.

### Scope of this audit

I inspected the pinned source, new registries, committed JSON results, tests, commit chronology, run journal, execution brief, and attached WISER proposal. I independently executed small reproductions of the critical implementation errors, using the actual source expressions/functions where possible. I also tested one proposed physical improvement on a small LP using the repository's solver and a separate island-balance calculation.

The original N-1/N-2/N-3 caches and trained model binaries are not published in this commit. I did not retrain the models, recompute the full benchmark metrics, or rerun the claimed full test suite. Consequently, the audit establishes that particular conclusions are unsupported and identifies likely failure mechanisms; it does not predict what the corrected benchmark scores will be.

## 1. Residual validation targets are misaligned — critical

In [the first validation pass][validation-first], the script samples control indices and builds `Xva` and `yva`. In [the residual validation pass][validation-second], it advances the same RNG and samples new indices, then computes:

`rva = yva - g2va`.

Those terms generally refer to different control states. The model is evaluated at the first state's features against a residual involving the second state's g2 value. This corrupts validation loss, early stopping, and checkpoint selection. It does not mean the training residuals themselves are misaligned; the defect is specifically in validation.

**Independent reproduction:** replaying seed 6006 produces mismatched control indices at **4,796 of 4,800 positions**. Some values may coincide across different states, so this is a count of wrong state alignment, not a claim that all 4,796 numerical targets differ.

The reported residual validation MSE, `7.474e-4`, should therefore not be interpreted as model quality. Its large magnitude is a debugging signal, not a reason to reject the method.

**Repair:** construct and retain `(canonical_key, X, y, g2)` together in one pass. Compute `r = y - g2` from those retained arrays. Assert both key equality and `r + g2 == y`. Save validation predictions and the selected checkpoint's metrics. Include the zero-residual predictor as the incumbent before training, so checkpoint selection explicitly compares learning against retaining g2.

## 2. Validation overlaps training; the claimed replication is too strong

Validation takes the first ten elements of `TRAIN_OUTAGES`, uses the first fifteen training operating points, and samples controls independently. The saved training order and seeds reproduce **612 of 4,800 validation records also appearing in training**: 12.75% exact overlap by `(outage, operating point, control index)`.

The other validation rows still share both outage sets and operating points with training. This measures interpolation over control states, while the reported test holds out outage combinations. It is a poor validation distribution for choosing a model intended to generalize to new combinations.

All 60 operating points appear in training and test. That is a legitimate *known-operating-point/new-outage* experiment if labeled precisely, but it does not fulfill the brief's requested holdout on both axes. P1 and P2 reuse the same fitted models and outage split. Two observation regimes are not two independent training replications. The published script has one global `torch.manual_seed(0)`, and the result artifact contains no multi-seed experiment.

**Repair:** freeze disjoint train/validation/test operating-point groups and outage groups, and explicitly report the intended generalization rectangles. Do not silently discard cross-rectangles: identify which are used for interpolation diagnostics and which for the primary test. Run a small charged-label learning curve and multiple training seeds. Resample the appropriate independent units, including training replicates when claiming stability to fitting.

## 3. The “commutative recurrent” model is neither the claimed invariant model nor a convincing recurrence test

The source defines [a feed-forward sum-pooling model][set-model] and a feed-forward residual MLP. There is no learned recurrent transition, GRU/LSTM, graph recurrence, or higher-cardinality transfer test. A sum can be written as an additive state accumulation, but testing that restricted architecture does not settle the broader structured recurrent hypothesis.

More seriously, the supposed invariant model constructs the tokens:

`(ya, Iab), (yb, Iac), (yc, Ibc)`.

Relabeling the same outage set changes which singleton is paired with which interaction. Pooling is invariant to reordering unchanged tokens; it does not fix the fact that these tokens change. The three pair-risk features are also passed to the readout in an ordered vector.

**Independent reproduction:** executing the committed `forward` method with a valid weight assignment changes its output from **0.10 to 0.11** when the same outages are relabeled. This is an architecture-level counterexample; it does not assert that the unpublished trained weights produce exactly those numbers.

There is another documented design concern. The registry says features are all roughly within `[0,1]` and need no rescaling, but [the reciprocal determinant feature][feature-scale] is capped at **10,000**, with no normalization. The actual distribution should be profiled. This is a plausible conditioning problem, not a quantified explanation of the observed loss without the original features and weights.

The feature contract is also narrower than the brief: no full demand/dispatch state or damaged-network representation is supplied; the registered new-cut flag is absent from the implemented 15-dimensional vector. “Every learner receives the same features” establishes matching between those learners, not that the features identify the missing higher-order response.

**Repair:** use separate symmetric singleton and pair encoders, with unordered endpoints for pair features and a symmetric physical-descriptor treatment. If testing recurrence, implement an actual shared state update on the damaged graph or an order-consistent residual update, retain g2 exactly, and evaluate the intended cardinality transfer. Test permutations with complete feature relabeling. Measure feature ranges, transform singularity indicators separately from finite risk values, and fit scalers on training data only.

Do not begin by making the network bigger. First fix targets, splits, invariance, conditioning, and information sufficiency. Search for identical/near-identical feature vectors with different higher-order labels to diagnose information loss. Then test whether physical features remove the ambiguity.

## 4. Adaptive predictions use the hidden state — critical leakage

The [prediction code][hidden-state] indexes `Lg[true_c_idx]` and `Ug[true_c_idx]` to decide the predicted class. `true_c_idx` is the unknown realized control state. The evaluator may use it for truth; the deployed policy cannot use it to predict under partial observation.

**Independent reproduction:** with the same observation, posterior, bounds, and query replies, changing only the hidden index changes the prediction. In a balanced two-state example, this expression achieves 100% classification accuracy even though an observation-only classifier can achieve at most 50%.

This is not merely an accounting issue. It invalidates the reported accuracy of the guided and random bound policies as practical partial-observation performance. It may affect their relative comparison in either direction; a corrected win is not guaranteed.

**Repair:** restrict prediction to the posterior and purchased-query information, for example a declared estimate from `[qL,qU]`, and use the hidden state only after predictions are frozen. Build the query interface that the brief required. Add a test that changes hidden truth while preserving observable inputs and purchased replies: the policy's next query and prediction must remain unchanged.

## 5. “Posterior Monte Carlo” is actually prior sampling

The [MC baseline][mc-source] calls `v2.sample_states`, whose own docstring explicitly says it samples the prior. It generates new observations into `obs_mc` and never uses them to condition the samples. Therefore it does not estimate the severe probability given the shared observation.

**Independent reproduction with the repository's communication model:** for an observation with all thirteen flags down, the prior probability of no commandable control is approximately **0.168706**, while the posterior probability is approximately **0.999998**. The implemented MC source estimates the former distribution.

**Repair:** draw control indices from the actual `pc` for the current snapshot, or draw hidden states from the conditional posterior. Reuse nested sample streams across budgets and matched random draws across appropriate estimators. Add a distribution test on a deliberately informative observation.

## 6. The adaptive experiment does not evaluate its registered policy or cost claim

Additional differences materially limit interpretation:

- The actual policy allocates a cap independently to every candidate and chooses an unresolved control whose g2 value is closest to the severity threshold. It does not allocate a shared solve budget around the shortlist boundary.
- It reports classification accuracy, without the requested fixed-budget shortlist recall/precision or the original screening endpoint.
- It changes the hidden state and observation with the **budget** because the budget is part of the RNG seed. The budget curve consequently changes the evaluation workload as well as the spending limit.
- MC stops increasing samples at eight while other arms reach 32; unique cached queries are not handled by a common ledger.
- The registry specifies 60 operating points and caps `{2,4,6,8,12,16}`; the script uses 20 points and `{2,4,8,16,32}`.
- The guided counts exclude lower-order preprocessing. The report calls this 223 lower-order queries per operating point, but each cached contingency table contains **1,024 scalar LP values**. The existing 41-singleton/182-pair tables contain 228,352 values per operating point. The actual 70-triple closure needs 40 singleton tables and 182 pair tables: **227,328** scalar values. A table lookup is not one scalar LP solve's cost.

The query-count difference within a particular executed cell is a descriptive observation about this heuristic. It does not establish the proposed method's cost frontier at matched screening quality. It also cannot justify a broad rejection of adaptive screening.

**Repair:** freeze one snapshot manifest and reuse it across budgets; compare against g2 without extra target queries; allocate a global solve budget; rank by posterior severe probability when that matches the decision; measure complete cold-start and explicitly defined warm-cache costs. Include an information-gain or bound-mass baseline that does not use g2. If bounds are too conservative, implement the residual/control-variate fallback already requested in the brief.

The latter fallback estimates `E[V-g2 | observation]` using valid posterior samples, or the analogous difference of severe indicators. It needs a fair comparison to plain posterior MC and complete preprocessing cost accounting. It was not attempted in this push.

## 7. The physical correction's missed-severe metric has the wrong sign

The evaluator stores `err = prediction - truth`, but [reconstructs the prediction][physical-sign] as `truth - err`. That equals `2*truth - prediction`; the correct reconstruction is `truth + err`.

**Independent reproduction:** true shedding 0.05 and predicted shedding 0 is an obvious missed severe event at threshold 0.01. The script reconstructs 0.10 and counts it as not missed.

The reported `1/1/1` and `12/12/12` missed-severe counts and the rejection gate based on them must be recomputed. The separately computed MAEs do not use this reconstruction and are not invalidated by this sign error.

The evaluator checks only full control and no control. Its header promises posterior-weighted evaluation and a richer stratification, but those computations are not implemented. The no-op result for the residual floor correction may still hold on this manifest; the evidence should come from direct floor-interaction checks, not the argument below.

## 8. “No new three-way cut means no third-order floor interaction” is false in general

The report treats `is_new_cut(S)==False` as ruling out third-order Möbius mass in the generator-less-island floor. That does not follow. Overlapping lower-order disconnection events can themselves create a third-order inclusion–exclusion term.

**Independent counterexample using the repository's own topology functions:** a triangle with a generator at one vertex and load at another has singleton floors 0, pair floors 1, 1, 0, and triple floor 1. Hence its third-order floor interaction is `1 - (1+1+0) = -1`, although `is_new_cut` is false because a proper subset already disconnects the graph. The residual correction returns the correct floor 1 instead of the second-order prediction 2.

**Repair:** compute `F(S) - Mobius_order_2[F](S)` directly when deciding whether the correction can change the prediction. Use this quantity, generator availability, connected-component structure, and compensation conditioning to design a targeted mechanism sample. Keep the natural-distribution benchmark separate and report prevalence or appropriate weights; an enriched sample does not establish population-wide improvement by itself.

The `(9,26,33)` example with zero generator-less floor also does not prove that congestion or a trip rule caused its shedding. An island containing generation can still have insufficient reachable upward capacity. Inspect island demand, generation bounds, and active constraints before naming the mechanism.

## 9. The “proper bootstrap over 70 triples” still bootstraps 4,200 rows

In [the linear uncertainty correction][linear-bootstrap], the script forms one squared-error difference for every operating-point/triple combination and calls `boot(diff_mse)` directly. There are **60 × 70 = 4,200 rows**. It does not aggregate into 70 triple means or pass cluster membership.

The output's description of a bootstrap over 70 triples is therefore false. This correction does not establish that the 4.68% MSE benefit is statistically confirmed. The point estimate remains; the uncertainty needs repair.

**Repair:** define the estimand first. For variation over the sampled operating points, aggregate by operating point. For variation over outage combinations, aggregate or resample whole triples. If claiming both, use an appropriate crossed resampling design, accounting for the shared fitted cross-validation models where relevant. Publish the underlying paired errors and group IDs so the calculation is inspectable.

## 10. Execution integrity and omitted work

The adaptive registry [does not parse as YAML][adaptive-registry]; an independent `yaml.safe_load` fails at its flow-style `policies` field. Its first published commit also contains the experiment and result, so the pushed history does not demonstrate the claimed prior committed registration for that stage. The neural registry does have a preceding commit, followed by a syntax-fix commit.

The [run journal][journal] explicitly records that the full query interface, fresh creative/literature stage, extension, and independent reviewer pass were not completed. Existing Phase 4 idea cards were substituted for the new brainstorming assignment. The control-variate fallback, matched label-budget curve, actual recurrent transition, and required training diagnostics were also not delivered.

The first documented stretch starts at 02:44 UTC and stops at about 03:25 UTC, roughly 41 minutes later; the recurrent stage was committed the following morning. This is not evidence of eight hours of continuous execution. An eight-hour limit does not require idle padding, but the journal supplies no completed blocker investigation explaining why substantial authorized work remained unfinished.

The neural checklist persists only target standard deviations, whether losses decreased between the first and last epochs, whether ridge beats a mean predictor, and whether train/test triple sets are disjoint. It does not establish tiny-batch fitting, correct feature scales, correct validation residuals, monotone loss decrease, or seed robustness. Passing generic tests cannot replace these scientific checks.

The committed monotonicity test uses `range(0, len(V), 50)`: **84 rows**, or **8,313,984** comparable state-pair checks. The report says a separate full 415,699,200-pair check ran once, but no executable full-run evidence is provided in this push. Treat that count as reported rather than independently reproduced. The bound direction and feasible-set monotonicity argument themselves are sound.

The push has eight new commits. The report's “99/100 commits this session” language should be replaced by an actual range/count; commit volume is not a scientific quality measure.

## What remains useful

| Work | Assessment |
|---|---|
| Original g2 confirmation and 68-triple sensitivity | Supported by the committed summaries and corrected evaluation structure; not numerically rerun in this audit. |
| Control-monotonicity argument | Correct for the frozen LP: increasing control expands feasible redispatch bounds. |
| Pointwise monotone lower/upper bounds | Correct mathematical directions and a useful implementation base; their deployment evaluation is faulty. |
| Physical residual decomposition | Useful implementation to retain; the missed-severe metric and general cut-based explanation need correction. |
| Learned-arm scores | Descriptive results for these fitted models and split; not a decisive verdict on recurrence or learning. |
| Claimed statistically confirmed L1 benefit | Point estimate retained; row-bootstrap claim requires replacement. |

Monotonicity is not a credible standalone novelty claim. The principle that relaxing a minimization problem's feasible set cannot raise its optimum is standard optimization. Its application and validation here can support a method, but the proof alone is not a new research contribution. See Boyd and Vandenberghe, *Convex Optimization*, especially perturbation and sensitivity: https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf. Deep Sets and its invariance conditions are also established: https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html.

## The better next sequence

### A. Repair the experiment before adding complexity

1. Preserve the old outputs and label them superseded where appropriate.
2. Fix validation-key alignment, the missed-severe sign, hidden-state access, posterior sampling, and the bootstrap units.
3. Make registries executable inputs rather than prose that can diverge from hardcoded scripts. Validate syntax and schema before a run.
4. Build the query ledger and information barrier. Save raw predictions, keys, per-group metrics, model hashes, configuration, and training histories. Large binaries can remain outside git if their identity and retrieval/regeneration path are recorded.
5. Rerun these cheap cached-data evaluations. Have an independent reviewer reproduce them before using them to decide a research direction.

Acceptance tests should include an intentionally missed severe case, two identical observed worlds with different hidden states, a strongly informative observation for posterior MC, full feature relabeling under outage permutation, and an assertion that every residual label's key matches its features and g2 value.

### B. Give learning a fair, bounded opportunity

Initialize the residual head at zero and retain the g2 checkpoint as a candidate. Use correct group validation and bounded tuning of scaling, learning rate, and regularization. Compare an unregularized/symmetric linear composition, residual shrinkage with zero allowed, GBM, a correct invariant set model, and one actual recurrent update.

Training currently samples control vectors uniformly. In the repository's prior, the all-zero control vector has probability about 0.1687; uniform sampling gives it probability 1/1,024. That is approximately a 173-fold weighting difference for this state. Uniform coverage is not inherently wrong, but it is a concrete distribution mismatch to investigate. Compare prior-weighted or mixed sampling and validate using the actual posterior-composed screening objective.

Provide the physical information needed to identify higher-order failures: operating-point demand and dispatch, damaged connectivity, available generation by island, and justified network-stress features. Give matched baselines the same information. Do not call the architecture incapable until optimization and information sufficiency have been checked. Do not assume the repaired model must win either.

### C. Test a stronger physical mechanism

Extend the generator-less floor to an available-generation floor. For each post-outage island I, let `D_I` be its load and `U_g(c)` the maximum reachable output allowed by the frozen LP, including current dispatch, ramp/control availability, and generator capacity. Then:

`F_capacity(S,c) = sum_I max(0, D_I - sum_{g in I} U_g(c)) / total_demand`.

For the local generator use its local corrective allowance; for a remote generator use its control-dependent range. This is a lower bound from island power balance, not a replacement for DC feasibility. Thermal constraints and the trip rule can make actual shed higher.

**Small experiment completed in this audit:** two islands each contain a generator. Their loads are 20 and 80 MW; base generation is 50 MW each; the second generator can reach only 60 MW. The generator-less floor is 0, the capacity floor is **20%**, and the repository's exact LP also gives **20%**. A separate island-balance calculation agrees. This shows a mechanism the existing floor misses; it does not establish a gain on the IEEE-30 benchmark or novelty in the literature.

A further candidate is a cut-capacity bound or network-flow relaxation that accounts for the maximum import into a region as well as its reachable generation. Derive and test its validity for this LP, charge its computation, and avoid adding overlapping regional deficits as if they were disjoint. Use such a bound to guide the residual learner or adaptive query policy only after its correctness is established.

### D. Implement the actual decision-aware acquisition idea

Spend queries on candidates whose posterior-severity intervals can change shortlist membership. Stop when the decision is resolved, rather than demanding exact classification at every posterior-supported control state for every candidate. Rank candidate/state queries by expected useful bound reduction per unit cost, compare with a generic policy without g2, and test the residual-MC fallback.

Use a fixed evaluation manifest across all budgets and include g2 alone, correctly conditioned MC, and complete cold/warm cost accounting. Do not make a cost-saving claim from the number of target lookups while omitting hundreds of thousands of preprocessing values. Test lazy lower-order evaluation on shared posterior samples instead of automatically materializing all 1,024 states. Measure the actual singleton/pair closure as the candidate pool grows; even lazy composition may lose to direct sampling for a small pool. A successful generic adaptive policy need not use g2 at all.

### E. Revisit claims and scope

Only after the repaired methods show a useful effect should the project spend another fresh confirmation block or attempt a large topology transfer. The attached WISER proposal targets impact propagation, critical assets, mitigation, and hardening ROI. The present static DC screening benchmark can be a component of that programme; it does not yet implement dynamic cascades, mission-layer effects, or ROI evaluation.

The next prompt should make the scientific interface tests above mandatory acceptance gates, with independent review before a branch is declared unsuccessful. My earlier brief was broad and left implementation choices to the PI; this run shows that a detailed checklist alone is insufficient when crucial checks remain self-reported prose.

## Reproducing the audit witnesses

The following script uses the pinned source and its published manifests; it does not require the private benchmark caches or model binaries. Save it outside the repository, run it from a checkout of `d962880`, and retain its JSON output. It exercises small counterexamples and indexing logic, not a full benchmark retraining. Dependencies are NumPy, SciPy, and PyYAML.

[validation-first]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_matched_recurrent_experiment.py#L88
[validation-second]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_matched_recurrent_experiment.py#L175
[set-model]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_matched_recurrent_experiment.py#L118
[feature-scale]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_matched_recurrent_experiment.py#L50
[hidden-state]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_adaptive_query_experiment.py#L100
[mc-source]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_adaptive_query_experiment.py#L104
[physical-sign]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p5_physical_correction_eval.py#L66
[linear-bootstrap]: https://github.com/akekulip/ML_FDNA/blob/d962880/scripts/p4_mobius_confirm_v2.py#L152
[adaptive-registry]: https://github.com/akekulip/ML_FDNA/blob/d962880/registry/phase5_adaptive_query.yaml#L13
[journal]: https://github.com/akekulip/ML_FDNA/blob/d962880/results/phase5/RUN_JOURNAL.md

```python
"""Read-only audit witnesses for ML_FDNA d962880; no original label cache required."""
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
from fdna.hik_diag import is_new_cut
from fdna.physical_correction import island_floor, g2_residual
from fdna import v2

out = {}

# Reproduce the exact two validation RNG passes and the saved training order.
training = json.loads((REPO / "results/phase5/matched_recurrent_training.json").read_text())
manifest = json.loads((REPO / "data_hik/manifest_k3_confirm.json").read_text())
train_outages = training["split"]["train"]
ops = manifest["op_ids"]
trng = np.random.default_rng(5005)
train_indices = {(tuple(o), op): trng.choice(1024, 128, replace=False)
                 for o in train_outages for op in ops}
vrng = np.random.default_rng(6006)
original = [vrng.choice(1024, 32, replace=False) for _ in range(150)]
resampled = [vrng.choice(1024, 32, replace=False) for _ in range(150)]
keys = [(tuple(o), op) for o in train_outages[:10] for op in ops[:15]]
overlap = sum(len(np.intersect1d(original[i], train_indices[key])) for i, key in enumerate(keys))
out["validation_alignment"] = {
    "rows": 4800,
    "mismatched_control_indices": int((np.array(original) != np.array(resampled)).sum()),
    "rows_also_present_in_training": overlap,
}

# Execute the committed forward method with a permissible weight assignment.
# One hidden unit is relu(singleton - interaction); all others are zero.
tree = ast.parse((REPO / "scripts/p5_matched_recurrent_experiment.py").read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSetsPairAware")
forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
def encode(x):
    z = np.zeros((*x.shape[:-1], 64))
    z[..., 0] = np.maximum(x[..., 0] - x[..., 1], 0)
    return z
fake_torch = SimpleNamespace(stack=lambda x, dim: np.stack(x, axis=dim),
                             cat=lambda x, dim: np.concatenate(x, axis=dim))
ns = {"torch": fake_torch}
exec(compile(ast.Module(body=[forward], type_ignores=[]), "committed_forward", "exec"), ns)
model = SimpleNamespace(elem_enc=encode, readout=lambda z: z[:, :1])
x = np.array([[.02, .08, .05, .01, .06, -.02, 0., 1., 1., 1., 0., 0., 0., 0., 0.]])
# Relabel a<->b: singles swap, ab stays, ac<->bc, corresponding pair risks swap.
permutation = [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14]
p1 = float(ns["forward"](model, x)[0])
p2 = float(ns["forward"](model, x[:, permutation])[0])
out["claimed_set_invariance"] = {"original_output": p1, "relabelled_output": p2,
                                  "invariant": bool(np.isclose(p1, p2))}

# Execute the actual hidden-state-dependent prediction expression.
aq = ast.parse((REPO / "scripts/p5_adaptive_query_experiment.py").read_text())
assign = next(n for n in ast.walk(aq) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == "pred_g" for t in n.targets))
expr = compile(ast.Expression(body=assign.value), "committed_prediction", "eval")
fixed = {"Lg": np.array([.02, 0.]), "Ug": np.array([.02, 0.]),
         "qLg": .5, "qUg": .5, "spec": SimpleNamespace(SEVERE=.01)}
predictions = [bool(eval(expr, {**fixed, "true_c_idx": i})) for i in range(2)]
out["hidden_state_leak"] = {"predictions_at_identical_observation": predictions,
    "accuracy_using_hidden_state": 1.0,
    "best_nonprivileged_accuracy_for_this_balanced_two_state_example": .5}

# The implemented MC source samples the actual world's prior, not this posterior.
w = v2.build_world()
obs = np.zeros((1, v2.N_FLAG), np.int8)
pc = v2.posterior_over_controls(w, v2.posterior(w, obs, .3))[0]
none = (w.CV == 0).all(1)
prior = np.bincount(w.cidx, weights=np.exp(w.logprior), minlength=len(w.CV))
out["prior_vs_posterior"] = {"observation": "all 13 flags observed down",
    "prior_probability_no_control": float(prior[none].sum()),
    "posterior_probability_no_control": float(pc[none].sum())}

# Missed-severe sign: the stored error is prediction minus truth.
y, prediction, tau = .05, 0., .01
error = prediction - y
wrong = y - error
out["physical_metric_sign"] = {"truth": y, "actual_prediction": prediction,
    "reconstructed_prediction_in_script": wrong,
    "correct_missed_severe": bool(y > tau and prediction <= tau),
    "script_missed_severe": bool(y > tau and wrong <= tau)}

# A triangle is a counterexample to the claimed necessary 'new cut' condition.
g = Grid(n_bus=3, frm=np.array([0, 0, 2]), to=np.array([1, 2, 1]), x=np.ones(3),
         gen_bus=np.array([0]), gen_pmax=np.array([10.]), load=np.array([0., 1., 0.]))
S = (0, 1, 2)
F = lambda s: island_floor(g, g.load, s)
ys = {a: F((a,)) for a in S}
yp = {p: F(p) for p in itertools.combinations(S, 2)}
g2f = sum(yp.values()) - sum(ys.values())
out["no_new_cut_does_not_imply_zero_third_order_floor"] = {
    "is_new_cut": bool(is_new_cut(g, S)), "F_triple": F(S), "g2_of_F": g2f,
    "third_order_floor_interaction": F(S)-g2f,
    "corrected_value": g2_residual(ys, yp, g, g.load, S)}

# Test the proposed stronger floor on an actual small LP with a generator in each island.
from fdna.lp import OperatingPoint, Params, ScenarioLP
from scipy.optimize import linprog
g4 = Grid(n_bus=4, frm=np.array([0, 1, 2, 3, 0]), to=np.array([1, 2, 3, 0, 2]),
          x=np.ones(5), gen_bus=np.array([0, 2]), gen_pmax=np.array([100., 100.]),
          load=np.array([0., 20., 0., 80.]))
op = OperatingPoint(demand=g4.load.copy(), p0=np.array([50., 50.]), rating=np.full(5, 1000.))
lp = ScenarioLP(g4, op, (1, 3, 4), Params(ramp_frac=.1, local_mw=100.))
c = np.array([1.])
generation_upper = np.minimum(g4.gen_pmax, op.p0 + np.r_[100., .1*g4.gen_pmax[1:]*c])
capacity_floor = sum(max(0., op.demand[lp.isl == i].sum()
                        - generation_upper[lp.isl[g4.gen_bus] == i].sum())
                     for i in range(lp.n_isl)) / op.demand.sum()
value = lp.y(c)
# Independent algebraic island-only program (all intra-island line limits are loose).
relaxed = linprog(np.array([1., 1.]), A_ub=-np.eye(2), b_ub=-np.array([0., 20.]),
                  bounds=[(0., 20.), (0., 80.)], method="highs")
out["capacity_floor_candidate_toy"] = {"n_islands": lp.n_isl,
    "generatorless_floor": lp.struct_mw / lp.total,
    "available_generation_floor": float(capacity_floor),
    "actual_repository_LP_value": value,
    "independent_island_balance_value": float(relaxed.fun / lp.total)}

try:
    yaml.safe_load((REPO / "registry/phase5_adaptive_query.yaml").read_text())
    yaml_status = "parses"
except yaml.YAMLError as e:
    yaml_status = str(e)
out["adaptive_registry_yaml"] = yaml_status
out["linear_bootstrap_units"] = {"actual_rows_passed_to_boot": len(ops)*len(manifest["outage_sets"]),
    "claimed_triple_clusters": len(manifest["outage_sets"])}
out["cold_cache_accounting"] = {"documented_lower_order_queries_per_op": 223,
    "scalar_LP_values_loaded_per_op": 223*1024, "undercount_factor": 1024}
out["committed_monotone_test"] = {"rows_checked_by_stride_50": len(range(0, 4200, 50)),
    "comparable_control_pairs_per_row": 10**5 - 4**5,
    "comparable_pairs_checked_by_test": len(range(0, 4200, 50))*(10**5 - 4**5),
    "reported_full_check": 4200*(10**5 - 4**5)}

(ROOT / "reproduced_findings.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
```
