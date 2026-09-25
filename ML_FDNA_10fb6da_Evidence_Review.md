# ML_FDNA — review of 10fb6da and the next experiment

Pinned commit: `10fb6daa061e8cf8b5c08c8cd41f00475494d975` (25 September 2026), compared with `842e6a9`.

**Verdict.** The remaining DeepSets invariance defect and the ineffective key/alignment checks have been repaired. The new residual diagnostics and adaptive-query outputs are real additions. Fixed composition remains the strongest reported arm. However, the adaptive stopping repair contains a threshold error, the control-variate headline uses a selected denominator, and the claims explaining the neural failures exceed the evidence. This should lead to one focused mechanism experiment with corrected evaluation, not another categorical rejection of learning or another repair-only programme.

**What I independently checked.** I read the 16 changed implementation, report and result files at the pinned commit. I executed the committed adaptive `resolve` function on a monotone counterexample, then changed only the erroneous threshold in a separate in-memory copy. I independently exercised the new topology-feature builder, shared scaler and actual DeepSets forward method under all six relabelings, using NumPy operations and an explicit valid weight assignment. I recomputed diagnostic denominators and validation-score differences from the committed JSON. These are not independent full-training results: this environment lacks Torch/pandapower and the bulk label caches/trained checkpoint files. The reported 86-test PASS and full retraining remain the repository author's account. The reproduction script is appended.

**1. Repairs that are now supported**

- `risk_ij` travels with its corresponding `(min(y_i,y_j), max(y_i,y_j), I_ij)` edge token. The readout receives only pooled edge features, the global floor and the generator control vector.
- Risk normalization now uses one shared training-only mean and standard deviation across edge slots. My complete feature-builder/scaler/forward check with unequal physical risks had a maximum spread of approximately 3.1e-15 across all six relabelings. This supports the architectural repair; it is not a measurement of the trained checkpoint's floating-point spread.
- Training and validation scenario keys no longer contain split-name prefixes. The overlap check is meaningful. Target/g2 alignment is checked by independently looking up sampled canonical keys in the raw tables rather than asserting an algebraic identity.
- The residual model starts at zero correction and evaluates that point before training. It records both validation MSE and a posterior-composed ranking diagnostic. Three residual seeds and a nontrivial tiny-batch fit check were added.
- Adaptive outputs now include distinct MC oracle evaluations, operating-point-clustered accuracy intervals and shortlist metrics. These additions are useful even though the stopping and metric contracts still need attention.
- The capacity-floor percentage and full/no-control missed-severe counts were corrected. The earlier bounds and physical argument are unaffected by this review.

**2. A real stopping-rule bug remains: severity units versus probability units**

Source: [adaptive script, lines 77–110](https://github.com/akekulip/ML_FDNA/blob/10fb6da/scripts/p5_adaptive_query_experiment_v2.py#L77).

The code currently stops when:

```python
if qL > 0.5 or qU <= spec.SEVERE:
    break
```

`spec.SEVERE` is 0.01, the threshold defining severe load shed. `qL` and `qU` bound the posterior probability that shed exceeds that threshold. The binary probability decision uses 0.5. Its negative certification condition is therefore `qU <= 0.5`, not `qU <= 0.01`.

The current condition is sufficient but unnecessarily restrictive: it does not produce an invalid certificate, but continues spending queries on some decisions that are already certified.

Independent witness using the production function:

- Four componentwise ordered control states have true values `[0.02, 0.02, 0, 0]` and posterior weights `[0, 0.2, 0.1, 0.7]`.
- After querying the extreme controls, the bounds are `qL=0`, `qU=0.3`. The binary decision is already negative.
- Current code uses four queries; changing only the negative probability threshold to 0.5 uses two. Both return the same negative decision.

**Fix and verification:** give the load-shed threshold and probability-decision threshold distinct parameter names. Test positive and negative certificates independently, including `qU` between 0.01 and 0.5 and the tie case at 0.5. Freeze random query streams by candidate so changing one candidate's stopping time does not alter subsequent candidates' random choices. Rerun the actual cost curves; do not infer the full-dataset saving from this toy witness.

Binary certification is also not ranking certification. An interval entirely below 0.5 can still straddle the shortlist boundary. A deployed shortlist policy needs a stopping rule tied to its ranking objective.

**3. The 97% control-variate result is promising, but narrower than reported**

Sources: [diagnostic implementation, lines 127–153](https://github.com/akekulip/ML_FDNA/blob/10fb6da/scripts/p5_adaptive_query_experiment_v2.py#L127), [result JSON](https://github.com/akekulip/ML_FDNA/blob/10fb6da/results/phase5/adaptive_query_experiment_v2.json).

The diagnostic appends a variance ratio only when `Var_p[h] > 1e-12`. Its denominator is not all 4,200 candidates:

| Cell | Included | Excluded | Favorable among included | Fraction among included |
|---|---:|---:|---:|---:|
| P1 | 2,292 | 1,908 | 2,225 | 97.08% |
| P2 | 2,053 | 2,147 | 1,995 | 97.17% |

The excluded cases are scientifically important. When plain MC has zero variance, a poor surrogate can introduce variance. For posterior weights `[0.5,0.5]`, true indicators `h=[0,0]` and proxy indicators `h0=[0,1]`, plain-MC variance is zero while `Var(h-h0)=0.25`. The committed diagnostic excludes this example. This establishes a missing failure category; it does not establish how often that category occurs in the real tables.

Likewise, a median ratio of zero does not by itself establish a favorable total error or query budget. The mean ratios of approximately 415,909 and 25,334 are unstable summaries of near-zero denominators; they do not by themselves establish an unfavorable total error either.

**The next diagnostic should save every candidate**, including both variances, covariance, posterior severe probability, proxy severe probability, disagreement mass and an exclusion/category reason if a ratio is undefined. Use centered weighted squared deviations to reduce cancellation near zero. Report:

- zero-variance ties, zero-variance cases harmed by the proxy, and nondegenerate cases;
- total/mean absolute variance and the ratio of summed variances, including all candidates, at a fixed number of posterior draws;
- tail losses and effects near the operational shortlist boundary;
- paired per-operating-point screening results and total oracle cost.

The suggested variance ratio is exact for a fixed control-variate estimator under its assumptions. It is an exploratory oracle diagnostic requiring the full truth table, not a deployable selector of which candidates should use the surrogate.

**4. DeepSets' worse score does not establish “information leakage”**

Source: [matched results](https://github.com/akekulip/ML_FDNA/blob/10fb6da/results/phase5/matched_recurrent_eval_v2.json).

| Arm | P1 R-precision | P2 R-precision |
|---|---:|---:|
| Fixed composition | 0.8921 | 0.9221 |
| Tuned GBM | 0.8895 | 0.9185 |
| Ridge | 0.8663 | 0.9065 |
| Fully repaired DeepSets | 0.8535 | 0.8840 |
| Residual model, selected seed 0 | 0.8380 | 0.8736 |

These results support retaining fixed composition as the lead. The current set model is worse than the previous partially invariant one. That does not establish that the earlier model used forbidden information: the pair risks are legitimate, deployable inputs, supplied to the other learned arms as well. The old architecture violated its claimed invariance; it was not thereby leaking hidden test labels or privileged control states.

Moving features through a bottleneck, changing input-layer dimensions and changing normalization alters the hypothesis class and optimization. Ordering may also provide a useful encoding on a fixed, canonically indexed topology. Whether that usefulness transfers to a different topology or branch renumbering is a separate question. The report's story about spurious shortcuts is a hypothesis, not an identified cause of the score change.

**A bounded ablation can answer this:** compare a canonical ordered baseline, the repaired edge-pooling model, and complete six-permutation averaging of the ordered baseline. Averaging the whole raw-input-to-prediction function over the permutation group is invariant by construction, even when its base model is not; charge its six forward passes. Include permutation-augmented training if justified by the initial ablation. Match data, validation protocol and several seeds. This separates a desired invariance property from one particular pooling architecture. It does not guarantee improved performance.

**5. Tiny-batch memorization does not prove absence of transferable residual signal**

Sources: [training diagnostic, lines 223–344](https://github.com/akekulip/ML_FDNA/blob/10fb6da/scripts/p5_matched_recurrent_experiment_v2.py#L223), [training JSON](https://github.com/akekulip/ML_FDNA/blob/10fb6da/results/phase5/matched_recurrent_training_v2.json).

The tiny-batch result is a useful sanity check: final MSE 4.87e-10 on 12 selected examples, with target variance 7.87e-6. It shows the network and gradient path can fit those examples using that optimizer configuration. It cannot rule out full-dataset optimization problems, unsuitable regularization, feature insufficiency or validation mismatch. The tiny fit also uses a different learning rate and no weight decay.

More specifically, checkpoint selection still does not match the final endpoint:

- `_diag_rprec` pools predictions from 12 `(outage, operating point)` blocks, with 50 observation draws each, and calculates one ranking. Final evaluation ranks within each operating point and then averages.
- The diagnostic uses P1 only. The final claim concerns both P1 and P2.
- The canonical seed's saved diagnostic takes only two ranking values: 0.8654545 and 0.8690909. The selected checkpoint improves this diagnostic by only 0.0036364. Identical best scores across three seeds on such a coarse diagnostic do not establish identical generalization.
- Only seed 0's residual checkpoint is evaluated in the final two-cell test table. The other seeds' matching validation scores are not three independent test replications.
- `lam` is trained jointly with the final linear layer. Its value can be absorbed into that layer's weights; a value such as 0.44 is not by itself evidence that an independently validated shrinkage factor was found.

**Repair the inference, not only the prose.** Rank and aggregate on validation using the same grouping and cells as the final endpoint. With a frozen residual predictor, evaluate a prespecified shrinkage grid including zero, rather than interpreting a jointly trained scale parameter as that experiment. Use all three saved seeds in the final exploratory evaluation, retaining per-operating-point scores and counting training variation in uncertainty.

Before enlarging the network, test whether its inputs contain the missing information. Many feature rows may have zero singleton/pair shed and zero island floor, while different operating points still have different higher-order outcomes. The current features could then be insufficient even when a physical correction exists. Measure exact or carefully specified near-collisions in input features versus residual labels; do not assume they exist. If supported, add relevant operating-point or congestion descriptors to every learned baseline at matched information and cost. An absence of improvement with the present features is not an impossibility result about all residual models.

No actual matched RNN/GRU experiment has been added in this push. The current learned experiment trains and tests on different N-3 combinations; it is not a test of extrapolating to larger cardinalities. The larger-cardinality recurrent question remains separate.

**6. The adaptive headline still depends on the chosen endpoint**

The committed bootstrap supports MC's classification-accuracy advantage over the current guided policy at each tested cap, conditional on the tested workload and protocol. That is a valid narrower result. It is not a blanket dominance result for operational screening.

For example, P1 at cap two has guided pooled precision@10% and precision@20% of 1.0, versus MC's 0.9619 and 0.9667, even though MC has higher classification accuracy. This is a direct result in the same JSON, not a hypothetical counterexample. A policy that is better at one threshold decision can be worse at selecting a small high-risk shortlist.

The shortlist metrics currently rank all 4,200 candidates together across 60 operating points. A controller screening 70 outages at one operating point cannot spend its unused slots on a different operating point. Pooled selection and per-operating-point selection are different tasks. My small metric witness produces pooled recall 0.0 and mean per-operating-point recall 0.5 from the same candidate scores. Keep pooled scores only as an explicitly separate offline selection task; implement per-operating-point shortlist metrics for the snapshot setting and bootstrap those paired differences.

Actual spending also differs at equal nominal caps. At P1 cap 16, guided uses about 6.72 target queries per candidate while MC uses about 9.57 distinct target queries. The guided arm additionally needs its lower-order cache. Report cold-start and warm-cache frontiers separately, including that cache's 228,352 scalar values per operating point. The zero-new-target-query g2 scorer and a clearly labeled posterior-oracle ceiling remain necessary reference arms.

**7. A concrete next run: make the control variate an actual candidate policy**

Do this after the small threshold and evaluation corrections; the existing cached labels are sufficient for an exploratory test. Do not stop at another favorable diagnostic percentage.

For `h=1[V>tau]`, `h0=1[g2>tau]`, posterior p and a fixed coefficient beta, test:

`p_hat_beta = beta * E_p[h0] + mean_j(h(c_j) - beta*h0(c_j)),  c_j ~ p`.

Beta=0 is plain posterior MC; beta=1 is the proposed full correction. For fixed beta chosen independently of the evaluation draws, this estimator is unbiased before clipping. Include both and a small validation-selected shrinkage option. If learning or estimating beta adaptively, separate the coefficient-selection samples from the evaluation samples, or derive a valid alternative; charge any pilot samples. A truth-table-optimal beta is an oracle diagnostic only.

Execution requirements:

1. Define the primary deployment task and freeze its metric before this comparison: per-operating-point shortlist quality under a real oracle-work budget, with classification accuracy secondary if screening is the goal.
2. Include g2 with zero new N-3 queries, posterior MC, corrected monotone bounds, fixed-beta control variates and a labeled exact-posterior ceiling. Use identical observations and stable candidate-specific randomness.
3. Save the full variance panel, including the excluded zero-variance cases, before interpreting a variance win. Test whether variance reduction improves ranking, not just mean estimation.
4. Cache repeated oracle evaluations while retaining the draw multiplicities/weights in the estimator. Deduplicating evaluations is valid; averaging distinct sampled states uniformly generally changes the posterior estimator.
5. Publish cold/warm total-cost curves, per-operating-point predictions, keys, query logs and all prespecified outcomes. Treat the repeatedly reused 600–659 block as development data for this new method.
6. If a method passes its exploratory gate, register the comparison and use genuinely unused operating points and a new outage set for confirmation. Verify reserve usage rather than assuming availability. Consider N-4 or a second topology only after the method and cost contract work.
7. If the control variate fails, inspect which of proxy mismatch, near-boundary ranking, cold-cache cost or posterior concentration causes the failure. Use that evidence to choose a selective correction or terminate this specific method. A failed mechanism is a result; an unimplemented diagnostic is not one.

For any further agent-led work, have the reviewer derive the statistical and dimensional contracts independently: probability thresholds, deployment grouping, which candidates enter each denominator, and what information a policy may use. Reproducing an author's script byte-for-byte checks reproducibility; it does not establish that those contracts answer the research question.

**Supported conclusion today:** fixed composition remains the best reported screener; the invariant implementation is now credible; the proposed control variate has a favorable conditional diagnostic worth testing. **Unsupported conclusions:** the earlier model benefited from forbidden information, transferable residual signal does not exist, or the control variate helps 97% of the complete candidate population.

**Executable reproduction appendix**

Run the following from a checkout pinned to 10fb6da with NumPy, SciPy and scikit-learn installed. It does not load the full label caches or train neural networks.

```python
"""Independent small checks for ML_FDNA 10fb6da, without Torch or private caches.

Run from a pinned checkout, or keep this script next to the audit's repo folder.
Forward execution uses NumPy and an explicit valid network weight assignment;
it is not a run of the saved trained Torch checkpoint.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import itertools
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT / "repo" if (ROOT / "repo/src/fdna").exists() else Path.cwd()
sys.path.insert(0, str(REPO / "src"))
from fdna import spec
from fdna.adaptive_query import bounds, posterior_mass_bounds, predict_from_bounds
from fdna.evalutil import recall_at
from fdna.grid import Grid
from fdna.lodf import ptdf_lodf
from fdna.nn import pairset_features as pf

out = {}

# Execute the production resolve function unchanged, on a monotone four-state truth.
tree = ast.parse((REPO / "scripts/p5_adaptive_query_experiment_v2.py").read_text())
resolve = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve")
CV = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([.02, .02, 0., 0.])
pc = np.array([0., .2, .1, .7])
ns = dict(np=np, CV=CV, spec=spec, bounds=bounds, posterior_mass_bounds=posterior_mass_bounds)
exec(compile(ast.Module(body=[resolve], type_ignores=[]), "committed_resolve", "exec"), ns)
original = ns["resolve"]("g2_guided_bounds", y, 0., y, pc, 4, np.random.default_rng(7))
L, U = bounds(CV, np.array([3, 0]), y[[3, 0]], 0.)
initial_bounds = posterior_mass_bounds(pc, L, U, spec.SEVERE)

class CorrectProbabilityThreshold(ast.NodeTransformer):
    def visit_Compare(self, node):
        node = self.generic_visit(node)
        if (isinstance(node.left, ast.Name) and node.left.id == "qU"
                and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Attribute)
                and node.comparators[0].attr == "SEVERE"):
            node.comparators[0] = ast.Constant(.5)
        return node

fixed = ast.fix_missing_locations(CorrectProbabilityThreshold().visit(copy.deepcopy(resolve)))
fixed_ns = ns.copy()
exec(compile(ast.Module(body=[fixed], type_ignores=[]), "threshold_only_counterfactual", "exec"), fixed_ns)
corrected = fixed_ns["resolve"]("g2_guided_bounds", y, 0., y, pc, 4, np.random.default_rng(7))
out["stopping_threshold"] = {
    "initial_probability_bounds_after_two_queries": initial_bounds,
    "current_query_count": original[0], "threshold_only_correction_query_count": corrected[0],
    "current_decision": predict_from_bounds(*original[1:]),
    "corrected_decision": predict_from_bounds(*corrected[1:]),
    "load_shed_threshold": spec.SEVERE, "probability_decision_threshold": .5,
}

# Exercise actual topology features, row builder and common scaler under every relabeling.
g = Grid(4, np.array([0, 0, 0, 1, 1, 2]), np.array([1, 2, 3, 2, 3, 3]),
         np.array([.7, 1.1, .8, 1.6, .9, 1.2]), np.array([0, 3]),
         np.array([100., 100.]), np.array([0., 20., 10., 60.]))
_, lodf = ptdf_lodf(g)
rng = np.random.default_rng(301)
control = rng.uniform(0, 1, (16, 5))
outage = (0, 1, 3)
y1 = {(7, b): rng.uniform(0, .1, 16) for b in outage}
y2 = {(7, (a, b)): y1[(7, a)] + y1[(7, b)] + rng.uniform(-.01, .03, 16)
      for a, b in itertools.combinations(outage, 2)}
truth = rng.uniform(0, .1, 16)  # synthetic labels: this is a wiring test, not a power-system experiment
raw = {}
for sigma in itertools.permutations(range(3)):
    o = tuple(outage[i] for i in sigma)
    phys = pf.physical_feats(g, lodf, g.load, o)
    raw[sigma] = pf.build_row(7, o, np.arange(16), y1=y1, y2=y2, phys=phys,
                             CV=control, y_true=truth)[0]
mu, std = pf.fit_shared_risk_scaler(raw[(0, 1, 2)])

class Tensor(np.ndarray):
    def unsqueeze(self, axis):
        return np.expand_dims(self, axis)

fake_torch = SimpleNamespace(Tensor=Tensor, minimum=np.minimum, maximum=np.maximum,
    stack=lambda a, dim: np.stack(a, axis=dim).view(Tensor),
    cat=lambda a, dim: np.concatenate(a, axis=dim).view(Tensor))
model_tree = ast.parse((REPO / "src/fdna/nn/pairset.py").read_text())
cls = next(n for n in model_tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepSetsPairAware")
forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
forward_ns = {"torch": fake_torch}
exec(compile(ast.Module(body=[forward], type_ignores=[]), "committed_forward", "exec"), forward_ns)
h = 8
W1, b1 = rng.normal(size=(4, h)), rng.normal(size=h)
W2, b2 = rng.normal(size=(h, h)), rng.normal(size=h)
W3, b3 = rng.normal(size=(h + 6, h)), rng.normal(size=h)
W4, b4 = rng.normal(size=(h, 1)), rng.normal(size=1)
model = SimpleNamespace(edge_enc=lambda x: np.maximum(x @ W1 + b1, 0.) @ W2 + b2,
    readout=lambda x: np.maximum(x @ W3 + b3, 0.) @ W4 + b4)
preds = []
for sigma, X in raw.items():
    X = X.copy()
    pf.apply_scaler(X, mu, std)
    preds.append(forward_ns["forward"](model, X.view(Tensor)))
out["fixed_invariance_pipeline"] = {
    "six_permutation_max_spread": float(np.ptp(np.stack(preds), axis=0).max()),
    "raw_risks": pf.physical_feats(g, lodf, g.load, outage)[1],
    "shared_scaler": bool(np.all(mu[1:] == mu[1]) and np.all(std[1:] == std[1])),
    "scope": "actual feature-builder/scaler/forward; explicit NumPy weights, not trained checkpoint",
}

# Read the exact selected denominator; do not silently label it all 4,200 candidates.
aq = json.loads((REPO / "results/phase5/adaptive_query_experiment_v2.json").read_text())
out["control_variate_denominators"] = {}
for cell in ("P1_v2b", "P2_v2c"):
    d = aq[cell]["control_variate_diagnostic"]
    n = d["n_candidates_measured"]
    favored = round(n * d["frac_candidates_favorable_ratio_lt_1"])
    out["control_variate_denominators"][cell] = dict(
        included=n, excluded=4200-n, favorable_included=favored,
        favorable_fraction_of_selected=favored/n,
        favorable_count_divided_by_total=favored/4200,
    )
pn = np.array([.5, .5]); htrue = np.array([0., 0.]); hproxy = np.array([0., 1.])
var = lambda x: float(pn @ (x*x) - (pn @ x)**2)
out["excluded_control_variate_harm_witness"] = dict(
    plain_MC_variance=var(htrue), corrected_MC_variance=var(htrue-hproxy),
    excluded_by_committed_filter=bool(var(htrue) <= 1e-12),
    scope="possible excluded harm; not evidence of its frequency in the real cache",
)

# Pooled rankings and per-operating-point rankings solve different selection problems.
ya = np.array([0.] * 8 + [1., 1.]); yb = np.array([1., 1.] + [0.] * 8)
pa = np.arange(100., 90., -1); pb = np.arange(10., 0., -1)
key = np.arange(10.)
macro = (recall_at(ya, pa, key, .2, thr=.5) + recall_at(yb, pb, key, .2, thr=.5)) / 2
pooled = recall_at(np.r_[ya, yb], np.r_[pa, pb], np.arange(20.), .2, thr=.5)
out["pooled_versus_per_op_metric_witness"] = dict(pooled_recall=pooled, mean_per_op_recall=macro)

tr = json.loads((REPO / "results/phase5/matched_recurrent_training_v2.json").read_text())
diag = tr["residual_diagnostics"]
hist = diag["canonical_seed_history"]
scores = sorted({r["rprec_val"] for r in hist if r["rprec_val"] is not None})
out["residual_selection_record"] = dict(
    zero_correction_val_score=hist[0]["rprec_val"],
    best_val_score=max(scores), distinct_recorded_validation_scores=scores,
    gain=max(scores)-hist[0]["rprec_val"], validation_blocks=12, draws_per_block=50,
    recorded_seed_scores=[d["best_rprec_val"] for d in diag["seeds"]],
    tiny_overfit=diag["tiny_batch_overfit"],
)

(ROOT / "reproduced_review.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
```
