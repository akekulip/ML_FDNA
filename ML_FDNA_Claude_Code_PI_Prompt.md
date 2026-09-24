# ML_FDNA: autonomous PI-led research programme

Copy this entire document into Claude Code while working in the ML_FDNA repository. This is an execution brief, not a request for another proposed plan.

Prepared from a read-only review of remote commit 943e09a81733c2538ec2289991f4b63a6f52584b on 2026-09-24. The review read source, tests, registries and reports; it did not rerun the repository's test suite or training experiments. The local checkout may contain newer work. Establish its actual state before proceeding.

## 1. Your mandate

Act as the principal investigator of ML_FDNA. Assemble and manage research, power-systems, machine-learning, statistics, literature, brainstorming, implementation, and independent review agents. Read the existing work, repair material scientific or implementation errors, select the strongest defensible hypothesis, implement the smallest decisive experiments, run them, analyze them, and leave a reproducible research result.

Philip's preference is to investigate FDNA through recurrence and generalization to larger failure combinations. Prefer that direction if evidence supports it. Do not manufacture a positive FDNA or RNN result. A well-supported negative finding, a stronger non-FDNA method, or a clearly justified pivot is an acceptable outcome. A list of ideas without executed experiments is not sufficient when execution is feasible.

Interpret the requested "N+k" provisionally as increasing k in N-k contingency screening: learning from smaller outage sets and predicting larger ones. Use conventional N-k terminology. If existing local notes explicitly mean future time n+k, distinguish that temporal task and assess it separately; do not silently conflate the two.

Continue autonomously through reversible local work, environment setup, code changes, tests, literature research, exploratory experiments, and eligible registered confirmation. Do not repeatedly ask whether to continue. Resolve ordinary choices as PI and record your reasoning. Ask only when a genuine external permission or access boundary prevents necessary work; meanwhile complete other independent work. Do not bypass approval controls, weaken scientific locks, purchase compute, contact people, publish, or push to a remote without an explicit applicable authorization.

Work in ~/Projects/ML_FDNA, or locate the actual checkout if that path is absent. Preserve user changes. Use the existing configured Git author identity; make coherent local commits with accurate messages, without invented authorship or added attribution trailers. Do not reset, discard, or overwrite unrelated work.

## 2. Establish the authoritative state

Before planning new experiments:

1. Read applicable AGENTS.md and CLAUDE.md instructions, if present. Inspect Git status, branch, HEAD, recent commits, and differences from the remote-tracking branch. Do not blindly pull over local changes.
2. Read results/SUMMARY.md, results/PHASE2.md, results/LEDGER.md, results/LITERATURE_NOTES.md, results/phase3/STEP1.md, results/phase3/STEP1_EXT.md, WORKING_NOTES.md, prereg/, configs/, registry/*.yaml and registry/locks/.
3. Inspect the implementation behind each relevant claim: lp.py, grid.py, features.py, baseline_data.py, v2.py, v2data.py, dataset.py, blocks.py, evalutil.py, rules.py, lodf.py; nn/layers.py, belief.py, inference.py, bayes.py, service.py, repair.py; value_table.py, fewshot.py, d_compose.py, p3_eqinfo.py, p3_repair.py and p3_step1_*.py; and their tests.
4. Inventory data, manifests, cached predictions, model checkpoints, current processes, GPU memory, CPUs, RAM, installed dependencies and actual test results. Reuse valid cached results; do not regenerate millions of labels simply because a report is stale.
5. Create a concise status audit: DONE AND VERIFIED HERE / REPORTED BUT NOT REPRODUCED HERE / IMPLEMENTED BUT NOT EVALUATED / STOPPED / PLANNED / INVALID OR NEEDS CORRECTION. Link every substantive item to its source and command or artifact.

At the reviewed remote head, the following distinctions are important:

- Phase 1 and Phase 2 exist. Registered FDNA arms were null or negative in the tested benchmark family. This does not prove that every FDNA implementation fails.
- The positive decomposition result was small, non-FDNA, and below the original 0.05 tier-1 bar. The two earlier test blocks shared training draws; do not call them fully independent training replications.
- Phase 3 diagnostics, A9 equal-information comparison, LODF code/tests, and FDNA repair code/tests are committed.
- STEP1 reports 143/820 islanding pairs and 58.9% of missed-severe mass in those pairs. Connected pairs retain 41.1% of misses. Reproduce only what is necessary for the next decision.
- Naive addition of single-outage flows was not accurate: reported p95 error was 42% of rating and overload-set disagreement was 24.1% in the sampled cases. The earlier "95% near-exact" statement was rejected.
- A9 already receives raw observations plus the same I2 per-generator probabilities. Its exploratory comparison did not close the decomposition gap. Do not reopen the obsolete claim that A9 still needs to be built.
- The repaired FDNA training screen was stopped before a completed comparison. It supplies neither positive nor negative evidence of repaired-FDNA screening performance.
- Step 3 has a registry, but no completed method evaluation at the reviewed head. No RNN or higher-k training pipeline was found.
- WORKING_NOTES.md contains stale remote-head/push/test-count statements. Refresh it from actual state, not memory.

The attached WISER proposal concerns multilayer dependency impact, disruption propagation, critical assets, hardening and ROI. If available locally, read it as motivation. The present DC corrective-shed benchmark does not establish dynamic cascading, business-process impact, or hardening ROI. Do not make those claims without the required dynamics, action models, probabilities and costs. Impact severity alone is not a calibrated risk estimate.

## 3. Team and ownership

Use actual available agents. If a named role is unavailable, instantiate a general agent with that role's instructions; do not claim an independent review that did not occur. Run read-only research in parallel. Limit concurrent heavy jobs according to measured resources. Use distinct file ownership or isolated worktrees for writers; the PI alone integrates shared registries, datasets and reports.

Required roles, which may be scheduled in waves:

| Role | Responsibilities and required output |
|---|---|
| PI/orchestrator | Own the scientific question, budget, decision ledger, integration and final claims. Select at most two implementation candidates after screening ideas. |
| Power-systems expert | Validate LP and island semantics, control availability, higher-k feasibility, flow approximations and any proposed dynamics. Identify the operational decision being improved. |
| FDNA/dependency expert | Distinguish canonical FDNA, adaptations, Boolean logic, probabilities and physical operability. Specify what the recurrent state and strength/criticality parameters mean. |
| ML/recurrent-learning researcher | Design the smallest meaningful recurrent and invariant models, extrapolation protocol, optimization checks and matched ablations. |
| Statistician/research-methods expert | Audit sampling, label budgets, endpoints, power/precision, multiplicity, uncertainty, registry amendments and holdout access. |
| Literature reviewer | Read closest primary papers beyond abstracts. Maintain verified citations, task/assumption comparisons, and novelty threats. |
| Brainstorming agent A: numerical methods/control | Independently propose mechanisms from active-set optimization, model reduction, system identification, uncertainty propagation and adaptive computation. |
| Brainstorming agent B: compositional learning/reliability | Independently propose mechanisms from reusable computation, set functions, reliability logic, experimental design and distribution shift. |
| Skeptical journal reviewer | Explain any apparent gain without invoking the claimed contribution; seek feature, label, simulator, split and compute confounds. |
| Implementation engineer and independent QA reviewer | Implement assigned work; a different agent verifies critical numerical claims and audits code/results. |

Have brainstorming agents work independently before they see each other's proposals. Require dissent to be recorded. Agent agreement is not experimental evidence. Prefer one targeted independent reproduction over repeated opinion-only reviews.

## 4. Correct the known issues before selecting a new mechanism

Preserve original outputs and history. Add explicit amendments/correction notes; do not silently rewrite an old preregistration as though it preceded the results.

### 4.1 Jensen calculation and attribution

scripts/p3_step1_jensen.py currently computes the posterior mean control and then selects its nearest discrete CV row. That is V(rounded E[C]), not V(E[C]). Consequently, the reported 0.0042/0.0027 values are not an exact Jensen-gap calculation.

For a bounded sample, evaluate ScenarioLP.y(c_bar) at the continuous posterior mean directly. Use the identical operating-point/outage sample across compared cells; compute the posterior expectation from the corresponding value table. Report per-observation gaps and solver tolerances, not only averages over observations. Verify the convexity argument for the actual clipped bounds by expressing each cap as separate affine constraints. Fix the script, report and registry explanation.

Even a correct Jensen gap does not structurally cap A8 or A9 at V(E[C]). A9 receives all 20 per-generator probabilities plus observations, and a generic regressor can learn an expected response. To test that mechanism, compare an explicit plug-in predictor against composition using the same posterior and value function, then separately compare learned summary-based predictors. Label any remaining causal explanation a hypothesis.

### 4.2 Nondegenerate metrics and fair information

The proposed Step-3 endpoint "R-precision on islanding-severe rows" becomes identically one if the evaluated subset contains only positives. Replace it prospectively with R-precision over all islanding rows, or subgroup severe recall under a common global screening budget. Define handling of zero-positive groups and report all-row performance and connected-case harm.

The Step-3 treatment gets an island partition, shed floor and recomputed flow information while the comparator gets only a topology category. That does not isolate the architecture. Give the generic comparator the same full derived descriptors and numerical features. Evaluate a cheaper approximate feature pipeline in a separate runtime/accuracy study.

### 4.3 FDNA repair semantics and cost

In repair.py, tau>0 uses noisy-OR while tau=0 switches gateway aggregation to max. The temperature ablation therefore changes two things for fractional inputs. Hold OR semantics fixed while changing temperature; add fractional-input tests, since binary tests cannot detect this.

Distinguish probability from operability. Noisy-OR has an independence interpretation for probabilities; max and FDNA performance aggregation need their own stated semantics. Shared gateways/site failures remain relevant beyond the shared control center.

Profile the joint repair implementation before relaunching it: its dense 16,384-by-1,024 conditional-control matrix is recomputed in forward passes. Do not blindly restart the stopped full screen. If needed, perform a small inference-quality test after correctness fixes, and cache only quantities that are actually fixed during optimization. Runtime exhaustion, insufficient statistical precision and a measured null result are different outcomes.

### 4.4 LODF claims and records

Keep the determinant's scope explicit: single-line bridges require separate treatment; the new-two-cut correspondence is not an unrestricted statement about all outage pairs. Recompute exact numerators and denominators for the reported top-decile trigger: "8.4% of connected pairs" is inconsistent with selecting at least the top 10% of those same pairs. Treat the repeated-pair correlation across operating points descriptively unless uncertainty accounts for the repeated structure.

Ordinary generalized LODFs, convexity of an LP value function, and generator-less-island shed floors are established mathematical structure. They can be ingredients or diagnostics, but are not automatically new contributions.

Deliver a compact corrected status report and commit these corrections before using them to choose the next experiment.

## 5. Literature and independent brainstorming

Search primary sources and follow references/citations, not only the phrase "FDNA neural network." Verify identity, version, publication status, assumptions, labels, splits, baselines and claims in the closest full texts. Record unverified papers as unverified. A keyword search with no match is not evidence of novelty.

Minimum prior-work starting points:

- Donnot et al., Fast Power System Security Analysis with Guided Dropout: https://arxiv.org/abs/1801.09870
- Nakiganda and Chatzivasileiadis, Graph Neural Networks for Fast Contingency Analysis of Power Systems: https://arxiv.org/html/2310.04213v3
- Dwivedi and Tajer, GRNN-based Real-time Fault Chain Prediction: https://arxiv.org/html/2303.08864v1 ; author manuscript https://par.nsf.gov/servlets/purl/10416896
- Christianson et al., Fast and Reliable N-k Contingency Screening with Input-Convex Neural Networks, L4DC 2025: https://proceedings.mlr.press/v283/christianson25a.html ; full text https://arxiv.org/html/2410.00796v1
- Chadaga, Wu and Modiano, Power Failure Cascade Prediction using Graph Neural Networks: https://arxiv.org/html/2404.16134v1
- Physics-Informed Graph Neural Jump ODEs for Cascading Failure Prediction in Power Grids, 2026 preprint: https://arxiv.org/html/2603.20838v1
- Deep Sets: https://papers.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Janossy Pooling: https://arxiv.org/abs/1811.01900
- Regularizing Towards Permutation Invariance in Recurrent Models: https://papers.nips.cc/paper_files/paper/2020/hash/d58f36f7679f85784d8b010ff248f898-Abstract.html
- Learnable Commutative Monoids for Graph Neural Networks: https://proceedings.mlr.press/v198/ong22a.html
- Roy and Hylviu, arXiv:2607.08918, and the FDNA/Cyber-FDNA sources already documented in the repository and proposal. Read their actual models and scope before relying on previous summaries.

Important novelty boundaries: N-1-to-N-2/N-3 transfer exists; graph-recurrent fault-chain prediction exists; permutation-regularized recurrence exists; convex-network contingency screening exists. The latter's specific guarantees must not be transferred to this different corrective-shed problem without satisfying their assumptions. Search temporal/cyclic FDNA, System Operational Dependency Analysis and other dependency propagation work; absence has not been established.

Produce at least six concrete idea cards across at least three fields, then shortlist at most two. Each card must contain: operational gap; mechanism; an equation or pseudocode; why current evidence supports trying it; closest prior work; equal-information baseline; smallest discriminating experiment; expected failure mode; runtime/label cost; pass/stop rule; and what contribution would remain if it succeeds.

Seed candidates to challenge rather than accept automatically:

1. A shared recurrent update that generalizes from smaller to larger outage sets.
2. Recurrent graph refinement or active-constraint discovery with a physically interpretable state, instead of arbitrary outage token order.
3. Exact island decomposition plus a residual learner, compared with a flat learner given identical descriptors.
4. Compensation/conditioning features plus learning of corrective-control active-set changes. First test whether a tuned tree already captures the gain.
5. Decision-aware posterior quadrature/compression preserving common-failure modes, compared with top-64 truncation and simple sampling.
6. Interaction/uncertainty-guided label acquisition under a counted LP budget.
7. Genuine temporal dependency filtering only if history contains operationally relevant information and a defensible independent temporal simulator can be specified.

Do not rename a familiar method and call it novel. Do not resurrect previously rejected ideas without new evidence. The PI selects by plausible effect, mechanism clarity, practical value, prior-art distance and experiment cost, and records why alternatives were deferred.

## 6. Preferred first screen: recurrent generalization to larger k

### 6.1 Define the task honestly

Distinguish three uses of recurrence:

- Outage-set processing: sequentially encode members of a static failure set. The final label is order independent.
- Iterative computation: repeatedly refine a state on a fixed post-outage graph. Iterations are computational steps.
- Temporal forecasting: track real changes in failures, dispatch, protection, communications and observations. This needs temporal transition labels absent from the current benchmark.

Start with one of the first two. Keep original initial dispatch fixed when using the existing static oracle. Do not introduce redispatch between arbitrary input tokens and still call the labels the same static task.

A possible set-processing model is:

    S_(r+1) = S_r union {e_r}
    h_(r+1) = F_theta(h_r, features(e_r), G minus S_(r+1), x, c)
    predicted_shed = readout_theta(h_k)

Here x is the operating point, c is available corrective control, r is a computational step, and k is outage cardinality. A graph refinement alternative repeatedly applies a tied cell to the final graph G minus S. Choose the smaller implementation that tests a stated mechanism.

Recurrence alone cannot infer arbitrary pair interactions from N-1 labels: functions with identical singleton values can differ arbitrarily on pairs. Explain the physical prior or the counted multi-outage supervision that makes the proposed extrapolation plausible.

### 6.2 Extend the data schema without invalidating old results

ScenarioLP accepts an arbitrary removed-branch tuple and post_outage_flows accepts arbitrary removed sets. The current dataset/value-table/v2data layouts, identifiers and dimensions remain specialized to two outage slots and IEEE-30.

Create a new versioned dataset path/schema with canonical sorted outage-set identifiers, unique branch IDs, explicit lengths/masks, topology ID, operating-point ID, control state and observation draw ID. Keep old V2 files and scripts reproducible. Remove pair-only assumptions from the new path and derive dimensions from the grid.

Verify against the old oracle on N-0/N-1/N-2 before generating higher-k labels. Test IDs, masks, duplicate handling, permutations, disconnected cases, structural shed floors and split disjointness. Failed or infeasible cases must be counted and explained; do not silently drop them.

Use sampled higher-k sets and controls. With 41 branches, there are 10,660 triples and 101,270 quadruples. Exhaustively crossing these with 100 operating points and 1,024 controls requires roughly 1.09 billion and 10.37 billion solves. Do not build those tables.

Freeze sampling before observing labels. Separate representative sampling from topology-based stress strata and report the actual test distribution. Count one oracle query per distinct operating-point/outage-set/control evaluation; a full control-response table is many queries. Count prefix labels, tuning labels, acquisition queries and privileged solver outputs separately. Permutations of one labelled set are augmentation, not new independent samples.

### 6.3 Staged experiments and fair baselines

First use exact controls to isolate electrical generalization. Freeze communication inference when it is reintroduced, so an electrical gain is not confused with changing inference.

- Reproduce the relevant N-1-to-N-2 anchor only as needed.
- Primary exploratory transfer: train on N-1 plus a fixed small budget of N-2 examples; assess unlabelled N-2 sets and exploratory N-3 sets on held-out operating points.
- Extend to N-4 only if the N-3 screen passes and severe/nonsevere prevalence leaves useful screening headroom.
- Keep a genuinely untouched higher-k test block. Once N-3 results guide design, that N-3 block is exploratory. Do not call it unseen confirmation again.
- Distinguish new operating points, new combinations, new cardinalities and new topologies. Sharing constituent branches/subsets across different full sets is expected in composition; exact target-set leakage must be excluded when claiming unseen-combination transfer.

Minimum staged comparator set:

1. Tuned LightGBM on the same final-set electrical, island, control and interaction descriptors.
2. An invariant DeepSets or graph/set model with the same information.
3. An ordinary GRU or shared-weight graph-recurrent model.
4. A simple convex/non-increasing value head where supported by the LP, with the published ICNN work treated as relevant prior art.
5. Existing D_I2/A9 anchors when partial observation is reintroduced; exact-control and decision-matched oracle references clearly labelled as privileged.
6. An FDNA recurrent variant only after the state semantics and the generic recurrent screen justify it. This does not reopen expensive static repair screens automatically.

Match label information, available features, reasonable tuning effort and training supervision. Report parameter counts, memory and compute; allow model-appropriate optimization. If recurrence sees each prefix's recomputed physical state or additional LP outputs, expose equivalent information to an appropriate baseline or separate that information advantage. Count feature construction, all recurrent iterations, posterior evaluations and permutation averaging in latency.

### 6.4 What would make the FDNA variant meaningful?

Specify the substate whose values represent operability or available service. FDNA is a dependency aggregation rule with parameters and topology, not automatically an interchangeable scalar activation like ReLU. An arbitrary signed embedding has no operability interpretation.

Identify mandatory versus redundant dependencies, intrinsic availability, strength, criticality, shared causes and the coupling to the electrical state. State which parameters are supplied, learned or uncertain. Keep generic latent features separate if useful. Do not make the label simulator use the same learned FDNA formula merely to create an advantage.

For a surviving recurrent model, compare identical cells/readouts/training with: generic aggregation; ordinary typed dependency logic; FDNA strength/criticality aggregation; and a justified wiring/parameter ablation. Attribute benefits separately to recurrence, graph information, logical repair, physical constraints and FDNA parameters. If generic recurrence wins equally, report a recurrent result rather than an FDNA result.

Do not force shed to increase after every added outage: the repository already contains violations of outage-set monotonicity. More available corrective control should not increase optimal shed at a fixed operating point and topology, but this is a different property. A sum of only nonnegative outage-damage increments would impose the wrong model.

### 6.5 Required recurrent checks

- For static sets, enumerate input permutations for k<=3 and sample them above that. Measure maximum/mean prediction variation, ranking variation and any averaging cost.
- No-op padding must leave predictions unchanged. Duplicate outages must be rejected or handled idempotently.
- Do not mistake canonical sorting or random permutation augmentation for a proof of invariance. Assess sensitivity to irrelevant branch-ID relabeling where the model claims graph equivariance/transfer.
- Test numerical stability and error growth beyond trained cardinalities/iteration counts.
- If training uses true intermediate states, evaluate free-running inference without them and count that supervision. Keep teacher-forced and deployed results separate.
- Report performance by k, island class, control availability and operating-point difficulty. Do not let almost-all-severe high-k samples manufacture a good score.

Gate: recurrence earns a larger run only if it improves a prespecified meaningful endpoint or label-efficiency/cost tradeoff over the strongest matched invariant/tree baseline, with stable exploratory evidence and no explanation from extra information or ordering artifacts. FDNA earns a claim only through the additional matched ablation. Failure of a generic recurrent model does not logically rule out a structured recurrent model: allow one bounded FDNA test if a specific representational diagnosis and known-answer examples justify it. A failed optimization run is not a scientific null; diagnose target scale, gradients and fit capacity before interpreting it.

## 7. Alternative direction and temporal gate

If recurrence does not survive, choose the best supported shortlisted alternative and execute its small decisive experiment; do not spend the remaining session cycling through many architectures. The choice and fallback order must be recorded before seeing their screening results.

For active acquisition, select with predictions or features available before buying the label. True measured pair synergy is an oracle comparator, not a deployable acquisition score. Separate offline learning from online budgeted screening. Compare random, uncertainty and an appropriate physics-based selector at matched total oracle cost.

For a convex value head, distinguish convexity in c at a fixed topology from any claim about topology changes. A max-affine non-increasing model can test the former. Solver dual supervision needs sign, clipping, normalization and finite-difference checks. Do not reuse a dual cut as a certified bound after changing the constraint matrix without proving feasibility.

If true temporal prediction appears stronger, the power-systems and literature agents must first establish an actual deployment question and relevant prior art. Independently specify state transitions, topology updates, protection, controller timing, stale observations and command delays; verify that history improves information beyond the latest snapshot. Compare an appropriate Bayesian/state-space filter and ordinary GRU with the FDNA version. Do not create arbitrary hidden memory just to favor an RNN. Use a separate simulator version and claim; do not relabel the static study as a temporal cascade study.

## 8. Statistical integrity, confirmation and practical relevance

Keep exploratory screens and confirmation separate. Register each experiment's question, inputs, arms, training/tuning/label budget, sampling, endpoints, effect threshold, seeds, stopping rule and intended claim before running that experiment. Amendments after exploratory results are allowed but must be dated and described honestly.

Use R-precision for continuity and add a fixed-budget recall/precision curve for operational screening. Freeze budgets before test evaluation and disclose severe prevalence. Keep expected shed and severe-case probability as separate scoring objectives. An exact posterior composed with an approximate value model and top-64 truncation is not a universal Bayes performance ceiling. Build an endpoint-appropriate oracle reference on a bounded tractable sample.

Use paired comparisons on the same scenarios; account for operating-point clusters and training variability, and for repeated outage sets if the claim generalizes across sets. Never treat millions of correlated rows as millions of independent replications. Report confidence intervals and practically meaningful effects, not only p-values. Keep existing tier definitions where comparable; any new endpoint/SESOI requires a prospective rationale, not a threshold chosen to admit the observed winner. Control multiplicity across the actual frozen confirmatory claim family.

Use exploratory variability to estimate precision and the needed number of independent training draws. Do not declare a study underpowered merely because it is slow, or a null because its interval is wide. Use predeclared futility/resource rules; keep stopped/failed/negative/equivalent outcomes distinct.

Inspect existing block usage. The 200-279 block is extensively inspected; 400-479 and 500-579 were used for H5. At the reviewed head, reserve 600-699 is unopened and disallowed by current slots. This execution brief authorizes new local study registrations and allocation of untouched seed blocks, including the unopened reserve if appropriate, within the stated compute budget. Add a transparent new registered allocation/slot before access and preserve the guard's semantics; this is a prospective study addition, not permission to bypass any lock or reuse inspected data. Never delete old locks, relabel inspected data fresh, or borrow an old hypothesis slot. If further untouched data are needed, allocate disjoint deterministic manifests and record them before generation. Use new training as well as test/observation draws for an independent replication.

Open a confirmatory block only after the corrected endpoint, fair comparator, code review, tuning, treatment selection, power/precision plan and novelty review are complete. Freeze the selected model and its competitors. Do not adapt after inspecting confirmation. An implementation failure consumes/discloses any actual data access; handle it by an explicit amendment and new block if needed.

For any claim extending beyond IEEE-30, require an appropriate second-topology experiment, preferably IEEE-118 if feasible. First remove hard-coded dimensions and document the communication/control mapping; do not silently attach the same five-generator assumptions to a larger grid. Distinguish training separately on a second topology from zero-shot cross-topology transfer. If a topology check cannot fit the budget, keep the claimed scope narrow and leave an exact executable follow-up.

Show end-to-end costs against the existing LP, including training-label generation and feature extraction. A millisecond small-grid LP may be a strong practical baseline. If a surrogate has value only across repeated state/contingency sweeps, state and measure that workload. Do not infer deployment value from neural inference time alone.

## 9. Execution discipline and bounded autonomy

Inspect actual hardware; historical notes suggest an RTX 2070 with 8 GB VRAM and 32 CPU cores, but do not assume these are current. Use the existing uv project environment and lockfile. Profile a smoke run before scheduling a sweep. Start with one GPU training process, bounded BLAS/LightGBM/LP worker counts, and enough RAM/CPU headroom to keep the machine usable. Parallel agents may read while one experiment uses the GPU.

Default first tranche: up to eight hours of local compute if no applicable project budget overrides it. Treat this as a resource ceiling, not a requirement to occupy eight hours. Complete the smallest decisive study sooner when possible. Allocate most compute to one selected hypothesis rather than many underpowered screens. Estimate time, memory and label counts before each escalation. If the full study exceeds the budget, finish a valid bounded experiment, save checkpoints and provide a runnable continuation; do not pretend the larger claim was tested.

Use durable logs, atomic result writes, per-run manifests, fixed output directories, explicit process IDs and checkpoints. Monitor training loss, validation loss, gradients where relevant, runtime, memory and output progress. Stop a broken or unjustified run, record why, fix the cause and resume only when the changed design is clear. Do not kill unrelated jobs.

Record a manifest per result with commit, configuration hash, dataset version/hash, seed split, oracle-query count, tuning budget, hardware and command. Validate the relevant tests and then the full project suite at integration gates. Do not claim tests passed from an old notes file.

The PI integrates local commits after each coherent stage: audit/corrections, frozen protocol, minimal implementation, exploratory findings, and eligible confirmation. Keep large data/checkpoints outside Git and preserve regeneration instructions. Do not push or publish unless explicitly authorized for this work.

Provide concise progress at natural milestones, state the leading uncertainty and next discriminating test, and avoid repeated permission questions. Maintain a handoff after each gate so compaction or interruption does not restart completed work.

## 10. Required outputs and final decision

Use the existing repository layout where suitable; suggested new files are:

- results/phase4/STATUS_AUDIT.md: actual state, corrections and provenance.
- results/phase4/LITERATURE_MATRIX.md: verified closest work and the remaining candidate gap.
- results/phase4/IDEA_CARDS.md: independent proposals, skeptic responses, ranking and selected/fallback hypotheses.
- registry/phase4_*.yaml: prospective protocols, budgets, decision rules and any amendments.
- Versioned higher-k data manifests and scripts, if that route is selected.
- Minimal models/baselines plus meaningful correctness and invariance tests.
- results/phase4/RESULTS.md and machine-readable metrics: effect sizes, intervals, costs, failures, strata and ablations.
- results/phase4/CLAIM_LEDGER.md: supported / contradicted / inconclusive / not tested, with evidence.
- Updated results/LEDGER.md, results/SUMMARY.md and WORKING_NOTES.md, preserving history.
- results/phase4/REPRODUCE.md: exact environment, commands, expected outputs, seeds and hashes.

The final report to Philip must answer plainly:

1. What was already done, what did you newly implement, and what actually ran?
2. Which previously reported statements changed after verification?
3. Does recurrence help, at which k and label budget, compared with which strongest baseline?
4. Does FDNA add anything beyond recurrence, dependency logic and physical features?
5. Is the result about static sets, iterative computation, or real temporal trajectories?
6. What is novel relative to the closest verified papers, and what remains unproven?
7. What operational gap is reduced, and what does the improvement cost?
8. What failed or was stopped, how much compute was used, and are any jobs still running?
9. Which local commits and artifacts contain the work? Was anything pushed?
10. Should the next action be a focused paper, one specific validation experiment, a justified pivot, or stopping this method branch?

Do not promise a venue or a positive result. Do not end after assigning agents or writing a plan. Begin with the state audit and concrete corrections, then execute the smallest scientifically defensible experiment that can decide whether the recurrent higher-k direction deserves to lead the project.
