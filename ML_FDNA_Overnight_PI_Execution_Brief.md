# ML_FDNA: autonomous overnight PI execution brief

Prepared from the repository at `7b7cb3ea33ff207177d2bb5679e77477960ccff8` on 25 September 2026. Repository: https://github.com/akekulip/ML_FDNA. This is an instruction to execute research and implementation, with independent agents, for up to eight hours from your actual start. It is not a request for another proposed plan.

## 1. Your mandate

Act as principal investigator, implementation lead, and coordinator of a research team. Inspect the current checkout, preserve the established evidence, build the most promising next methods, run fair experiments, investigate failures, and deliver a reproducible decision about what deserves to become the paper. Work autonomously within this brief. Do not end with “shall I continue?” while useful authorized work remains.

The scientific objective is to determine whether low-order outage composition can support reliable screening of larger outage combinations under uncertain communication, with fewer exact solves or better transfer than strong alternatives. Explain its failure modes and test ways to repair them. FDNA is not a required winner. A recurrent approach should become the main method if a properly matched experiment shows that it is stronger; it should receive an actual implementation and evaluation, not be rejected from intuition alone.

Persistence means testing explanations and alternative mechanisms. It does not mean promising a positive result, searching indefinitely, changing the simulator to favor a method, or treating the absence of a published solution as proof of impossibility. A rigorous negative result or mathematical obstruction is an acceptable outcome after adequate investigation.

Authorized work: read the repo and relevant primary literature; use the existing research tools and agents; edit project code and documentation; create a project environment; run bounded local experiments; create isolated branches/worktrees when needed; and make local commits using the configured identity. Preserve unrelated changes. Do not push, publish, purchase compute, contact people, alter access controls, delete prior evidence, or bypass permission/registry mechanisms without separate applicable authorization. Do not use permission-skipping flags. If an external approval really is required, state the action and the rule requiring it, leave a concrete artifact ready for review, and continue other unblocked work.

Start the clock now. Record actual UTC and America/New_York start/deadline, current branch/head, dirty files, available CPU/RAM/GPU/disk, and existing jobs. Default budget: up to eight hours; use less if the useful work is complete. Maintain a persistent run journal so context compaction or an agent failure does not restart completed work.

## 2. Establish the actual checkpoint before changing anything

Read applicable `AGENTS.md`/project instructions, `WORKING_NOTES.md`, `results/SUMMARY.md`, `results/LEDGER.md`, `results/phase4/IDEA_CARDS.md`, the frozen spec, and relevant code. In particular inspect:

- `registry/phase4_mobius_confirm.yaml`, `registry/slots.yaml`, `src/fdna/blocks.py`, and reserve locks.
- `scripts/p4_mobius_confirm.py`, `scripts/hik_label_confirm.py`, `scripts/hik_freeze_manifest_confirm.py`.
- `data_hik/manifest_k3_confirm.json`, `data_hik/manifest_amortization_new200.json`, other historical manifests, and `data_hik/confirm_run.json`.
- `results/phase4/mobius_confirm.json`, `src/fdna/hik.py`, the diagnostic utilities, `src/fdna/lp.py`, the communication/posterior implementation, and existing neural/baseline modules.

If the checkout is newer than `7b7cb3e`, inspect the intervening diff and incorporate completed repairs. Do not blindly overwrite newer work or repeat an experiment already completed. If local label caches are absent, check documented authorized locations and deterministic generation scripts. Estimate the smallest useful regeneration before declaring data unavailable; avoid regenerating the complete 18-million-solve run merely to repair a report.

The audited checkpoint contains these facts; verify against current files and correct this brief if newer evidence supersedes them:

| Item | Evidence at the audited checkpoint |
|---|---|
| Benchmark | IEEE-30, static DC corrective load shedding, synthetic communication uncertainty; this is not an AC or dynamic cascading-failure simulator. |
| Established method | Second-order Möbius composition of lower-order exact values, composed with the existing posterior. The positive result is not FDNA-specific. |
| Confirmation sample | Operating points 600–659, 70 unique triples, 182 required pairs, 41 single outages, 1,024 control vectors. |
| Generation | 18,001,920 LP solves; zero recorded infeasible solves; 1,948.6 seconds on 24 workers. |
| P1 R-precision | g2 0.874905; g1 0.624440; prior-only 0.733967. Gains +0.250465 and +0.140938. |
| P2 R-precision | g2 0.911688; g1 0.641721; prior-only 0.734908. Gains +0.269967 and +0.176781. |
| Uncertainty | Reported 90% lower bounds exceed 0.05 for all four comparisons; uncertainty is over operating points for the sampled outage pool. |
| Preregistration | Registry commit `37289f2`, manifest `af109f2`, label metadata `3dc21e4`, results `7b7cb3e`. Verify provenance; freeze future executable evaluation code before opening data too. |

Do not claim that this audit executed the current full test suite. Run the applicable suite yourself and report its actual result. Historical suite counts are not current validation.

## 3. First-hour corrections: preserve the result, repair its interpretation

These are specific audit findings to verify and address before generating new scientific claims:

1. **Two confirmation triples occurred in the earlier 200-triple screen:** `(0, 23, 34)` and `(9, 13, 40)`. All 60 operating points are fresh according to the committed provenance, but “entirely unseen outage sets” is not true for all 70. Preserve the original 70-triple output and lock. Add a clearly labeled 68-triple sensitivity analysis, keeping the original observation draws and tie keys for retained rows. Exclusion is based only on prior manifest membership, never performance. Recompute metrics and paired operating-point intervals; this is a sensitivity analysis, not a new independent confirmation.
2. **The learned-model correction is still too broad.** Under no control, g2 MSE is `7.690919697e-5`; the L1-trained linear model has `7.330709152e-5`, about 4.68% lower. OLS loses, and learned MAEs are worse. Preserve this metric-dependent point estimate. Do not conclude either “learning works” or “the gain was a small-sample artifact” without uncertainty and a relevant matched comparison. The existing linear comparison does not settle posterior-composed P1/P2 screening.
3. **The decision code checks only part of the registered claim.** Verify that executable scoring requires success against both baselines in both cells, at the stated margin, using the intended intersection-union rule. The current numbers appear to pass all four comparisons; fixing the rule still matters. Check failed/missing/NaN cases and report the bootstrap resolution, rather than interpreting a floor p-value as a precise tail probability.
4. **Retain reusable evidence.** Save per-row predictions/IDs where practical and per-operating-point metrics, not only pooled summaries. Include canonical keys, actual integer seeds, observation IDs, code/config/data hashes, and query provenance. Have an independent reviewer regenerate reported metrics from these saved artifacts without retraining.
5. **Fix cost attribution.** Of the 18,001,920 solves, 13,701,120 construct the N-1/N-2 tables and 4,300,800 provide N-3 evaluation truth. N-1 alone costs 2,519,040. For this 70-triple pool, direct target labeling is cheaper than constructing the full required lower-order tables. Training a learned comparator on N-3 labels also consumes information that fixed g2 does not train on. Record these distinctions explicitly.
6. **Do not reuse consumed confirmation data as fresh evidence.** Points 600–659 are now development data for any new method. Check the global usage of 660–699; their numerical location in a reserve range does not prove freshness. `open_block` returning an entire range and locks being per slot are not guarantees of sample independence. Never delete locks or create a new slot to relabel used samples as fresh.

Update current-facing summaries with dated corrections and links to preserved historical outputs. Do not silently rewrite the research history. Audit overlap against the union of all previously labeled target sets, including relevant subsets of higher-order manifests, for future claims of unseen outage combinations.

## 4. Agent team and ownership

Create the following specialist roles. Run independent reads and proposals concurrently, but serialize shared mutations and registry openings. Use worktrees or explicit file ownership. The PI integrates; no two agents edit the same core module simultaneously.

| Role | Responsibility and concrete output |
|---|---|
| Integrity/statistics reviewer | Independently audit provenance, overlap, splitting, metrics, decision rules, bootstrap units, and oracle access. Own audit reports; approve evidence quality, not the desired outcome. |
| Power-systems/oracle expert | Verify LP properties, control monotonicity, island floors, numerical tolerances, and representative failure mechanisms. Produce small counterexamples and tests. |
| Creative methods researcher | Independently propose at least three mechanisms from different fields, with equations, necessary information, cheapest falsification test, and likely failure. |
| Literature/adversarial reviewer | Search primary sources independently; challenge novelty and strongest baseline choices. Distinguish verified full-text findings, abstracts, and unverified leads. |
| Adaptive-method implementer | Build the audited query interface and the selected bounded-query method; own its scripts and focused tests. |
| Neural/recurrent implementer | Build one properly matched recurrent correction and its essential comparators; verify training and order behavior before judging its scientific value. |

If concurrency is limited, run roles in waves rather than abandoning them. A final independent verifier must review code/results it did not author. Each agent returns: files or commands inspected, findings with evidence, implementation/results, uncertainty, failed hypotheses, and the next discriminating test. “Could try X” alone is not a completed implementation assignment.

Require a short PI decision after each stage: what was learned, which hypotheses survived, what is blocked, and what will run next. Maintain a job ledger with PID, command, start, log, resource cap, expected runtime, and exit status. Do not leave duplicate or untracked shells consuming the night.

## 5. Resource and execution schedule

Use the actual machine inventory. If it matches the previous setup, start with at most 24 CPU workers, one GPU training process on the RTX 2070 8 GB, and one heavy labeling pool. Limit BLAS/OpenMP threads inside multiprocessing workers. Increase concurrency only after measuring throughput, memory, and oversubscription. No cloud spend is authorized.

Use this allocation as a practical default, adjusting after measured pilot runtimes:

- **0–1 h:** checkpoint audit, cached replay corrections, query-access contract, focused tests, independent idea generation.
- **1–2 h:** residual diagnostics, cheap objective-aligned score comparison, physical repair, shortlist at most two substantial methods.
- **2–5.5 h:** implement and run adaptive-query experiments; run matched neural work concurrently if resources permit.
- **5.5–7 h:** finish learning curves and one justified extension: larger outage cardinality or a bounded second-topology pilot. Repair verified failures.
- **7–8 h:** independent reproduction, corrections, report, local commits, clean job shutdown. Reserve this time instead of filling it with another large run.

Do not launch a large experiment without a measured pilot cost and a completion margin. Default to leaving verified-unused confirmation samples unopened tonight: build a credible method before spending another confirmation block. A genuinely ready survivor may receive a separately frozen confirmation only if its design, sample sufficiency, independence, executable code, and runtime are settled with adequate time remaining. Otherwise deliver exploratory evidence and a concrete confirmation design without opening its data.

## 6. Creative programme: derive and screen, then choose

Have the creative and domain agents generate at least six total idea cards spanning at least three fields. Do this independently before converging on the default options below. Each card must contain: mechanism, equation or executable sketch, what could be new, closest prior art, available versus missing information, a small decisive experiment, and a rejection condition. “Use an RNN/GNN/transformer” is not a mechanism.

Promising starting points, not mandatory winners:

1. **Reliability/cut decomposition:** remove exact generator-less island demand before truncating interactions, then restore the target outage's floor.
2. **Order theory and active inference:** use control monotonicity to bound posterior severe probabilities; purchase solves only where uncertainty changes shortlist membership.
3. **Variance reduction:** treat g2 as a control variate and estimate the higher-order residual rather than the entire posterior value or severe probability.
4. **Recurrent structured correction:** retain the exact low-order update and learn only missing higher-order structure from appropriately charged supervision or physical state.
5. **Sparse interaction discovery:** identify a small motif library of third-order failures or active constraints; test whether residuals concentrate and transfer rather than assuming sparsity.
6. **Stochastic-programming bounds:** partition posterior control states and use convexity/monotonicity to refine expected-value bounds, if those properties hold for the frozen LP.

Novelty is not established by combining names from different fields. Test whether the mechanism provides a capability or tradeoff that the strongest matched alternative lacks. A plain Möbius truncation, posterior expectation, control variate, or RNN is established mathematics. The plausible contribution is a demonstrated, well-scoped screening method and an explanation of when it works or fails.

Use primary sources and inspect the relevant method/assumption sections. Reading leads include:

- Cremer, *Polynomial Line Outage Distribution Factors for Estimating Expected Congestion and Security*, DOI `10.1109/TPWRS.2024.3463410`: https://research.tudelft.nl/en/publications/polynomial-line-outage-distribution-factors-for-estimating-expect/
- Nakiganda et al., contingency GNN transfer: https://arxiv.org/abs/2310.04213
- Roy and Hylviu, interdependent power/communication surrogate: https://arxiv.org/abs/2607.08918
- Kang et al., *Learning to Understand: Identifying Interactions via the Möbius Transform*: https://arxiv.org/abs/2402.02631
- Dehghani et al., adaptive network reliability analysis: https://arxiv.org/abs/2109.05360
- Forcier and Leclère, adaptive partitioning for stochastic LPs: https://leclere.github.io/files/papers/2022-GAPM.pdf
- Christianson et al., input-convex contingency screening: https://proceedings.mlr.press/v283/christianson25a.html
- Dwivedi and Tajer, graph recurrent fault-chain search: https://arxiv.org/abs/2303.08864

Confirm bibliographic details before using them in the paper. These are starting points, not an exhaustive novelty clearance. Search beyond the repository's keywords, including multiline contingencies, corrective recourse, active reliability, sparse pseudo-Boolean functions, multifidelity Monte Carlo, and sequential experimental design. Finding no relevant paper establishes neither impossibility nor priority.

## 7. Common experiment contract

Freeze a small new exploratory registry before model comparisons: development sample IDs, topology, operating-point/outage splits, posterior assumptions, training labels, candidate pools, budget units, primary endpoint, baselines, search budget, promotion gates, and deterministic seeds. Adaptation is allowed on development data; date and record it. Do not change a failed gate after seeing its result and call it preregistered success.

Keep the existing per-operating-point R-precision as a continuity endpoint. Preserve the exact definition and threshold; at this checkpoint severity is strictly `shed > 0.01`. R-precision requires the realized severe count for evaluation; do not give this unknown count to an online budget-selection policy. Add fixed shortlist-budget recall and precision, using predeclared budgets such as 10%, 20%, and 40%. Distinguish shortlist size from the number of LP queries purchased.

Evaluate both clearly named settings:

- **Original scenario-library ranking:** preserve the existing protocol, where each candidate scenario can have its own hidden communication state and observation.
- **Common-observation snapshot:** for an operating point, draw a communication state and observation once and share them across candidate outage sets. Rank the candidates under that common posterior. The evaluator may use the shared hidden state to generate realized truth; the method may not access it.

The second setting changes the estimand and is a new exploratory deployment analysis. Do not retrospectively substitute it for the old confirmation. Test posterior expected-shed scores against posterior severe-probability scores for g1, g2, and the true-table reference. Threshold-probability ranking is an essential objective-aligned comparator, not automatically a novel method.

Use canonical keys such as `(topology, spec_hash, op_id, sorted_outage_tuple, control_vector, observation_id)` and explicit stable integer seeds. Avoid Python's randomized `hash`, iteration-order seeds, and seeds derived from multiprocessing completion order. Preserve row identities across subsets. Verify reproducibility in fresh processes and changed worker completion order with a small meaningful case.

Split whole operating points and outage sets before fitting or selecting hyperparameters. A method using N-3 labels is a few-shot method, even if those labels already exist in a local cache. Show strict lower-order-only and paid higher-order supervision as distinct regimes. Report training, validation, and evaluation labels separately. Use paired operating-point uncertainty for continuity; if claiming generalization over outage sets too, add independently sampled manifests or an appropriate crossed uncertainty analysis. A bootstrap over 60 operating points alone does not establish robustness to every outage pool.

Implement a query wrapper that exposes only purchased values to a policy. Bulk truth tables may live in the evaluator, but policies must not inspect unpurchased entries. Log unique `(op, outage, control)` requests, cache hits, training purchases, and any higher-order solver states used. Policies may choose hypothetical control vectors to query, but may not read the realized hidden control vector or use it to choose queries. Keep oracle-only headroom arms clearly labeled and out of practical winner selection.

Report both cold-start and warm-cache cost. Charge construction of every required lower-order entry in cold start. State the exact cache already available in warm start, with amortization volume. Include posterior inference, preprocessing, fitting, wall time, peak memory, and query counts; raw solves/s alone is insufficient. Compute actual subset closure from the manifest. Do not revive the corrected claim that 477 cached pairs cover all 8,436 triples: they cover 2,698 in the audited example.

## 8. Diagnose and implement the smallest physical correction

Write a general composition utility with the base value explicit. For a triple `{a,b,c}`:

`g2 = V(ab) + V(ac) + V(bc) - V(a) - V(b) - V(c) + V(empty)`.

Prove and test why `V(empty)=0` in the current generator rather than assuming this on every future topology. For larger sets, use the general singleton/pair Möbius expansion with its correct base coefficient.

Compute `r3 = V(true triple, control) - g2` on existing tables. Report posterior-weighted as well as uniform errors. Stratify missed-severe cases and residual mass by generator-less islands, islands with generation, new three-line cuts, connected cases, compensation-matrix conditioning, control availability, and residual sign. Inspect a few LP solutions/duals to verify any claim about active thermal or ramp constraints. Correlation alone does not establish the mechanism.

Let `F(S)` be the exact demand fraction in generator-less islands after outage set `S`. Implement and compare:

- g2 as currently defined;
- simple clipping of g2 into `[F(S), 1]`;
- `F(S) + Mobius_order_2[V(T,c) - F(T)](S)`, with base terms handled correctly.

This tests whether known topological shedding should be removed before approximation. Do not double-count it. Give learned comparators the same descriptors. Predeclare an exploratory gate, for example a 20% relative reduction in missed-severe errors without material deterioration in R-precision, and report uncertainty rather than turning a noisy screen into a claim.

Construct at least one tiny hand-checkable case where all N-1/N-2 labels are harmless but an N-3 outage creates shedding. Verify with the independent LP formulation if available. Use this to explain what lower-order labels cannot identify. Outage inclusion is not generally monotone; keep Braess and negative-interaction cases in the data.

## 9. Main candidate: bounded, adaptive oracle queries

First have the power-systems expert prove or refute monotonicity in available control for the actual frozen LP. More available control must enlarge the relevant feasible set; inspect the constraints rather than trusting the name of a feature. This is a statement about control vectors, not outage-set inclusion.

If the property holds, for fixed operating point and outage set, queried values `V(c_j)` imply:

`L(c) = max(F(S), max over c_j >= c of V(c_j))`

`U(c) = min(1, min over c_j <= c of V(c_j))`.

The order is componentwise in actual control availability. Empty inner maxima/minima fall back to `F(S)` and `1`. Include justified numerical slack and test bound coverage against the existing full-grid truth; investigate every meaningful violation.

For posterior mass `p(c | observation)` and severe threshold `tau`, derive:

`qL = sum_c p(c) * 1[L(c) > tau]`

`qU = sum_c p(c) * 1[U(c) > tau]`.

Start with extreme-control queries when useful, then choose states whose order closures resolve substantial posterior uncertainty. Prioritize candidate outages whose score intervals overlap the current shortlist boundary. Use g2 or the physical correction to guide queries, while keeping exact bounds independent of the surrogate's accuracy. Query cost must include the lower-order values needed to construct that guidance.

Compare against direct posterior Monte Carlo, random/uniform acquisition, a generic monotone-bound policy without g2, and a cheap topology-based policy, under identical budgets and candidate information. Reuse existing baselines instead of rebuilding them unnecessarily. Include a labeled oracle selector only to quantify headroom.

If monotone bounds remain too loose, implement the bounded fallback:

`E[V | o] = E[g2 | o] + E[V - g2 | o]`.

Estimate the residual with posterior samples; use the analogous difference of threshold indicators for severe probability. Compare to plain Monte Carlo using matched random draws and budgets. Deterministic selection of apparently difficult states is not an unbiased sample; retain valid sampling probabilities or use a correctly weighted estimator. Account for any estimated control-variate coefficients and adaptive stopping when reporting uncertainty.

A separation certificate concerns posterior scores under the specified LP and posterior model. It does not guarantee realized recall, calibration under a wrong posterior, or real-grid safety. State the assumptions and which candidates remain unresolved.

Predeclare a practical exploratory promotion gate, for example at least 25% fewer total charged queries at screening quality within 0.01 of the strongest practical baseline, or a material quality gain at equal cost. Publish the whole predeclared budget curve. Choose the precise engineering gate before candidate results; it is distinct from the historical 0.05 confirmation SESOI.

## 10. Required matched recurrent experiment

Allocate a bounded, real implementation and training attempt to a recurrent method. Begin from the exact low-order update:

`g2(S + e) - g2(S) = m1(e) + sum over i in S of m2(i,e)`.

Retain this known part and learn only a correction or a physically useful latent state. Important identifiability check: the higher-order residual is identically zero on N-1/N-2. Training a residual head only on those scalar residual targets cannot teach arbitrary third-order interactions. Do not run that empty supervision experiment and call its failure evidence against recurrence.

Choose and explicitly label one viable regime:

- strict N-1/N-2 transfer using additional available physical/solver-state supervision, with that information also supplied to competitors; or
- a charged budget of N-3 labels to learn corrections, followed by held-out N-3 testing and, if feasible, N-4 transfer.

Give all compared learners the same operating-point, topology, control, singleton, pair, and physical-descriptor information. Compare fixed g2, the physical repair if useful, a properly symmetric linear/ridge model, tuned GBM, a compact pair-aware set model, and one compact recurrent correction. Reuse existing implementations and tuning infrastructure. Limit the validation search and architecture size to fit the night; do not interpret a deliberately weak baseline as a win. Compose every practical arm with the same P1/P2 posterior and evaluate the same screening endpoints, using matched posterior samples if exact integration is too expensive. An exact-control-only MSE gain cannot determine the lead method for the partial-observation task.

Static outage sets have no physical sequence order. Prefer a commutative set-state update or graph recurrence over the final damaged topology. If using a generic GRU over outage tokens, test all six orderings for triples and sampled orderings for larger sets; disclose and charge permutation averaging. Canonical sorting produces a convention, not a proof that the learned function respects outage symmetry. Evaluate open-loop higher-cardinality transfer without secretly supplying true higher-order prefix labels.

Before rejecting a neural result, verify target variation, feature alignment, gradient flow, loss scale, tiny-batch fitting, validation split correctness, and whether the chosen loss matches screening. Use a small predeclared training-label budget curve and more than one training seed where feasible. A bounded training smoke test alone is not a scientific comparison; complete held-out evaluation or document the exact verified obstacle.

Retain the RNN as the main route only if its benefit survives the strongest equal-information non-recurrent alternative at meaningful cost and uncertainty. If it ties a simpler model, prefer the simpler method and retain the result. If evidence is too noisy to exclude a useful effect, label it inconclusive rather than “does not work.”

## 11. Extensions that solve a real remaining uncertainty

Choose at most one substantial extension after the main experiments:

- **N-3 to N-4 transfer:** distinguish existing full-control-only N-4 files from a posterior-capable evaluation. Do not silently use missing control states as zeros. A complete 1,024-control table is needed for an exact full-posterior reference, but not for every valid experiment: realized shared-snapshot screening can use exact truth at sampled hidden states, and posterior integration can use a separately controlled Monte Carlo reference. Specify which estimand and uncertainty you are evaluating. Use this distinction to avoid declaring N-4 impossible solely because exhaustive labels are expensive.
- **Second topology:** build a small IEEE-118 adapter/pilot only if it tests a surviving method's mechanism or transfer. Remove hardcoded generator/control dimensions carefully; explicitly define the communication-to-control mapping and frozen stress calibration. Do not casually map a 5-generator control design onto a different grid and claim realistic transfer. A verified adapter and limited pilot are useful even if full validation cannot finish tonight.
- **Posterior/wiring robustness:** if a posterior-aware method wins, test a bounded, separately labeled misspecification case rather than claiming exact-posterior gains imply robustness. Preserve the original simulator and posterior assumptions for anchor comparisons.

Select the extension because it resolves the most consequential uncertainty, not because it is easiest to produce a positive number. If runtime is insufficient, preserve a working small reproducer and a costed next experiment.

## 12. Blocker protocol: investigate, repair, then decide

Before declaring an approach blocked or scientifically unsuccessful, write a blocker entry with: exact failing command or example; expected and observed behavior; versions/configuration; smallest reproduction; current classification; evidence; remedies attempted; and a specific next decision.

Classify the problem as environment/access, missing data, implementation, numerical, computational, statistical, or scientific/identifiability. Different classes require different remedies. “No prior paper does this” is not a blocker classification.

For material blockers:

1. Form at least two plausible explanations when appropriate and run the smallest test that distinguishes them.
2. Have an independent domain or implementation agent check the diagnosis.
3. Try at least two materially different bounded remedies when feasible. Repeating the same command or changing random seeds without a hypothesis is not a second remedy.
4. If literature has no answer, derive a toy model, invariant, bound, counterexample, alternative representation, or small executable construction. Work from the equations.
5. Stop a branch when a demonstrated obstruction, predeclared utility/resource gate, or verified unavailable external dependency warrants it. Record what evidence would reopen it. Continue the strongest remaining unblocked branch.

Examples: profile and fix oversubscription before calling labeling too slow; estimate residuals before requiring a full truth table; isolate connected components before blaming numerical conditioning; verify tiny-batch fitting before rejecting a network; distinguish zero supervision from poor optimization; inspect effective sample size before calling a small effect absent; and supply paid labels or physical information when lower-order values provably cannot identify a higher-order interaction.

Time-box ordinary blockers to roughly 30–60 minutes and unusually consequential ones to a documented maximum around 90 minutes. Do not consume the entire night on an infrastructure issue while cached-data analysis can proceed. Never repair a scientific failure by filtering difficult valid cases, moving thresholds after results, leaking labels, weakening comparators, or changing the generator to favor an arm.

## 13. Verification and evidence gates

Use focused tests for scientific invariants and meaningful integration risks, then the project's required suite. In particular verify composition coefficients/base terms; subset and worker-order reproducibility; control-bound direction and coverage; query accounting/no label leakage; grouping; severity threshold/ties; and complete claim decision logic. Avoid tests that simply copy the implementation's calculation as their expected answer.

Before promoting a result, the independent reviewer must reproduce metrics from saved predictions, inspect training/evaluation information access, check paired uncertainty and sample freshness, and challenge runtime/novelty language. Review a small deterministic end-to-end run. Fix identified errors and rerun the affected evidence; do not repeat unrelated expensive experiments without a concrete reason.

Use historical, exploratory, sensitivity, and confirmatory labels precisely. Newly developed methods on the old confirmation cache are exploratory. Keep the historical effect separate from the new method's evidence. Do not infer “RNN superiority,” “FDNA success,” “real cascade prediction,” “certified operational safety,” “universal speedup,” or “first in the literature” from an experiment that does not test that claim.

If a new confirmation is justified, freeze executable code, baselines, data ownership, exact fresh IDs, disjoint training draws, target-set exclusions, sample-size rationale, endpoints, margins, multiplicity rule, costs, and decision code before opening it. Respect global sample consumption. If these conditions cannot be met tonight, leave it unopened.

## 14. Required deliverables and completion behavior

Use a dated run directory and the repository's existing conventions where possible. Finish with actual artifacts, not a list of future ideas:

- A concise current status/report linking the audited head, code changes, completed runs, corrected claims, strongest result, and material limitations.
- A machine-readable claim ledger and experiment registry separating historical confirmation from every new exploratory or sensitivity result.
- Deterministic manifests, configs, run commands, dependency record, raw/per-group metrics, prediction artifacts as practical, hashes, seeds, and complete cost/query logs. Keep large local data in appropriate ignored storage; commit manifests and reproducible scripts.
- Error-stratification and budget-versus-quality tables/figures, plus the matched recurrent comparison and its information/training budget.
- At least six serious idea cards, reasons for promoting or rejecting them, primary-source novelty notes, and any original derivation/counterexample.
- A blocker-and-remedy log showing actual attempts and a ranked next experiment with measured cost if something remains unresolved.
- An independent review with findings and their resolution, the actual test results, clean local commits scoped to this work, and a final list of any remaining jobs. Stop only jobs belonging to this run when their useful work is finished; preserve unrelated processes.

The final morning message should answer plainly: What was built and run? What new evidence changed the decision? Does a recurrent method deserve the lead? Does composition actually save oracle work under which cache assumptions? What is potentially novel relative to the closest prior art? What failed after verification? What remains unknown?

Lead with the scientific outcome, give exact comparisons and uncertainty, and distinguish completed work from blocked work. Do not manufacture success and do not ask for permission to finish work already authorized here. Begin execution now.
