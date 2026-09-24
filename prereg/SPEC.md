# Frozen modeling specification (Milestone 1)

Frozen before any model comparison. Hash of `configs/spec.json` + this file is recorded in every result file
(`scripts/spec_hash.py`). Any change requires a new spec version and re-running everything.

**Target.** y = minimum load shed / total demand after corrective action, for one operating point, a set of removed
branches, and a set of commandable generators. DC model, single static snapshot. Not an AC or dynamic claim.

**Information at inference.** Demand vector, pre-contingency dispatch, removed branches, and the service-failure set
(component up/down states). Never the LP solution, shed, or redispatch.

**Corrective interval / ranges.** A remote generator's corrective range is +-ramp_frac*Pmax scaled by the commandable
fraction of its two units (shares 0.6/0.4). The local balancer (generator 0, must-run >= 80 MW) is always commandable
within +-local_mw. Justification: local range represents primary/spinning reserve that needs no remote command.

**Communication-to-control mapping (v1, binary).** Unit commandable iff CC up, its RTU up, and at least one parent
gateway up (redundant parents are alternatives). No bandwidth-to-MW scaling. Synthetic layer: 14 binary components
(CC, 3 gateways, 10 unit RTUs); generators 3 and 4 are dual-homed to GW1/GW2.

**Islands.** Islands without generation shed their whole demand (structural; stored separately). Generation may trip
only to rebalance an island's shed plus its initial surplus, so tripping is not free downward redispatch.

**Reference mechanism condition for M1.** Truth = binary reachability above ("independent mechanism with similar
dependency behavior" relative to FDNA). The FDNA-exact and mismatch/stale-parameter conditions are deferred to after
Milestone 1.

**Correctness checks (tests/).** N-0 needs no shed; enlarging the commandable set never increases minimum shed;
shed >= structural shed; agreement with an independent cvxpy formulation within 1e-5 MW; reachability logic.

**Calibration (development pilot, seeds 1000+, no model results used).** kappa=1.6, ramp_frac=0.3, local=40 MW chosen so
that ~19% of N-2 scenarios shed under full control and ~38% change with control state (target 15-40%).

**Splits.** Operating points 100/20/80. Train: N-0 and N-1 only, failure sets = empty, singletons, 50% of pairs (hash).
Val: 15% of pairs. Test: remaining pairs and all size-3/4 sets. Cells reported: {N-1, N-2} x {seen, unseen failure sets}.

**Primary endpoint (amended v1.1, before any model result).** Severe (y > 1%) recall at a verification budget equal to the
severe prevalence of the cell (R-precision), ranked within each test operating point, averaged over operating points;
paired cluster bootstrap over operating points. Amendment reason: the dataset shows ~30-36% severe prevalence in N-2 cells
(comm-state distribution includes frequent control-centre loss), so a fixed 10% budget caps recall at ~0.28. Secondaries:
recall at 10/20/40% budgets, AUPRC, Spearman, per-bin MAE. Hyper-parameters for tree baselines are tuned on validation
N-2 scenarios (disclosed deviation from strict N-0/N-1-only information).

## Amendments and disclosures (v1.2, 2026-09-23; written after Milestone 1 results were inspected)
These do not change any label; `spec_hash()` now covers `configs/spec.json` only, `doc_hash()` covers this prose.

1. **Smallest effect of interest = +0.05 R-precision** was proposed by the statistics review before any test result
   but was not in this file; recorded here after the fact. It is a convention, not derived from operating cost.
2. **Training / validation**: models train on N-0/N-1 labels; hyper-parameters and early stopping use N-1 and N-2
   validation labels (validation failure sets differ from train and test). Describe results as "N-1 training with
   N-2 validation".
3. **Evaluation**: test rows are subsampled by a hash of the scenario id (25%); ties in predictions are broken by a
   fixed model-independent hash key, so a constant predictor scores at chance (verified: zero-baseline R-precision equals
   prevalence within 0.003). Fixed-budget recall (10/20/40%) is reported next to prevalence-matched R-precision.
4. **Training config**: tuning and seed runs share one config (12-trial random search with bagging_fraction 0.8).
   The earlier five-seed run (commit 31aff42) added bagging that the tuning run lacked; those numbers are superseded.
5. **Sensitivity calibration**: ~38% of pilot and 43% of test N-2 groups are control-sensitive against a 15-40% target.
   Recorded as a deviation; no recalibration against inspected test data. Consequence: control-dependence is a large
   share of N-2 severity, which enlarges (not shrinks) the room for a control-aware method.
6. **Unseen-cell composition**: unseen test failure sets are 40/30/30% size 2/3/4 with 23% complete control loss, seen
   ones mostly size 1-2 with 12.7%; 36.1% of unseen rows have a control vector that also occurs in train. Results must be
   reported separately for familiar vs novel control vectors and by size (prevalence differs: 39.8% vs 23.9%).
7. **Existing test operating points (seeds 200-279) are exploratory** because they have been inspected. Confirmatory
   claims use fresh operating points (seeds 300-379) generated after the hypothesis was committed.

### Justification of physical benchmark assumptions (these are benchmark assumptions, not claims of realism)
- **Load shedding needs no communication.** Represents local under-frequency/under-voltage relays and is the standard
  last resort in corrective-action models; a variant where shedding also needs remote command would make many scenarios
  infeasible and is out of scope.
- **Protection trips only rebalance an island's shed plus its initial surplus** (frozen rule). Sensitivity pilot on
  development seeds (10 operating points, all N-1 + 150 N-2 each, 14 control vectors, N-2 groups): share of groups whose shed
  varies with control by more than 0.5% of demand is 0.382 (frozen rule), 0.382 (surplus-only; 2.7% of scenarios have no
  feasible action) and 0.368 (free tripping, an upper bound on control-independent relief). The value of control is therefore
  not created by the trip rule.
- **DC, static snapshot, corrective interval as ranges**: results say nothing about voltage, reactive power or dynamics.
