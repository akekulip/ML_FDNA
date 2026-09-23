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
