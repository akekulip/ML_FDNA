# Frozen v2 specification: partial / stale observation of control availability (Phase 2, Branch 1)
Written before any Phase-2 result exists; extends prereg/SPEC.md (the DC corrective-shed labels are unchanged).

**Generative model.** Hidden state x in {0,1}^14 (1 = up) for the components of `fdna.comm` (CC, 3 gateways, 10 unit RTUs).
Each component fails independently with p_fail = 0.15; two common-cause events (regional site A = GW0 + RTU0-3, prob 0.08;
gateway facility = GW1 + GW2, prob 0.08) force their members down. Physical basis: shared substation power/comm hub and
co-located gateways (shared-risk groups). Parameters fixed a priori, not tuned.

**Observations.** 13 binary flags: for each of the 10 units the poll-response flag (true value = the unit is reachable:
CC up, its RTU up, at least one parent gateway up) and for each gateway the heartbeat (CC up and gateway up). Each flag is
missing with probability q; an observed flag of a failed item is reported "up" (stale) with probability s, never the reverse.
Grid: q in {0.1, 0.3} x s in {0.0, 0.2}; all four cells are reported. Wiring is observed.

**Label.** Realised shed y = V(operating point, outage, control vector of the TRUE state) from an LP value table over all
1,024 control vectors; the model sees only observations plus electrical/physics features.

**Bayes oracle.** Posterior over the 16,384 states by exact enumeration, prediction = posterior mean of V. Missing flags are
ignorable; observed flags contribute e(v | truth): e(1|1)=1, e(0|1)=0, e(1|0)=s, e(0|0)=1-s.

**Freeze rule.** Nothing in this file or in the observation/wiring/label code may change after the first FDNA (A5-A7) run;
selection of whether to proceed uses only the oracle and tuned tree arms (gate E1). Any later change opens a new spec version.

**Design-for-difficulty disclosure.** v2 was designed after Phase 1 showed the fully observed binary setting gives structure
nothing to add. It is chosen on physical grounds; E1 measures whether headroom exists for any learner.

## Addendum (frozen 2026-09-24, before any FDNA arm A3-A7 was ever run on v2)
**Variants (pre-registered in registry/registry.yaml before their gates ran).** v2 (mild: q in {0.1,0.3}, s in {0,0.2}), v2b
(harsh telemetry: q in {0.5,0.7}, s in {0,0.3}), v2c (sparse coverage: only the 3 gateway heartbeats and unit poll flags of
units 0, 4, 8 are observed; q in {0.1,0.3}, s in {0,0.2}). All 12 cells are reported.

**Gate outcomes (exploratory test block 200-279, n=100 training operating points, tuned trees).**
- E1 (oracle vs best tree) passed as registered (+0.12 to +0.16) but is invalid as an inference-headroom measure: the oracle
  receives the exact LP value table, and trees given the exact posterior control marginals (A2b) gain nothing.
- E1b (corrected gate: exact-control GBM minus best observation tree): v2 max +0.027 [+0.022,+0.033] (fail, mild telemetry costs
  trees almost nothing); v2b +0.018/+0.064/+0.054/+0.102; v2c +0.009/+0.036/+0.024/+0.064. v2b and v2c pass (>= 0.03 with CI
  excluding 0).

**Primary cells for the confirmatory hypotheses (declared now).** P1 = v2b q=0.7, s=0.3 (largest headroom, +0.102); P2 = v2c
q=0.3, s=0.2 (sparse coverage, +0.064). Selection used only the oracle and tuned tree arms.

**Freeze.** Observation model, coverage masks, prior, wiring, labels and the value table are frozen from this commit (git tag
`v2-freeze`). Any change opens a new spec version and invalidates results produced after it.
