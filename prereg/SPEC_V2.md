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
