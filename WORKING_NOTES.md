# WORKING_NOTES — ML_FDNA Milestone 1
Task: validated reference generator + strong non-FDNA baselines (plan: ~/.claude/plans/using-brainstorming-research-ideas-and-d-witty-stonebraker.md).
Status (2026-09-23): spec frozen (prereg/SPEC.md, configs/spec.json); LP + comm layer + tests (5 pass); calibration done
(kappa 1.6, ramp 0.3, local 40); dataset in data/ (5.3M rows, ~8.5 min on 30 cores); gate0 report run; baselines running.
Next: read baseline table, write results/MILESTONE1.md, code-review + qa-verify, decide with Philip whether FDNA layers are justified.
Decisions/deviations: primary endpoint amended to R-precision (prevalence 30-36%, 10% budget caps recall ~0.28); GBT tuned on val N-2.
Commits: authored by Philip, no attribution lines.
