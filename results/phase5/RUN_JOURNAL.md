# Phase 5 (overnight PI execution brief) — run journal

Start: 2026-09-25T02:44:09Z (UTC) / 2026-09-24 22:44:09 EDT. Deadline (8h ceiling): 2026-09-25T10:44:09Z / 06:44 EDT.
HEAD at start: 7b7cb3ea33ff207177d2bb5679e77477960ccff8 (matches the brief's audited checkpoint). Branch: main. Dirty: only the new brief file (untracked).
Hardware: RTX 2070 8GB (274 MiB used, 0% util — idle), 32 cores, 31Gi RAM (26Gi available). No ML_FDNA jobs running (only unrelated MCP servers for other projects).

## Stage 0 — first-hour corrections (6 items, all verified before this run started)
1. Triple overlap (0,23,34) / (9,13,40) confirmed present in both manifest_k3_confirm.json and manifest_amortization_new200.json — verified exactly, zero overlap with the original 60-triple screen.
2. Cost breakdown verified exact: N-1=2,519,040; N-2=11,182,080; N-3=4,300,800; N-1+N-2=13,701,120; total 18,001,920.
3. Decision-rule gap confirmed: p4_mobius_confirm.py never ran an executable intersection_union check.
4. Amortization coverage (2,698/8,436) already correct from earlier session, not regressed.
Work log below.

## Stage 0 — completed (all 6 items)
1. 68-triple sensitivity (excludes (0,23,34), (9,13,40), manifest-membership only): P1 +0.254 (was +0.250), P2 +0.273 (was +0.270) -- nearly identical, overlap triples had negligible effect on the conclusion.
2. Executable intersection_union decision (src/fdna/rules.intersection_union, SESOI=0.05 margin on all 4 components): tier1_confirmed_strict_lo90_ge_margin = TRUE, all_positive = TRUE. Formal, code-based confirmation, not eyeballed prose.
3. No-control linear finding restated with paired bootstrap over the 70 triples: g2 MSE 7.691e-5 vs L1 7.331e-5 (4.68% gap), mean diff 3.60e-6, 90% CI [1.62e-6, 5.65e-6] -- a small but real, statistically confirmed effect (lo90>0), not a bare point estimate. Note: this is the OPPOSITE direction from OLS in the earlier exploratory screen (OLS beat g2 by 17.9% there); in this confirmatory run g2 beats OLS outright and only L1 keeps a small edge.
4. Per-row predictions with canonical keys saved: data_hik/confirm_predictions.npz (87MB, op_id/outage/cell/draw/g1/g2/prior_only/y_true/key), enabling independent recomputation without retraining or resolving.
5. Cost attribution already recorded exactly (verified this session): N-1=2,519,040, N-2=11,182,080, N-3=4,300,800, total 18,001,920.
6. Audited: op_ids 660-699 confirmed untouched by any manifest in the project (grep across all data_hik/*.json). Genuinely fresh, available for any future look.

Scripts: scripts/p4_mobius_confirm_v2.py (kept p4_mobius_confirm.py, not deleted). Output: results/phase5/mobius_confirm_corrected.json, results/phase5/linear_nocontrol_paired.json.
No new LP solves needed -- reused cached n1/n2/n3_confirm.npz label tables from the original confirmatory run.
