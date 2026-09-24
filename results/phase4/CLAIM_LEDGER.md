# Phase 4 — Claim ledger (brief section 10)

| # | Claim | Status | Evidence |
|---|---|---|---|
| 1 | Jensen gap E_post[y(c)]-y(E_post[c]) is positive, direction P1>P2 | SUPPORTED (corrected) | scripts/p3_step1_jensen.py exact continuous evaluation; results/phase3/step1_jensen.json; 98.2%/93.4% non-negative, small negatives at solver-tolerance scale |
| 2 | Jensen gap CAUSES the D_I2-vs-A8/A9 decomposition effect | NOT TESTED (hypothesis only) | discriminating plug-in-vs-learned-summary test not yet run; explicitly labelled a candidate mechanism, not proven |
| 3 | generalized_det(S) exactly recovers topological cuts at k=2, LP-free | SUPPORTED | tests/test_hik_diag.py::test_generalized_det_zero_recovers_new_2cuts (26/26) |
| 4 | minimal_cut_struct matches ScenarioLP.struct_mw exactly | SUPPORTED (after a caught bug fixed) | tests/test_hik_diag.py, 2 tests, exact match atol 1e-6 |
| 5 | Low-degree (k=2-additive) Mobius truncation from N-1/N-2 predicts exact N-3 shed better than a naive singleton sum AND an equal-information GBM | SUPPORTED, EXPLORATORY ONLY | results/phase4/RESULTS.md, results/phase4/mobius_k3_robust.json; 4/4 control states, no confirmatory block, no cluster bootstrap yet, N-4 untested |
| 6 | Repaired-FDNA (OR-aware, hard-min, joint/CC-prior) representability is FIXED | SUPPORTED | tests/test_repair.py (6/6, exact motif fit) |
| 6b | The repaired FDNA arm beats plain dependency logic given the same head (registry/phase3_step2.yaml's own attribution rule) | NOT SUPPORTED, both cells complete | results/phase4/RESULTS.md: P1 J_OR_hard_CC KL 0.174+-0.008 vs L_CC 0.099+-0.006, L_indep 0.103+-0.005; P2 J_OR_hard_CC KL 0.169+-0.010 vs L_CC 0.087+-0.012, L_indep 0.097+-0.010. Repaired FDNA arm is WORSE than plain logic with the same joint head in BOTH cells; ties the generic MLP in both. |
| 7 | Recurrent architecture (GRU/DeepSets) beats matched baselines on N-1->N-k transfer | NOT SUPPORTED at this label budget | results/phase4/RESULTS.md, results/phase4/setmodels_k3k4.json: both DeepSets and GRU, trained on 20 ops (N-1 + 300 sampled N-2), zero-shot on N-3/N-4, perform WORSE than the trivial naive-sum baseline g1, even with equal information (known singleton values as a feature). Read as a data-scale limitation (brief 6.5's own gating language), not a claim against recurrence in principle. Convex value head not yet tested. |
| 8 | FDNA (strength/criticality algebra specifically) adds anything beyond recurrence/logic/physical features | NOT TESTED, PRIOR EVIDENCE NEGATIVE | Phase 1-3: no FDNA arm beat matched baselines in any registered comparison; nothing in Phase 4 changes this yet |
| 9 | Second-topology (IEEE-118) generalization | NOT STARTED | grid.py hard-coded to case_ieee30 |
