# Literature notes (full text read where stated)

## Roy & Hylviu, arXiv 2607.08918 (July 2026) — full text read (PDF), 2026-09-23
- Task: predict contingency severity S(F) in [0,1] (operability loss over a horizon) for N-k failures in interdependent
  power-communication networks; ground truth is the MIIM cascade model (a discrete dependency simulator, not a power-flow
  LP); IEEE 14- and 118-bus.
- Features: thirteen pre-simulation structural features (order k; counts of failed power/communication/other nodes; number and
  fraction of severed inter-layer edges; gateway involvement and failed-gateway fraction; maximum failed-node degree; failed-subgraph
  connectivity; ratios; event type). No electrical operating-point information.
- Models: linear regression, random forest, gradient boosting only. No neural network or GNN was tried; the paper names
  "stronger boosted-tree and graph-neural-network models, cross-network transfer, richer dependency-aware representations" as future work.
- Evaluation: 80/20 severity-class-stratified random split (not held-out contingency combinations), Spearman and Precision@5%; 118-bus
  Spearman 0.849. No learning curve or data-efficiency analysis.
- Ablation: removing the four inter-layer features lowers Spearman from 0.849 to 0.799; inter-layer features alone reach 0.763.
- Speed: 228.5 s for 10,446 MIIM evaluations (21.87 ms each) vs 1.44 s end-to-end surrogate screening (0.14 ms each, feature extraction
  dominates; inference 1.2 microseconds) = about 158x marginal speed-up; break-even after about 10,600 further contingencies. Measured
  by the authors on a laptop; not independently reproduced here.
- Consequence: a tuned tree model on dependency features is the incumbent, and the room left is exactly what this project probes:
  held-out failure combinations, low training-data regimes, and whether structured (FDNA-like) dependency modelling beats tree-on-features.
  It does not support a broad "communication-aware screening" novelty claim.
