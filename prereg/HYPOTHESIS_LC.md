# Pre-specified hypothesis for the learning-curve experiment (Step 4)
Written and committed BEFORE any fresh (seed 300-379) operating point is generated or any learning-curve model is trained.

**Question.** With few training operating points, does an explicitly computed control-availability feature set beat the
14 raw component states (same everything else, tuned tree model)?

**Design.** Training operating points n in {5, 10, 25, 50, 100}, five replicates per n (random subset of the 100 training
operating points, seed = replicate index; n=100 replicates differ by seed only). Feature sets: elec+ctrl vs elec+raw_comm,
and elec+phys+ctrl vs elec+phys+raw_comm; also elec as reference. Hyper-parameters per feature set fixed to the values
tuned on the full training set; early stopping on the fixed validation operating points. LightGBM, same config for all.

**Primary hypothesis H1 (fresh operating points, seeds 300-379, cell n2_unseen|novel).** For each n in {5, 10, 25}, the
operating-point-level paired difference in R-precision (explicit minus raw, replicates averaged within operating point)
is at least +0.05 with a 95% cluster-bootstrap confidence interval whose lower bound exceeds 0, for the pair
elec+phys+ctrl vs elec+phys+raw_comm. H1 holds only if all three sizes satisfy it.

**Secondary (reported, not confirmatory).** The no-physics pair; the cells n2_unseen|familiar and n2_unseen overall; sample-size
ratio (smallest n at which each set reaches the other's n=100 R-precision); recall at 20% budget; MAE. The existing test
operating points (seeds 200-279) are reported as exploratory only.

**Decision rule (from the approved plan).** H1 not supported: stop the FDNA architecture claim, do not enlarge or
re-tune the simulator to obtain a gap, write up the narrow Milestone 1 finding. H1 supported: propose (needs fresh
approval) a small FDNA module vs an equally informed generic neural model and a sum/min/max aggregation baseline on the
same electrical predictor, plus a changed-wiring test.

**Not allowed after seeing results:** changing n values, the pair, the cell, the threshold, the metric, or excluding
replicates.
