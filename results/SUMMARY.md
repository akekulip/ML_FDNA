# ML_FDNA — what the overnight programme established (2026-09-24)

**Question.** Can an FDNA-structured neural model give a novel, defensible advantage for cyber-physical grid contingency screening?
**Short answer.** Not on this benchmark family, and the evidence says why. The one confirmed positive effect is small and is not about FDNA.

## Findings, ordered by how much weight they can bear
1. **Tier-2 effect confirmed on a fresh block and re-observed on a second fresh block; small; non-FDNA.** Learning the value model on full-information simulation rows and the inference module on cheap
   (obs, c) pairs, then composing them, beats the strongest end-to-end baseline that also receives the cheap pairs by +0.034..+0.036 (P1) and
   +0.025 (P2) R-precision on two fresh blocks (tier 2: reliable but below the pre-declared 0.05; tier 1 not met). The two runs share their training draws, so this is not an independent replication of training variability. About half of this is the choice of inference
   module; the treatment uses simulator-side privileged information.
2. **Structure cannot buy much on inference.** An exact differentiable Bayesian layer over the known graph (17 parameters) gets within KL 0.02 of the exact
   posterior from 1,000 samples, yet is worth at most ~0.01 R-precision over a per-generator GBM (secondary scoring) and nothing under the registered
   scoring. FDNA's mean-field operability algebra is dominated by both.
3. **This exact-structure inference model is fragile in the tested family.** With 10% of the wiring wrong, the exact-structure model falls 0.07-0.13 below the wiring-agnostic learner; a robust variant with a
   learnable leak does not fix it, and all corruption seeds stay below the generic learner (one corruption family, one replicate per setting, exploratory; not a general statement about structured inference).
4. **The dominant error is not dependency modelling.** Unseen N-2 generalisation under the train-on-N-1 protocol (0.93-0.97 on N-1 rows vs 0.79-0.89 on N-2 with
   exact control); better features do not fix it; 10 labelled N-2 pairs per operating point close 25-40% of it.
5. **Method lessons.** Gates that compare against an exact-control model conflate irreducible information loss with reducible inference headroom; neural
   baselines need a target-scale check; the posterior mean of shed is not the R-precision-optimal ranking; an un-normalised soft-min silently capped
   FDNA operability; block-seed reuse and an advisory lock made a "fresh" block partly non-fresh. All found by reference arms or independent review and fixed.

## What this is not
No FDNA advantage; no claim beyond a 14-component, enumerable, DC-LP, synthetic-wiring benchmark; not fully pre-registered (a registered exploratory
protocol with one amended confirmatory claim; forking-paths ledger in `results/LEDGER.md`).

## Where a novelty path might still exist (untested)
(a) **Scale**: large or deep dependency graphs where exact enumeration fails and generic learners' sample complexity bites (needs a pre-registered generator and a
headroom gate on reducible inference gain, computed against an approximate-posterior oracle, before any FDNA arm). (b) **N-2 label efficiency**: the actual error
source here; a value learner that exploits N-1 information for N-2 pairs, with the few-shot protocol. (c) A second topology (e.g. IEEE-118) to test whether
the characterisation transfers.

## Recommended framing for a paper
A benchmark + evaluation-protocol paper with a headroom decomposition, a ceiling-and-fragility analysis of structured inference, the confirmed small
privileged-information effect, and the N-1 to N-2 label-efficiency result; workshop or IEEE Access tier (independent reviewers' estimate), stronger with a
second topology.
