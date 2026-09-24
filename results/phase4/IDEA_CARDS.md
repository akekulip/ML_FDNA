# Phase 4 — Idea cards (brief section 5)

Three agents ran in parallel and independently: a literature reviewer (verified primary sources, not abstracts),
and two brainstorming agents blind to each other — Agent A (numerical methods / control), Agent B (compositional
learning / reliability). Agreement between A and B is reported as agreement, not treated as evidence; the
convergence noted below was checked directly against the code after both reported.

## Literature verification (headline findings)
- **Nakiganda & Chatzivasileiadis (2023)**, verified full text: feed-forward N-1-trained, zero-shot N-2/N-3
  evaluation on PGLib grids. Closest non-recurrent competitor; recall degrades badly at N-3 (<40% undervoltage
  recall for their GDNN variant) — a concrete baseline weakness. No comms layer, no corrective shed, no recurrence.
- **Physics-Informed Graph Neural Jump ODEs (2026 preprint)**, verified full text: trains N-1..N-4 *jointly*
  (not small→large transfer), autoregressive, own authors flag error accumulation as unresolved. Closest to
  "iterative processing across outage size" but not a transfer claim and no comms layer.
- **Christianson et al., ICNN N-k screening (L4DC 2025)**, verified full text: convexity guarantee is for a
  **fixed dispatch** and **fixed, pre-specified contingency set**, binary feasibility only. Does NOT cover: a
  continuous corrective-shed value, cross-k transfer, or comm-dependency uncertainty. Any convex-value-head work
  in this project must not claim their certificate — only their proof *pattern* is reusable, not the result.
- **Manoharan (arXiv 2607.13221)**, verified: risk-controlled selective verification for N-1 screening already
  exists. The "active label acquisition" candidate direction is no longer clean-room at N-1; N-k>1 + comms
  extension remains open.
- **Roy & Hylviu (arXiv 2607.08918)**, re-verified from the actual abstract (not a prior summary): uses MIIM
  (Boolean/multi-valued implicative model), not FDNA, and gradient boosting, not a neural net. Does not pre-empt
  an FDNA-specific neural claim, but does take the narrower "fast ML surrogate for power-comm criticality
  ranking" framing.
- **Searched, not assumed absent:** temporal/cyclic FDNA extension — none found. System Operational Dependency
  Analysis (SODA, Guariniello & DeLaurentis) is real but general systems-engineering scope, no recurrent/N-k
  angle, does not encroach.
- **Least-crowded direction, per the literature agent:** recurrent/iterative set-processing achieving true
  small→large transfer (not joint training) + exact order-independence + corrective-shed label + comm-dependency
  layer. No single verified paper covers that combination. Matches the brief's own preferred direction.

## Independent convergence (checked, not just reported)
Agent A's #1 idea (generalized k-way LODF compensation determinant) and Agent B's #1-ranked-by-B idea (minimal
cut-set/struct_mw generalized to arbitrary k) are BOTH exact, LP-solve-free generalizations of the two k=2
mechanisms already verified in this project (`src/fdna/lodf.py`'s `compensation_det`, `ScenarioLP.struct_mw`).
Implemented and correctness-gated this session as `src/fdna/hik_diag.py` (`tests/test_hik_diag.py`, 5/5 pass):
`generalized_det` exactly reproduces the 26 known 2-cuts; `minimal_cut_struct` exactly matches
`ScenarioLP.struct_mw` (a real bug — nominal `grid.load` vs the operating point's actual `demand` — was caught
and fixed by this test before being trusted).

## Idea cards (>=6 across >=3 fields, per the brief's schema)

**1. [Agent B, #2 by B, PROMOTED — see Results] Möbius/interaction-index truncation of y(S).**
Field: compositional learning / cooperative game theory. Mechanism: `g_k(S) = sum_{T subset S, |T|<=k} I(T)`,
`I(T)` the Möbius transform of the set function `y`. Gap: turns "does N-1/N-2 generalize to N-3/N-4" into an
explicit, falsifiable truncation-order model instead of an architectural hope. Prior work: Grabisch & Roubens
1999 (interaction indices in cooperative games); Stobbe & Krause 2012 (unverified this session). Equal-info
baseline: GBM given the same numbers. **Tested this session — see Results below; the strongest positive finding
of Phase 4 so far.**

**2. [Agent A, #1] Generalized k-way LODF compensation determinant `Det_k(S)`.**
Field: numerical methods / parametric sensitivity analysis. `det(M_S)`, `M_S[i,i]=1`, `M_S[i,j]=-LODF(i,j)`.
Reuses the one Laplacian factorization already computed at k=2; free for all 10,660 triples / 101,270
quadruples. Implemented and correctness-gated (`hik_diag.generalized_det`). Role: a cheap risk/triage signal for
where naive flow-addition breaks, generalized from k=2 (verified rho=0.945) — not a shed predictor itself.

**3. [Agent B, #1 by B] Minimal-cut-set structural floor, generalized to k=3/4.**
Field: reliability engineering (fault trees, minimal cut sets). Exact, LP-free, graph-connectivity-only shed
floor for outage sets whose islands have no generator. Implemented and correctness-gated
(`hik_diag.minimal_cut_struct`). Targets exactly the failure region G1 already identified (islanding pairs hold
58.9% of missed-severe mass at k=2). Self-flagged risk (both agents, independently): this only covers the
topological share; 41% of k=2 misses are in connected pairs with no cut at all, and that share may grow at
higher k.

**4. [Agent A, #2] Risk-stratified adaptive labeling budget.** Field: multi-fidelity / experimental design.
Combines cards 2+3 into a labeling-budget policy (top-decile-by-risk gets exact recompute; rest gets cheap
estimate), mirroring the already-measured k=2 result (82% of p95-error tail from ~10.1% of pairs). Agent A's own
self-critique: may collapse into the already-known static rule with no real gain from an "adaptive" layer — not
yet tested.

**5. [Agent A, #3] Active-set / basis warm-starting across the outage lattice.** Field: numerical
optimization. Warm-start N-3 LP solves from the parent N-2 solution's basis. Flagged by Agent A as contingent
on an unverified fact: scipy's `linprog(method="highs")` wrapper may not expose basis warm-starting without
dropping to `highspy` directly — not checked this session. Pure infrastructure/cost idea, not a predictive claim.

**6. [Agent B, #3 by B] Interaction-guided active N-3/N-4 label acquisition.** Field: experimental design.
Downstream of cards 2+3 — score triples by risk, buy labels for the highest-risk decile plus a random control
decile. Not run this session (correctly sequenced behind cards 2/3/1 per Agent B's own ordering).

## Dissent (required output, not optional)
- **Agent A rejected:** numerical continuation/homotopy in outage count (the LP is linear and cheap to
  cold-solve; continuation methods solve a problem — nonlinear equilibrium tracking — this project doesn't have);
  a learned active-set classifier (circular: needs the N-3/N-4 training data the brief says is scarce); a
  data-driven fit of the k-way compensation matrix (strictly worse than the free exact closed form, card 2,
  unless card 2 is shown to fail).
- **Agent B rejected:** noisy-OR aggregation for `y(S)` (category error — `y` is continuous/convex/monotone LP
  value, not a Bernoulli event; conflates the comm-bit OR-motif finding with a different, unrelated domain); full
  BDD/ADD compilation of the outage space (needs the same scarce labels it's supposed to avoid needing);
  DeepSets/Set-Transformer trained N-1+N-2, zero-shot N-3/N-4 as a *standalone* claim (architecturally the same
  bet as the already-published Donnot 2018 / Nakiganda 2023 without an interaction-order diagnosis).
- **Cross-check performed by the main session (not by either agent):** Agent B flagged, correctly, that the
  comm-bit degree-3 structure (STEP1 item 5, R2=0.998) lives in a *different* Boolean cube (14 comm components)
  than the outage-set cube (41 branches) and must not be cited as evidence for outage-set low-degree structure.
  The Möbius test (card 1, Results below) is therefore a genuinely new, separately-measured finding, not a
  restatement of the comm-bit result.

## Shortlist (PI selection, recorded before running any confirmatory look)
**Selected: card 1 (Möbius/interaction truncation)** — already screened this session with a strong, robust
positive result (see RESULTS.md). **Selected as the structural-feature layer for the main recurrent screen: cards
2+3** (generalized Det_k and minimal-cut floor) — cheap, exact, correctness-gated, and both agents independently
converged on them; they feed the brief's staged comparator set as equal-information features for every arm, not
as a separate model. **Deferred:** cards 4-6 (label-budget policy, warm-starting, active acquisition) — real but
downstream of 1-3 per both agents' own sequencing, and out of scope for the remaining Phase-4 budget this
session. Order recorded here BEFORE any further screening result could bias it.
