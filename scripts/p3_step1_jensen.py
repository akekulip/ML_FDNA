"""Phase 4 correction (brief section 4.1): exact Jensen-gap check. The Phase-3 version snapped the posterior mean
control to the nearest of 1024 discrete CV rows (V(round(E[C]))), not the true V(E[C]) at the continuous mean.
This version calls ScenarioLP.y(c_bar) directly at the continuous posterior mean -- the LP's _bounds() already
expresses each corrective cap as its own affine constraint (min(ramp_frac*pmax*c, ...)), so any c in [0,1]^5 is a
valid continuous evaluation, not an approximation.

y(c) is convex in c for fixed (op, outage) because c enters the LP only through those bound caps (composition of a
convex LP value function with a concave/convex monotone bound map); Jensen's inequality then gives
E_post[y(c)] >= y(E_post[c]) whenever the posterior over c has spread. This script tests that inequality exactly
and reports per-observation gaps (not only the mean), solver status, and both cells' gap distributions.
"""
import json
import numpy as np
from fdna import spec, v2, v2data
from fdna.dataset import G, RATING
from fdna.lp import ScenarioLP
from fdna.opgen import sample_op

w = v2.build_world(); te = v2data.load_vtable("data_v2", "test")
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
rng = np.random.default_rng(0)
out = {}
for name, (q, s, cov) in CELLS.items():
    gaps, failed = [], 0
    idx = rng.choice(te["V"].shape[0] * te["V"].shape[1], 40, replace=False)
    io, jc = np.unravel_index(idx, te["V"].shape[:2])
    for i, j in zip(io, jc):
        y = te["V"][i, j].astype(np.float64)
        if y.std() < 1e-9:
            continue
        op_id = int(te["op_ids"][i]); removed = tuple(int(x) for x in te["conts"][i, j] if x >= 0)
        op = sample_op(G, RATING, np.random.default_rng(op_id))
        lp = ScenarioLP(G, op, removed, spec.PARAMS)
        st = v2.sample_states(w, rng, 200)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]                  # (200, 1024) exact posterior per draw
        Ey_c = (pc * y[None]).sum(1)                                 # E_post[y(c)], exact from the table (no approximation)
        c_bar = pc @ w.CV                                            # (200, 5) continuous E_post[c] in [0,1]^5, NOT snapped
        for row in range(len(c_bar)):
            try:
                y_Ec = lp.y(c_bar[row])                               # exact continuous LP evaluation, y(E[c])
            except RuntimeError:
                failed += 1; continue
            gaps.append(float(Ey_c[row] - y_Ec))
    g = np.array(gaps)
    out[name] = {"n_rows": len(g), "n_solver_failed": failed, "mean_jensen_gap": float(g.mean()),
                 "median": float(np.median(g)), "p10": float(np.percentile(g, 10)), "p90": float(np.percentile(g, 90)),
                 "frac_nonnegative": float((g >= -1e-9).mean()), "min": float(g.min()), "max": float(g.max())}
    print(name, out[name])
out["method_note"] = "y(E[c]) evaluated by direct continuous LP solve (ScenarioLP.y), not nearest-discrete-CV-row lookup"
out["prediction"] = "P1 (wider posterior spread, q=.7 s=.3) mean gap > P2 (tighter, q=.3 s=.2)"
out["observed_direction_matches_prediction"] = bool(out["P1_v2b"]["mean_jensen_gap"] > out["P2_v2c"]["mean_jensen_gap"])
json.dump(out, open("results/phase3/step1_jensen.json", "w"), indent=1)
