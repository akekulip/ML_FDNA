"""Phase 3: exact Jensen-gap check (no training). y(c) is convex in c for fixed (op,outage) (LP-duality argument,
c enters only via variable bounds). E_post[y(c)] - y(E_post[c]) is a lower bound on how much a mean-of-c predictor
must lose relative to the true posterior expectation. Predicts: gap larger where posterior over c has more spread,
i.e. P1 (q=.7,s=.3) > P2 (q=.3,s=.2). Uses the exact posterior (no learned inference), test table, 25 sampled rows/cell."""
import json
import numpy as np
from fdna import v2, v2data

w = v2.build_world(); te = v2data.load_vtable("data_v2", "test")
CELLS = {"P1_v2b": (0.7, 0.3, None), "P2_v2c": (0.3, 0.2, v2.COVERAGE_SPARSE)}
rng = np.random.default_rng(0)
out = {}
for name, (q, s, cov) in CELLS.items():
    gaps = []
    idx = rng.choice(te["V"].shape[0] * te["V"].shape[1], 60, replace=False)
    io, jc = np.unravel_index(idx, te["V"].shape[:2])
    for i, j in zip(io, jc):
        y = te["V"][i, j].astype(np.float64)                       # (1024,) over CV
        if y.std() < 1e-9: continue
        st = v2.sample_states(w, rng, 400)
        obs = v2.emit(w, st, q, s, rng, cov)
        pc = v2data.oracle_features(w, obs, s)[0]                  # (400, 1024) exact posterior per draw
        Ey_c = (pc * y[None]).sum(1)                                 # E_post[y(c)] per draw
        c_bar = pc @ w.CV                                            # E_post[c] per draw
        # y(E[c]): nearest CV row to c_bar (evaluate via the same LP-derived table by nearest neighbour, exact for CV rows themselves)
        d = ((w.CV[None] - c_bar[:, None]) ** 2).sum(2); near = d.argmin(1)
        y_Ec = y[near]
        gaps.append((Ey_c - y_Ec).mean())
    out[name] = {"n_rows": len(gaps), "mean_jensen_gap": float(np.mean(gaps)), "frac_positive": float(np.mean(np.array(gaps) >= -1e-9))}
print(out)
out["prediction"] = "P1 gap > P2 gap (wider posterior spread at q=.7,s=.3 than q=.3,s=.2)"
out["observed_direction_matches_prediction"] = bool(out["P1_v2b"]["mean_jensen_gap"] > out["P2_v2c"]["mean_jensen_gap"])
json.dump(out, open("results/phase3/step1_jensen.json", "w"), indent=1)
