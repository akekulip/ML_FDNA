"""Anomaly check: is the posterior-mean-of-V oracle the R-precision-optimal ranking? Compare with the exact posterior
probability of severity P(y > 0.01 | obs). Cell P1 (v2b q=0.7,s=0.3) and P2 (v2c q=0.3,s=0.2); screening block 200-279."""
import numpy as np
from fdna import spec, v2, v2data
from fdna.evalutil import op_metrics
D = "data_v2"
w = v2.build_world()
te = v2data.load_vtable(D, "test")
dummy = np.zeros(te["conts"].shape[:2] + (1,), np.float32)
for name, q, s, cov in (("P1 v2b q=0.7 s=0.3", 0.7, 0.3, None), ("P2 v2c q=0.3 s=0.2", 0.3, 0.2, v2.COVERAGE_SPARSE)):
    d = v2data.build(te, dummy, w, q, s, 8, seed=13, only_n2=True, coverage=cov)
    pc = v2data.oracle_features(w, d["obs"], s)[0]
    mean_v = np.concatenate([(pc[a:a + 8192] * te["V"][d["io"][a:a + 8192], d["jc"][a:a + 8192]]).sum(1) for a in range(0, len(pc), 8192)])
    p_sev = np.concatenate([(pc[a:a + 8192] * (te["V"][d["io"][a:a + 8192], d["jc"][a:a + 8192]] > spec.SEVERE)).sum(1) for a in range(0, len(pc), 8192)])
    res = {}
    for nm, pr in (("posterior mean of V", mean_v), ("P(severe | obs)", p_sev)):
        res[nm] = np.nanmean([op_metrics(d["y"][d["op"] == o], pr[d["op"] == o], d["key"][d["op"] == o])["rprec"] for o in np.unique(d["op"])])
    print(name, {k: round(v, 4) for k, v in res.items()})
