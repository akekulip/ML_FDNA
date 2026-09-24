"""Error analysis of the best tree model (seed-averaged predictions) on unseen N-2, by candidate error drivers.
Descriptive only: it does not isolate causes."""
import sys
import numpy as np
import pandas as pd
from fdna import baseline_data, spec
from fdna.evalutil import rank_order

DATA = sys.argv[1] if len(sys.argv) > 1 else "data"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "elec+phys+ctrl"
d = baseline_data.load(DATA)
P = np.load(f"{DATA}/preds_seeds.npz")
pred = np.mean([P[f"{MODEL}|{s}"] for s in range(5)], axis=0)
m = d.te & (d.lab.cell.values == "n2_unseen")
lab = d.lab[m].copy()
lab["pred"] = pred[m]
X = d.F["elec+phys+ctrl"][m]
nb = 41
ph0 = 30 + 6 + nb + 1  # physics block start in the feature matrix
lab["max_overload"] = X[:, ph0 + nb]
lab["n_over"] = X[:, ph0 + nb + 1]
lab["ctrl_total"] = X[:, -1]
lab["y_struct"] = lab.y_struct
lab["sev"] = lab.y > spec.SEVERE
lab["err"] = (lab.pred - lab.y).abs()
lab["dist"] = (lab.y - spec.SEVERE).abs()
# ranking miss: within each op, prevalence-matched top-k
lab["miss"] = False
for o, g in lab.groupby("op"):
    k = int(g.sev.sum())
    top = set(g.index[rank_order(g.pred.values, d.key[m][lab.index.get_indexer(g.index)])[:k]])
    lab.loc[g.index, "miss"] = g.sev & ~g.index.isin(top)

def report(name, col):
    g = lab.groupby(col, observed=True).agg(rows=("y", "size"), severe=("sev", "mean"), mae=("err", "mean"),
                                            miss_rate_of_severe=("miss", "sum"), n_sev=("sev", "sum"))
    g["miss_rate_of_severe"] = g.miss_rate_of_severe / g.n_sev
    print(f"\n-- {name} --")
    print(g.drop(columns="n_sev").round(4).to_string())

lab["islanding"] = np.where(lab.y_struct > 0, "structural shed", "no structural shed")
lab["control"] = pd.cut(lab.ctrl_total, [-1, 1e-9, 1.5, 3.5, 4.99, 5.01], labels=["none", "<=1.5", "1.5-3.5", "3.5-5", "full"])
lab["overload"] = pd.cut(lab.max_overload, [-1e-9, 1, 1.25, 1.5, 2, 1e9], labels=["<=1", "1-1.25", "1.25-1.5", "1.5-2", ">2"])
lab["dist_thr"] = pd.cut(lab.dist, [-1, 0.002, 0.005, 0.01, 0.03, 1], labels=["<0.2pt", "0.2-0.5pt", "0.5-1pt", "1-3pt", ">3pt"])
lab["size"] = d.fs_size[m]
lab["novel"] = np.where(d.novel[m], "novel control vector", "familiar")
print(f"model {MODEL}; unseen N-2 test rows {len(lab)}; severe prevalence {lab.sev.mean():.3f}; overall MAE {lab.err.mean():.5f}")
for name, col in [("structural islanding", "islanding"), ("commandable control (sum of generator fractions)", "control"),
                  ("worst post-outage overload ratio (DC, no control)", "overload"),
                  ("distance of true shed from the 1% threshold", "dist_thr"), ("failure-set size", "size"), ("control-vector novelty", "novel")]:
    report(name, col)
