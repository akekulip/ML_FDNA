"""Phase 5 repair round 2: feature-building logic factored OUT of the training/eval scripts so it is
importable and testable without running the whole pipeline. This is the same lesson `pairset.py`'s own
docstring already recorded for the model class itself ("src/fdna/nn was the wrong place for it to live -- it
was embedded directly in a top-level script, which is why the bug survived: importing the class for testing
required running the whole training pipeline") applied to the feature-building side, where the SAME class of
bug (a symmetric physical quantity handled asymmetrically) was found again by an independent second review.

`risk_ab/ac/bc` (a per-EDGE, not per-vertex, LODF-based quantity) must travel WITH its own edge token through
both the model (see `pairset.DeepSetsPairAware`) and preprocessing (a single pooled scaler here, not 3 separate
per-slot ones) for the whole pipeline to be genuinely invariant to which branch is called "a" vs "b" vs "c"."""
from __future__ import annotations

import itertools

import numpy as np

from ..grid import Grid
from ..lodf import compensation_det

PAIRS = [(0, 1), (0, 2), (1, 2)]  # (a,b),(a,c),(b,c) slot order, vertex positions 0,1,2 within an outage triple
PAIR_TO_SLOT = {p: i for i, p in enumerate(PAIRS)}
ALL_VERTEX_PERMS = list(itertools.permutations(range(3)))


def physical_feats(grid: Grid, LODF: np.ndarray, demand: np.ndarray, outage: tuple[int, int, int]) -> tuple[float, list[float]]:
    """F (island floor) and [risk_ab, risk_ac, risk_bc] for one outage triple. `compensation_det` is symmetric
    under swapping its two branch arguments, so each risk is a genuine unordered-edge quantity, in `outage`'s
    own (a,b),(a,c),(b,c) order (matching `PAIRS`)."""
    from ..physical_correction import island_floor  # local import: keeps this module's import surface minimal
    a, b, c = outage
    F = island_floor(grid, demand, outage)
    pairs = [(a, b), (a, c), (b, c)]
    risks = [min(1.0 / max(abs(compensation_det(LODF, p[0], p[1])), 1e-9), 1e4) for p in pairs]
    return F, risks


def edge_slot_perm(sigma: tuple[int, int, int]) -> list[int]:
    """sigma[i] = the OLD vertex index filling NEW position i. Returns, for each NEW edge slot k (in `PAIRS`
    order: ab, ac, bc), which OLD edge slot's data now belongs there. E.g. sigma=(1,2,0) ("bca": new-a=old-b,
    new-b=old-c, new-c=old-a): new edge (a,b)=(old1,old2)->old slot bc(2); new edge (a,c)=(old1,old0)->sorted
    (0,1)->old slot ab(0); new edge (b,c)=(old2,old0)->sorted(0,2)->old slot ac(1) => [2,0,1]."""
    return [PAIR_TO_SLOT[tuple(sorted((sigma[i], sigma[j])))] for i, j in PAIRS]


def column_perm_for_vertex_perm(sigma: tuple[int, int, int]) -> list[int]:
    """Full 15-column index permutation for X = [ya,yb,yc, Iab,Iac,Ibc, F, risk_ab,risk_ac,risk_bc, c1..c5]
    realizing the vertex relabeling sigma. Applying `X[:, column_perm_for_vertex_perm(sigma)]` to a row is
    mathematically equivalent to relabeling which physical branch is called "a"/"b"/"c" when the row was built
    -- correct IFF every labeling-dependent column is permuted, which is exactly what this function computes
    (F and c1..c5 are labeling-independent and stay fixed). Replaces manually-maintained permutation tuples,
    independently found error-prone twice already in this repo's history (see
    tests/test_deepsets_invariance.py's own account of a caught-and-fixed wrong tuple)."""
    es = edge_slot_perm(sigma)
    return list(sigma) + [3 + s for s in es] + [6] + [7 + s for s in es] + [10, 11, 12, 13, 14]


def fit_shared_risk_scaler(Xtr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ONE shared (mean,std) over all 3 risk slots (columns 7,8,9), pooled together -- NOT 3 separate per-slot
    statistics -- so a risk value's normalized value does not depend on which slot (ab/ac/bc) it lands in under
    relabeling. F (column 6) keeps its own separate statistic since it is never permuted. Train-only stats."""
    F_mean, F_std = float(Xtr[:, 6].mean()), float(Xtr[:, 6].std()) + 1e-8
    risk_pool = Xtr[:, 7:10].reshape(-1)
    risk_mean, risk_std = float(risk_pool.mean()), float(risk_pool.std()) + 1e-8
    phys_mean = np.array([F_mean, risk_mean, risk_mean, risk_mean], dtype=np.float64)
    phys_std = np.array([F_std, risk_std, risk_std, risk_std], dtype=np.float64)
    return phys_mean, phys_std


def apply_scaler(X: np.ndarray, phys_mean: np.ndarray, phys_std: np.ndarray) -> None:
    """In-place standardisation of columns 6:10 (F, risk_ab, risk_ac, risk_bc)."""
    X[:, 6:10] = (X[:, 6:10] - phys_mean) / phys_std


def build_row(op_id: int, outage: tuple[int, int, int], cv_idx_arr: np.ndarray, *, y1: dict, y2: dict,
              phys: tuple[float, list[float]], CV: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X, y, g2) for one (op_id, outage) block at the given control indices. `phys` = `physical_feats(...)`
    output for THIS outage (F, risks) -- passed in rather than recomputed so a caller can supply the SAME
    physical features for an outage's original and permuted forms in an invariance test. `y_true` is the
    already-selected label array for this (op_id, outage) at `cv_idx_arr` (label storage layout is caller/
    script-specific, so this stays a plain array parameter rather than a lookup callback)."""
    a, b, c = outage
    ya, yb, yc = y1[(op_id, a)][cv_idx_arr], y1[(op_id, b)][cv_idx_arr], y1[(op_id, c)][cv_idx_arr]
    pairs = [(min(a, b), max(a, b)), (min(a, c), max(a, c)), (min(b, c), max(b, c))]
    Iab, Iac, Ibc = (y2[(op_id, p)][cv_idx_arr] - y1[(op_id, p[0])][cv_idx_arr] - y1[(op_id, p[1])][cv_idx_arr] for p in pairs)
    F, risks = phys
    risks_scaled = [np.log1p(r) for r in risks]
    n = len(cv_idx_arr)
    phys_rep = np.tile([F, *risks_scaled], (n, 1))
    cvec = CV[cv_idx_arr]
    X = np.column_stack([ya, yb, yc, Iab, Iac, Ibc, phys_rep, cvec]).astype(np.float32)
    g2 = (ya + yb + yc + Iab + Iac + Ibc).astype(np.float32)
    y = y_true.astype(np.float32)
    return X, y, g2
