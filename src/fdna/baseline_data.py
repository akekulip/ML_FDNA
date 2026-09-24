"""Feature matrices and metadata for the tree baselines (no LP outputs are used as inputs)."""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import comm, spec
from .evalutil import check_spec_hash, in_subsample, op_metrics, tiebreak_key
from .features import control_features, post_outage_flows, raw_comm_states
from .grid import load_case30
from .opgen import nominal_rating

_G = load_case30()
_RATING = nominal_rating(_G, spec.RATING)
N_PHYS = _G.n_branch + 4


def _phys_chunk(args):
    demand, p0, keys = args
    out = np.zeros((len(keys), N_PHYS))
    for r, (i, b1, b2) in enumerate(keys):
        removed = tuple(b for b in (b1, b2) if b >= 0)
        ratio, struct = post_outage_flows(_G, demand[i], p0[i], removed, _RATING)
        out[r, : _G.n_branch] = ratio
        out[r, _G.n_branch:] = [ratio.max(), (ratio > 1).sum(), np.clip(ratio - 1, 0, None).sum(), struct]
    return out


def physics_table(data_dir: Path, lab: pd.DataFrame, ops) -> pd.DataFrame:
    """Post-outage DC-flow features per unique (op, b1, b2); cached next to the data."""
    import hashlib

    ops_sha = hashlib.sha1((data_dir / "ops.npz").read_bytes()).hexdigest()[:8]
    cache = data_dir / f"phys_cache_{spec.spec_hash()}_{ops_sha}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    keys = lab[["op", "b1", "b2"]].drop_duplicates().reset_index(drop=True)
    idx = {int(k): i for i, k in enumerate(ops["op_id"])}
    arr = np.array([(idx[o], b1, b2) for o, b1, b2 in keys.itertuples(index=False)])
    chunks = np.array_split(arr, 60)
    with Pool(30) as p:
        res = p.map(_phys_chunk, [(ops["demand"], ops["p0"], c) for c in chunks])
    tab = pd.concat([keys, pd.DataFrame(np.vstack(res), columns=[f"p{i}" for i in range(N_PHYS)])], axis=1)
    tab.to_parquet(cache)
    return tab


@dataclass
class Data:
    lab: pd.DataFrame
    F: dict
    y: np.ndarray
    key: np.ndarray
    tr: np.ndarray
    va: np.ndarray
    te: np.ndarray
    novel: np.ndarray  # control vector never seen in the (full) training failure sets
    fs_size: np.ndarray
    loss: np.ndarray  # complete remote-control loss


def load(data_dir="data", test_frac=0.25) -> Data:
    data_dir = Path(data_dir)
    check_spec_hash(data_dir)
    lab = pd.read_parquet(data_dir / "labels.parquet")
    ops = dict(np.load(data_dir / "ops.npz"))
    keep = (lab.split.values != "test") | in_subsample(lab, test_frac)  # stable, id-based
    ph = physics_table(data_dir, lab, ops)
    lab = lab[keep].reset_index(drop=True)
    ph = lab[["op", "b1", "b2"]].merge(ph, on=["op", "b1", "b2"], how="left")
    assert not ph.isna().any().any()
    n = len(lab)
    idx = {int(k): i for i, k in enumerate(ops["op_id"])}
    opi = lab.op.map(idx).values
    mh = np.zeros((n, _G.n_branch), np.float32)
    for c in ("b1", "b2"):
        v = lab[c].values
        m = v >= 0
        mh[np.flatnonzero(m), v[m]] = 1
    elec = np.hstack([ops["demand"][opi], ops["p0"][opi], mh, mh.sum(1, keepdims=True)]).astype(np.float32)
    all_fs = np.arange(len(spec.FAILURE_SETS))
    fsC, fsU, fsT = control_features(all_fs)
    fsR = raw_comm_states(all_fs)
    fs = lab.fs.values
    ctrl = np.hstack([fsC[fs], fsU[fs], fsT[fs]]).astype(np.float32)
    raw = fsR[fs].astype(np.float32)
    phys = ph[[f"p{i}" for i in range(N_PHYS)]].values.astype(np.float32)
    F = {"elec": elec, "elec+raw_comm": np.hstack([elec, raw]), "elec+ctrl": np.hstack([elec, ctrl]),
         "elec+phys": np.hstack([elec, phys]), "elec+phys+raw_comm": np.hstack([elec, phys, raw]),
         "elec+phys+ctrl": np.hstack([elec, phys, ctrl])}
    keyv = np.array([hash(tuple(r.round(6))) for r in fsC])
    train_keys = set(keyv[spec.fs_ids("train")])
    sizes = np.array([len(f) for f in spec.FAILURE_SETS])
    sp = lab.split.values
    return Data(lab=lab, F=F, y=lab.y.values, key=tiebreak_key(lab), tr=sp == "train", va=sp == "val", te=sp == "test",
                novel=~np.isin(keyv[fs], list(train_keys)), fs_size=sizes[fs], loss=fsC[fs].sum(1) == 0)


def strata(d):
    cell = d.lab.cell.values
    s = {c: cell == c for c in ("n1_seen", "n1_unseen", "n2_seen", "n2_unseen")}
    u = s["n2_unseen"]
    s["n2_unseen|familiar"], s["n2_unseen|novel"] = u & ~d.novel, u & d.novel
    for k in (2, 3, 4):
        s[f"n2_unseen|size{k}"] = u & (d.fs_size == k)
    for k in (2, 3, 4):
        s[f"n2_unseen|size{k}|novel"] = u & (d.fs_size == k) & d.novel
        s[f"n2_unseen|size{k}|familiar"] = u & (d.fs_size == k) & ~d.novel
    return {k: m & d.te for k, m in s.items()}


def evaluate(d, name, pred, seed, S):
    rows = []
    op = d.lab.op.values
    for sname, m in S.items():
        for o in np.unique(op[m]):
            mm = m & (op == o)
            rows.append(dict(model=name, seed=seed, stratum=sname, op=int(o), **op_metrics(d.y[mm], pred[mm], d.key[mm])))
    return rows


