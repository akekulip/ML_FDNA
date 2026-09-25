"""Development-only screening at enforced total LP budgets; no reserve access.

Run from repository root. Bulk reproducibility traces are generated under
data_hik/cost_quality_dev; compact summaries/provenance are committed separately.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from fdna import spec, v2, v2data
from fdna.cost_quality import paired_summary, screening_metrics, select_gate
from fdna.evalutil import scenario_uniform
from fdna.query_budget import BudgetedOracle, bound_probability, estimate_probability, g2_proxy


CELLS = {'P1_v2b': (.7, .3, None), 'P2_v2c': (.3, .2, v2.COVERAGE_SPARSE)}
ANCHORS = (2, 4, 8, 16, 32, 64)


def execute_policy(kind, backend, op_id, outages, CV, pc, *, budget,
                   construction_cost, seed, proxy=None, floors=None):
    """Policy boundary: only posterior, paid query callback, and known proxy.

    Allocation is equal across candidates and fixed before seeing any target
    label. Cache savings are reported, never used as a label-dependent stop rule.
    """
    start = time.perf_counter()
    n = len(outages)
    remaining = budget - construction_cost
    support = np.flatnonzero(pc > 0)
    draws = min(max(0, remaining // n), len(CV))
    if kind in ('random_bounds', 'guided_bounds'):
        draws = min(draws, 16)  # retain the existing bound policies' declared maximum cap
    feasible = remaining >= 0 and (kind in ('proxy', 'random_bounds', 'guided_bounds') or draws > 0)
    if kind == 'exact':
        feasible = remaining >= n * len(CV)
    result = dict(feasible=bool(feasible), target_queries=0, draws_per_candidate=draws,
                  scores=None, events=[], mode=kind, replay_wall_s=0.)
    if not feasible:
        return result
    oracle = BudgetedOracle(backend, int(remaining))
    scores = []
    mode = 'posterior_support_enumeration' if kind in ('mc', 'cv') and remaining >= n * len(support) else kind
    for i, outage in enumerate(outages):
        rng = np.random.default_rng([op_id, *outage, seed, 99])
        h0 = None if proxy is None else proxy[i]
        if kind == 'proxy':
            score = float(pc @ (h0 > spec.SEVERE))
        elif mode in ('exact', 'posterior_support_enumeration'):
            indices = range(len(CV)) if kind == 'exact' else support
            score = sum(pc[j] * (oracle.query(op_id, outage, int(j)) > spec.SEVERE) for j in indices)
        elif kind in ('mc', 'cv'):
            score = estimate_probability(oracle, op_id, outage, pc, draws, rng,
                                         proxy=h0 if kind == 'cv' else None, tau=spec.SEVERE)
        elif kind in ('random_bounds', 'guided_bounds'):
            score = bound_probability(oracle, op_id, outage, CV, pc, draws,
                                      floor=0. if floors is None else floors[i],
                                      proxy=h0 if kind == 'guided_bounds' else None,
                                      rng=rng, tau=spec.SEVERE)
        else:
            raise ValueError(f'unknown policy {kind}')
        scores.append(float(score))
    assert oracle.spent + construction_cost <= budget
    result.update(scores=scores, events=oracle.events, target_queries=oracle.spent,
                  mode=mode, replay_wall_s=time.perf_counter() - start)
    return result


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def summarize(records):
    frame = pd.DataFrame(records)
    summaries, comparisons = [], []
    metrics = ['rprecision', 'probability_mse', 'clipped_probability_mse'] + [f'{m}_{p}' for m in ('recall', 'precision') for p in (10, 20, 40)]
    groups = dict(tuple(frame.groupby(['regime', 'cell', 'budget', 'method'], sort=True)))
    for keys, group in groups.items():
        reg, cell, budget, method = keys
        good = group[group.feasible]
        entry = dict(regime=reg, cell=cell, budget=int(budget), method=method,
                     feasible_evaluations=len(good), total_evaluations=len(group),
                     mean_total_queries=float(good.total_queries.mean()) if len(good) else None,
                     mean_target_queries=float(good.target_queries.mean()) if len(good) else None,
                     construction_queries=int(group.construction_queries.iloc[0]),
                     no_severe_evaluations=int((good.n_severe == 0).sum()))
        for metric in metrics:
            vals = good[metric].dropna()
            entry[metric] = float(vals.mean()) if len(vals) else None
        summaries.append(entry)
        if not method.startswith('cv_') or not len(good):
            continue
        for comparator in ('mc', method.replace('cv_', 'proxy_', 1)):
            base = groups[(reg, cell, budget, comparator)]
            base = base[base.feasible]
            pairs = good.merge(base, on=['op_id', 'replicate'], suffixes=('_a', '_b'))
            pairs = pairs.dropna(subset=['rprecision_a', 'rprecision_b'])
            expected = good.rprecision.notna().sum()
            if len(pairs) != expected or not len(pairs):
                continue
            pairs['difference'] = pairs.rprecision_a - pairs.rprecision_b
            matrix = pairs.pivot(index='op_id', columns='replicate', values='difference').to_numpy()
            if not np.isfinite(matrix).all():
                raise ValueError('incomplete paired replicate matrix')
            comparisons.append(dict(regime=reg, cell=cell, budget=int(budget), method=method,
                                    comparator=comparator, **paired_summary(matrix)))
    return summaries, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ops', type=int, default=60, help='prefix of already-open development ops')
    parser.add_argument('--op-start', type=int, default=600, help='first already-open op for independent shards')
    parser.add_argument('--replicates', type=int, default=10, help='paired target-sampling repeats per fixed observation')
    parser.add_argument('--budgets', type=int, nargs='+')
    parser.add_argument('--output-dir', type=Path, default=Path('results/phase5/cost_quality_dev'))
    parser.add_argument('--trace-dir', type=Path, default=Path('data_hik/cost_quality_dev'))
    args = parser.parse_args()
    if not 1 <= args.ops <= 60 or args.replicates < 1:
        parser.error('ops must be 1..60; replicates must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path('data_hik/manifest_k3_confirm.json')
    manifest = json.loads(manifest_path.read_text())
    ops = [op for op in sorted(manifest['op_ids']) if op >= args.op_start][:args.ops]
    if len(ops) != args.ops:
        parser.error('requested operating points exceed the development manifest')
    if any(op < 600 or op > 659 for op in ops):
        raise ValueError('this script is restricted to the already-open 600–659 development block')
    outages = sorted(tuple(o) for o in manifest['outage_sets'])
    tables = {}
    input_paths = [manifest_path, Path('configs/spec.json')]
    CV = None
    for k in (1, 2, 3):
        path = Path(f'data_hik/n{k}_confirm.npz')
        input_paths.append(path)
        with np.load(path) as data:
            if CV is None:
                CV = data['CV'].copy()
            elif not np.array_equal(CV, data['CV']):
                raise ValueError('label tables disagree on control-grid ordering')
            for op, out, values in zip(data['op_ids'], data['outages'], data['V']):
                if int(op) in ops:
                    if not np.isfinite(values).all():
                        raise ValueError('nonfinite label: refuse to drop scenarios')
                    tables[(int(op), tuple(int(x) for x in out[:k]))] = values.astype(float)

    def backend(op, outage, control):
        return tables[(op, outage)][control]

    singles = {b for outage in outages for b in outage}
    pairs = {p for outage in outages for p in combinations(outage, 2)}
    lower_per_control = len(singles) + len(pairs)
    full_cost = lower_per_control * len(CV)
    budgets = sorted(set(args.budgets or ([0] + [len(outages) * 2 ** i for i in range(11)] + [full_cost, full_cost + len(outages) * 16])))
    if budgets[0] < 0:
        parser.error('budgets must be nonnegative')
    config = dict(scope='exploratory development only; reused operating points and outages',
                  op_ids=ops, n_outages=len(outages), n_controls=len(CV), replicates=args.replicates,
                  budgets=budgets, anchors=list(ANCHORS), primary='per-op severe-case R-precision',
                  practical_margin=.01, ranking='raw unbounded CV score; stable candidate key',
                  warm_cache='method-specific lower-order proxy cache only; no target labels initially',
                  cold_cache='all unique construction labels plus target labels charged',
                  primary_missing='no-severe snapshots explicitly undefined; common eligible-op mask',
                  allocation='equal fixed draws per candidate, floor(remaining budget / candidate count), capped at control-grid size',
                  mc='MC and CV enumerate posterior support when affordable before target access',
                  bound_policies='legacy secondary comparators capped at 16 queries/candidate; unused budget reported',
                  observations='one common observation/hidden state per op/cell, identical to adaptive_query_v3',
                  full_cache_cost=full_cost, exact_full_grid_cost=len(outages)*len(CV),
                  cache_reuse='no sharing of target queries across policies, budgets, reps, cells, or ops',
                  selection='smallest budget with paired one-sided 95% lower bound >0.01 against MC and own zero-query proxy in both cells')
    write_json(args.output_dir / 'config.json', config)  # before examining policy outcomes
    source_paths = [Path(__file__), Path('src/fdna/query_budget.py'), Path('src/fdna/cost_quality.py'),
                    Path('src/fdna/adaptive_query.py'), Path('src/fdna/evalutil.py'), Path('src/fdna/v2.py'),
                    Path('src/fdna/v2data.py'), Path('src/fdna/lp.py'), Path('src/fdna/grid.py'),
                    Path('src/fdna/opgen.py'), Path('src/fdna/spec.py')]
    provenance = dict(head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                      inputs={str(p): sha256(p) for p in input_paths},
                      sources={str(p): sha256(p) for p in source_paths},
                      config_sha256=sha256(args.output_dir / 'config.json'))
    world = v2.build_world()
    if not np.array_equal(CV, world.CV):
        raise ValueError('posterior and label-table control grids differ')
    from fdna.dataset import G, RATING
    from fdna.opgen import sample_op
    from fdna.physical_correction import island_floor
    records = []
    trace_path = args.trace_dir / 'traces.jsonl.gz'
    trace_id = (args.op_start - 600) * 1_000_000
    started = time.perf_counter()
    started_epoch = time.time()
    with gzip.open(trace_path, 'wt', compresslevel=3) as trace:
        def save_trace(value):
            nonlocal trace_id
            idx = trace_id
            trace.write(json.dumps(dict(trace_id=idx, **value), separators=(',', ':'), allow_nan=False) + '\n')
            trace_id += 1
            return idx

        for op in ops:
            demands = sample_op(G, RATING, np.random.default_rng(op)).demand
            floors = np.array([island_floor(G, demands, o) for o in outages])
            truth_grid = np.array([tables[(op, o)] for o in outages])
            for cell_index, (cell, (q, stale, coverage)) in enumerate(CELLS.items(), 1):
                rng = np.random.default_rng([op, cell_index])
                state = v2.sample_states(world, rng, 1)
                obs = v2.emit(world, state, q, stale, rng, coverage)
                pc = v2data.oracle_features(world, obs, stale)[0][0].astype(float)
                pc /= pc.sum()
                labels = truth_grid[:, int(world.cidx[state][0])] > spec.SEVERE
                exact_probability = (truth_grid > spec.SEVERE) @ pc
                keys = scenario_uniform(np.full(len(outages), op), np.array(outages)[:, 0],
                                        np.array(outages)[:, 1], np.array(outages)[:, 2], salt=17)
                snapshot_id = save_trace(dict(kind='evaluation_snapshot', op_id=op, cell=cell,
                    outages=outages, observation=obs.tolist(), posterior=pc.tolist(),
                    labels=labels.tolist(), exact_probability=exact_probability.tolist(), keys=keys.tolist()))
                proxies = {}
                for suffix, count in [('full', None)] + [(f'sparse{m}', m) for m in ANCHORS]:
                    start = time.perf_counter()
                    oracle = BudgetedOracle(backend, full_cost if count is None else lower_per_control * count)
                    grid, anchor_indices = g2_proxy(oracle, op, outages, CV, pc, count)
                    assert oracle.spent == lower_per_control * len(anchor_indices)
                    elapsed = time.perf_counter() - start
                    lower_id = save_trace(dict(kind='construction', op_id=op, cell=cell, proxy=suffix,
                        anchors=anchor_indices.tolist(), spent=oracle.spent, events=oracle.events,
                        replay_wall_s=elapsed))
                    proxies[suffix] = (grid, oracle.spent, lower_id, elapsed)
                methods = [('mc', 'mc', None), ('exact', 'exact', None),
                           ('random_bounds', 'random_bounds', None), ('guided_bounds', 'guided_bounds', 'full')]
                for suffix in proxies:
                    methods += [(f'proxy_{suffix}', 'proxy', suffix), (f'cv_{suffix}', 'cv', suffix)]
                # Identical fixed-draw computations across cold/warm budgets reuse their
                # recorded execution, never their target cache or query charge.
                executions = {}
                for regime in ('cold', 'warm'):
                    for budget in budgets:
                        for name, kind, suffix in methods:
                            grid, construction, lower_id, prep_time = proxies[suffix] if suffix else (None, 0, None, 0.)
                            charged_prep = construction if regime == 'cold' else 0
                            for rep in range(args.replicates):
                                seed = cell_index * 100 + rep
                                remaining = budget - charged_prep
                                affordable = remaining >= 0
                                enum = kind in ('mc', 'cv') and remaining >= len(outages) * np.count_nonzero(pc)
                                effective_draws = min(max(0, remaining // len(outages)), len(CV))
                                if kind in ('random_bounds', 'guided_bounds'):
                                    effective_draws = min(effective_draws, 16)
                                execution_key = ('support_enumeration' if enum else name,
                                    affordable, 'enumerated' if enum else effective_draws,
                                    rep if kind in ('mc', 'cv', 'random_bounds') and not enum else 0)
                                if execution_key not in executions:
                                    result = execute_policy(kind, backend, op, outages, CV, pc, budget=budget,
                                        construction_cost=charged_prep, seed=seed, proxy=grid, floors=floors)
                                    execution_id = save_trace(dict(kind='execution', op_id=op, cell=cell,
                                        payload_only=True, method=None if enum else name,
                                        sampling_seed=None if enum else seed,
                                        **result))
                                    result.pop('events')  # bulky query log is on disk, not kept for every budget
                                    executions[execution_key] = (result, execution_id)
                                result, execution_id = executions[execution_key]
                                row = dict(regime=regime, cell=cell, op_id=op, replicate=rep, budget=budget,
                                    method=name, feasible=result['feasible'], target_queries=result['target_queries'],
                                    construction_queries=construction, charged_construction_queries=charged_prep,
                                    total_queries=charged_prep + result['target_queries'] if result['feasible'] else None,
                                    unused_budget=remaining-result['target_queries'] if result['feasible'] else None,
                                    replay_wall_s=result['replay_wall_s'], construction_replay_wall_s=prep_time,
                                    execution_trace=execution_id, construction_trace=lower_id, snapshot_trace=snapshot_id)
                                if result['feasible']:
                                    row.update(screening_metrics(labels, result['scores'], keys, exact_probability))
                                records.append(row)
            print(f'op {op}: {len(records)} evaluations, {time.perf_counter()-started:.1f}s', flush=True)
    records_path = args.trace_dir / 'metrics.jsonl.gz'
    with gzip.open(records_path, 'wt', compresslevel=3) as f:
        for row in records:
            f.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    summaries, comparisons = summarize(records)
    gate = select_gate(comparisons)
    gate['scope'] = 'development selection only; no confirmatory inference or fresh-data access'
    gate['regime_claims'] = 'independent gates: a warm-only candidate supports ONLY preexisting-cache operation, never cold-start efficiency'
    gate['complete_development_run'] = args.ops == 60 and args.replicates >= 10
    write_json(args.output_dir / 'summary.json', summaries)
    write_json(args.output_dir / 'comparisons.json', comparisons)
    write_json(args.output_dir / 'gate.json', gate)
    provenance.update(artifacts={str(p): sha256(p) for p in (trace_path, records_path)},
                      replay_total_wall_s=time.perf_counter()-started,
                      started_epoch=started_epoch, finished_epoch=time.time(),
                      trace_records=trace_id - (args.op_start - 600) * 1_000_000,
                      note='replay wall time includes cached query-accounting and trace IO; not LP solve time')
    write_json(args.output_dir / 'provenance.json', provenance)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == '__main__':
    main()
