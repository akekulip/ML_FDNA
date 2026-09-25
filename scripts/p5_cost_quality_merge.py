"""Merge disjoint development shards without rewriting their gzip trace members."""
import argparse
import gzip
import json
import shutil
from pathlib import Path

if __package__:
    from .p5_cost_quality import sha256, summarize, write_json
else:
    from p5_cost_quality import sha256, summarize, write_json
from fdna.cost_quality import select_gate


REPO_ROOT = Path(__file__).resolve().parents[1]


def verify_artifacts(part, provenance):
    """Verify the supplied bundle, even if its original absolute path moved."""
    names = set()
    for filename, digest in provenance['artifacts'].items():
        name = Path(filename).name
        if name not in {'traces.jsonl.gz', 'metrics.jsonl.gz'} or name in names:
            raise ValueError(f'unexpected or duplicate artifact: {filename}')
        names.add(name)
        local = part / 'traces' / name
        if sha256(local) != digest:
            raise ValueError(f'artifact checksum mismatch: {local}')
    if names != {'traces.jsonl.gz', 'metrics.jsonl.gz'}:
        raise ValueError('missing trace or metrics artifact')


def verify_merge_sources(sources, repo_root=REPO_ROOT):
    """Summary/gate code must still match the implementation frozen by shards."""
    required = ('scripts/p5_cost_quality.py', 'src/fdna/cost_quality.py')
    for relative in required:
        candidates = [digest for filename, digest in sources.items()
                      if Path(filename).as_posix() == relative
                      or Path(filename).as_posix().endswith('/' + relative)]
        if len(candidates) != 1 or sha256(repo_root / relative) != candidates[0]:
            raise ValueError(f'merge-time source differs from frozen shard: {relative}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('parts', nargs='+', type=Path, help='shard directories containing results/ and traces/')
    parser.add_argument('--output-dir', type=Path, default=Path('results/phase5/cost_quality_dev'))
    parser.add_argument('--trace-dir', type=Path, default=Path('data_hik/cost_quality_dev'))
    args = parser.parse_args()
    configs, provenance = [], []
    ops = []
    for part in args.parts:
        config = json.loads((part / 'results/config.json').read_text())
        prov = json.loads((part / 'results/provenance.json').read_text())
        if set(ops) & set(config['op_ids']):
            raise ValueError('overlapping operating-point shards')
        ops.extend(config['op_ids'])
        comparable = {k: v for k, v in config.items() if k != 'op_ids'}
        if configs and comparable != {k: v for k, v in configs[0].items() if k != 'op_ids'}:
            raise ValueError('shard protocols differ')
        if provenance and (prov['inputs'] != provenance[0]['inputs'] or prov['sources'] != provenance[0]['sources']):
            raise ValueError('shard code/data provenance differs')
        verify_artifacts(part, prov)
        if sha256(part / 'results/config.json') != prov['config_sha256']:
            raise ValueError(f'config checksum mismatch: {part}')
        configs.append(config)
        provenance.append(prov)
    if sorted(ops) != list(range(600, 660)):
        raise ValueError('complete development merge requires exactly operating points 600–659')
    verify_merge_sources(provenance[0]['sources'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    for filename in ('traces.jsonl.gz', 'metrics.jsonl.gz'):
        with open(args.trace_dir / filename, 'wb') as target:
            for part in args.parts:
                with open(part / 'traces' / filename, 'rb') as source:
                    shutil.copyfileobj(source, target)
    with gzip.open(args.trace_dir / 'metrics.jsonl.gz', 'rt') as f:
        records = [json.loads(line) for line in f]
    summaries, comparisons = summarize(records)
    config = dict(configs[0], op_ids=sorted(ops))
    gate = select_gate(comparisons, config['practical_margin'])
    gate.update(scope='development selection only; no fresh-data access', complete_development_run=True,
                regime_claims='independent cold/warm gates; warm-only success requires a preexisting proxy cache')
    write_json(args.output_dir / 'config.json', config)
    write_json(args.output_dir / 'summary.json', summaries)
    write_json(args.output_dir / 'comparisons.json', comparisons)
    write_json(args.output_dir / 'gate.json', gate)
    write_json(args.output_dir / 'provenance.json', dict(head=provenance[0]['head'],
        inputs=provenance[0]['inputs'], sources=provenance[0]['sources'],
        merge_source_sha256=sha256(__file__), config_sha256=sha256(args.output_dir / 'config.json'),
        artifacts={str(args.trace_dir / name): sha256(args.trace_dir / name) for name in ('traces.jsonl.gz', 'metrics.jsonl.gz')},
        trace_records=sum(p['trace_records'] for p in provenance),
        parallel_replay_wall_s=max(p['finished_epoch'] for p in provenance)-min(p['started_epoch'] for p in provenance),
        summed_worker_replay_wall_s=sum(p['replay_total_wall_s'] for p in provenance),
        shards=[str(p) for p in args.parts], note='cached replay and accounting time; actual LP times are reported separately'))
    print(json.dumps(gate, indent=2))


if __name__ == '__main__':
    main()
