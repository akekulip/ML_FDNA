"""Render development curves and a bounded, artifact-derived research report."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results/phase5/cost_quality_dev'))
    args = parser.parse_args()
    root = args.results
    data = pd.DataFrame(json.loads((root / 'summary.json').read_text()))
    gate = json.loads((root / 'gate.json').read_text())
    config = json.loads((root / 'config.json').read_text())
    comparisons = pd.DataFrame(json.loads((root / 'comparisons.json').read_text()))
    data.to_csv(root / 'curves.csv', index=False)
    methods = [('mc', 'Posterior MC / affordable enumeration', 'black', '-'),
               ('proxy_full', 'Full g2, no target queries', '#777777', '--'),
               ('cv_full', 'Full g2 + correction', '#0072B2', '-')]
    colors = ['#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#882255']
    methods += [(f'cv_sparse{n}', f'{n} control anchors + correction', c, '-') for n, c in zip(config['anchors'], colors)]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for row, cell in enumerate(('P1_v2b', 'P2_v2c')):
        for col, regime in enumerate(('cold', 'warm')):
            ax = axes[row, col]
            for name, label, color, style in methods:
                points = data[(data.cell == cell) & (data.regime == regime) & (data.method == name)].sort_values('budget')
                ax.plot(points.budget, points.rprecision, color=color, linestyle=style, marker='.', label=label)
            ax.set_xscale('symlog', linthresh=70)
            ax.set_xlim(0, max(config['budgets']) * 1.2)
            ax.set_xticks([0, 70, 280, 1120, 17920, 227328])
            ax.set_xticklabels(['0', '70', '280', '1.1k', '18k', '227k'])
            ax.set_title(f'{cell.split("_")[0]} — {regime} cache')
            ax.set_xlabel('Total scalar LP budget per operating point')
            if col == 0:
                ax.set_ylabel('Mean per-operating-point R-precision')
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9, bbox_to_anchor=(.5, -.015))
    fig.suptitle('Development data only: fixed cost–quality comparison')
    fig.tight_layout(rect=(0, .10, 1, .95))
    fig.savefig(root / 'cost_quality.svg', bbox_inches='tight')
    fig.savefig(root / 'cost_quality.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    lines = ['# Development cost–quality result', '',
        '**Exploratory only.** These are reused operating points 600–659 and the existing 70 triples.',
        'No fresh confirmatory block was opened. Primary endpoint: per-operating-point severe-case R-precision.', '',
        '| Cache regime | Frozen development gate | Selected method / budget |', '|---|---|---|']
    for regime in ('cold', 'warm'):
        selected = gate[regime]['selected']
        choice = f"{selected['method']} / {selected['budget']}" if selected else 'None'
        lines.append(f"| {regime} | {gate[regime]['status']} | {choice} |")
    for regime in ('cold', 'warm'):
        selected = gate[regime]['selected']
        if selected:
            lines += ['', f"### Selected {regime} candidate at {selected['budget']:,} labels/op", '',
                      '| Cell | Comparator | R-precision difference | One-sided 95% lower bound |',
                      '|---|---|---:|---:|']
            for r in selected['evidence']:
                lines.append(f"| {r['cell']} | {r['comparator']} | {r['mean']:+.4f} | {r['lo95']:+.4f} |")
    lines += ['', 'The gate requires a paired one-sided 95% lower bound above **0.01 absolute R-precision**',
        'against **both** posterior MC and the same proxy without target queries in **both** P1 and P2.',
        'Cold and warm are separate operational claims. A warm result assumes its explicitly priced lower-order cache already exists.',
        'Selection across development candidates is exploratory; these intervals are not post-selection confirmatory evidence.', '',
        '## Cost and evaluation contract', '',
        f"- Full lower-order cache: **{config['full_cache_cost']:,}** scalar LP labels/op.",
        '- This manifest needs 40 unique single outages and 182 unique pairs, each at 1,024 controls; unused single-outage labels are not charged.',
        f"- Full target enumeration: **{config['exact_full_grid_cost']:,}** scalar LP labels/op; posterior-support enumeration can be cheaper.",
        '- MC and CV use exact posterior-support enumeration when affordable, decided before accessing target labels.',
        '- Every cold curve charges unique lower-order and N-3 queries; warm curves charge target queries and report sunk construction separately.',
        '- Fixed draw counts retain sampling multiplicities. Repeated labels are evaluated once. Unused budget is reported.',
        '- R-precision severe counts and hidden labels are evaluator-only. Raw CV scores are ranked without clipping.',
        '- Undefined R-precision for zero-severe snapshots is counted explicitly, with common paired eligibility.',
        '- Each operating point/cell has one common observation. Ten target-sampling repeats measure MC variation, not new observations or topologies.',
        '- Infeasible methods have missing plotted points. Full metric and feasibility tables are in `summary.json` / `curves.csv`.',
        '- Curves show point estimates; paired uncertainty is in `comparisons.json`. Sparse warm caches assume the same observation-specific anchors already exist.',
        '- Legacy monotone-bound comparators retain a 16-query/candidate cap; surplus budget is not silently spent.', '',
        '## Paired differences at 1,120 labels/op', '',
        '| Regime | Cell | Method | Comparator | Difference | One-sided 95% lower bound |', '|---|---|---|---|---:|---:|']
    view = comparisons[(comparisons.budget == 1120) & comparisons.method.isin(['cv_full', 'cv_sparse2', 'cv_sparse64'])]
    for _, r in view.iterrows():
        lines.append(f"| {r.regime} | {r.cell} | {r.method} | {r.comparator} | {r['mean']:+.4f} | {r.lo95:+.4f} |")
    lines += ['', '## Secondary endpoints and limitations', '',
        'Probability-estimation MSE (raw and clipped), and recall/precision at 10%, 20%, and 40% are all retained.',
        'Saturated shortlist endpoints remain in the report; they cannot support a claim of incremental benefit.',
        'The posterior and labels are specific to the existing finite-grid IEEE-30 benchmark.',
        'Sparse interpolation is a surrogate, not a certified physical bound.', '',
        '## Verification and reproduction', '',
        'Artifact hashes, input/source hashes, and replay timing are in `provenance.json`.',
        'Complete prediction/query records are generated at `data_hik/cost_quality_dev/` and are not checked into Git.',
        'Run `.venv/bin/python scripts/p5_cost_quality.py` for a serial full reproduction.',
        'Run `.venv/bin/python scripts/p5_cost_quality_replay.py` to reconstruct metrics, audit costs, and replay the frozen real-LP timing workload.',
        'The optional figure renderer needs pandas and Matplotlib. In this environment, both already exist in system Python:',
        '`MPLCONFIGDIR=/tmp/fdna-matplotlib python3 scripts/p5_cost_quality_report.py`.',
        'No new project dependencies were added.', '']
    for filename, title in (('trace_audit.json', 'Independent query-accounting audit'), ('lp_replay.json', 'Measured LP timing')):
        if (root / filename).exists():
            evidence = json.loads((root / filename).read_text())
            lines += [f'### {title}', '']
            if filename == 'trace_audit.json':
                lines += [f"Audited {evidence['metrics_records']:,} policy evaluations and independently reconstructed all {evidence['metrics_reconstructed']:,} feasible rows; failures: {len(evidence['failures'])}.", '']
            else:
                lines += [f"Replayed {evidence['total_unique_lp_queries_charged_with_per_scenario_cache_reset']:,} scalar LP queries; summed solve time: {evidence['lp_wall_s']:.2f} seconds.",
                          f"Maximum absolute cached-label deviation: {evidence['max_label_deviation']:.3g}.",
                          evidence['disclaimer'], '']
    hardware_path = root / 'HARDWARE_VERIFICATION.json'
    if hardware_path.exists():
        hardware = json.loads(hardware_path.read_text())
        if hardware['selection']['complete_frozen_scope']:
            lines += ['### Actual CPU solver verification', '',
                      'The frozen warm candidate and both baselines were rerun with real `ScenarioLP` target queries',
                      'on all 60 development operating points, both cells, and ten sampling repeats.',
                      f"Actual unique scalar solves: {hardware['counts']['unique_physical_solves_within_op_workers']:,}; "
                      f"maximum label deviation: {hardware['maxima']['event_abs_error']:.3g}.",
                      f"Maximum score/metric differences: {hardware['maxima']['score_abs_diff']:.3g} / "
                      f"{hardware['maxima']['metric_abs_diff']:.3g}; failures: {hardware['mismatches']['failure_count']}.",
                      'Realized severe labels were also re-solved. The warm lower-order cache remains an input,',
                      'and probability-MSE truth remains the frozen exact-posterior snapshot. This verifies numerical',
                      'reproduction of the selected warm result; it is not a new-data confirmation or a physical-grid experiment.',
                      'Full scope, accounting, timing, and provenance: `HARDWARE_VERIFICATION.json`.', '']
    if all(gate[r]['selected'] is None for r in ('cold', 'warm')):
        lines += ['## Decision', '', '**No-go for fresh confirmation under this gate.** Preserve the negative result.',
                  'The tested control-variate/sparse-proxy family did not clear the frozen practical screening margin',
                  'against both baselines in both cells. Variance reduction alone does not overturn that outcome.', '']
    else:
        lines += ['## Decision', '', 'The selected regime-specific development candidate is recorded in `gate.json`.',
                  'It is not a confirmed result. The warm result assumes the full lower-order cache already exists;',
                  'building it costs more than enumerating these 70 target outages directly. There is no cold-start promotion.',
                  'The proposed follow-up protocol and sample-size rationale are in `CONFIRMATION_PROTOCOL.md`; it is not yet registered.',
                  'no fresh data are opened by this development run or its hardware verification.', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
