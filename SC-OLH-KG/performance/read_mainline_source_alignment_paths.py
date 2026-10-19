"""Read sealed Stage94 summaries and split C by the existing G support."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from performance.mainline_source_alignment_summary import AFTER, POINT, PERMUTED_AFTER, PERMUTED_POINT
from performance.run_mainline_legacy_backend_bridge import ROOT, attempt, read, save
from performance.summarize_mainline_source_alignment import ZERO
from performance.summarize_mainline_two_comparison_development import estimate


CORE = (AFTER, POINT, PERMUTED_AFTER, PERMUTED_POINT)
COMPARISONS = ('real_AFTER_minus_permuted_AFTER', 'source_by_propagation_interaction')


def support_diagnostic(comparison):
    rows = [dict(seed=row['seed'], domain=row['domain'], C=row['delta']['C'], G=row['delta']['G'])
            for row in comparison['rows']]
    for row in rows:
        row['C_on_G_support'] = row['C'] if row['G'] is not None else None
        row['C_without_G_support'] = row['C'] if row['G'] is None else None
    fields = ('C', 'C_on_G_support', 'C_without_G_support')
    return dict(name=comparison['name'], rows=rows,
        estimates={key: estimate([row[key] for row in rows]) for key in fields},
        by_family=[dict(domain=domain, estimates={key: estimate([
            row[key] for row in rows if row['domain'] == domain]) for key in fields})
            for domain in dict.fromkeys(row['domain'] for row in rows)],
        scope='Endpoint support only. All registered geometries remain, with excluded or missing values null.')


def focus_projection(task):
    streams = []
    for stream in task['streams']:
        item = {key: value for key, value in stream.items() if key not in (
            'outcomes', 'queries', 'policy_paths', 'policy_terminal_paths', 'query_comparisons',
            'comparisons', 'multi_safe_events', 'initialization_contrast')}
        for key in ('outcomes', 'queries', 'policy_terminal_paths', 'multi_safe_events'):
            item[key] = [row for row in stream.get(key, []) if row['arm'] in CORE]
        item['policy_paths'] = [row for row in stream.get('policy_paths', [])
                                if row.get('history_arm', row.get('arm')) in CORE]
        item['query_comparisons'] = [{key: value for key, value in row.items()
            if key != 'common_policy_changes'} for row in stream.get('query_comparisons', [])
            if row.get('left') in CORE and row.get('right') in CORE]
        item['comparisons'] = [{key: value for key, value in row.items() if key != 'deployments'}
                              for row in stream.get('comparisons', [])]
        streams.append(item)
    return dict(seed=task['seed'], domain=task['domain'], status=task['status'], streams=streams,
                source_result_file=task.get('source_result_file'), limits=task.get('limits'))


def readout(plan, *, root=ROOT):
    root = Path(root)
    folder = root / plan['output_dir']
    output = folder / 'cluster_control/readout/stage94_attempt1'
    target = output / 'result.json'
    if target.exists():
        return read(target)
    completion = read(folder / 'completion.json')
    if completion['status'] not in ('complete', 'complete_with_analysis_failures'):
        raise ValueError('The existing Stage94 collection must finish before readout')
    with attempt(output, 'read-saved-source-alignment-paths-support') as cost:
        cost.update(ZERO)
        summary = read(folder / 'summary.json')
        headline = read(folder / 'headline.json')
        chosen = [row for row in summary['independent']['comparisons'] if row['name'] in COMPARISONS]
        if len(chosen) != 2 or any(len(row['rows']) != len(plan['tasks']) for row in chosen):
            raise ValueError('Retain both registered comparisons and every geometry in support masks')
        focus = [focus_projection(read(folder / 'cases' / f'seed_{seed}' / 'result.json'))
                 for seed in plan['focus_seeds']]
        save(output / 'headline.json', headline)
        save(output / 'focus.json', dict(tasks=focus, scope=plan['focus_scope'], new_work=dict(ZERO)))
        result = dict(status='complete', created_utc=datetime.now(timezone.utc).isoformat(),
            baseline_support_diagnostic=[support_diagnostic(row) for row in chosen],
            source_file_sizes={name: (folder / name).stat().st_size for name in (
                'summary.json', 'headline.json', 'completion.json')},
            completion={key: value for key, value in completion.items() if key != 'attempts'},
            compact_file=str((output / 'headline.json').relative_to(root)),
            focus_file=str((output / 'focus.json').relative_to(root)),
            new_work=dict(ZERO),
            scope='Existing Stage94 statistics and focused saved paths; only C support-mask estimates are new.')
        save(target, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='performance/manifests/mainline_source_alignment_paths_v1.json')
    args = parser.parse_args()
    result = readout(read(ROOT / args.manifest))
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
