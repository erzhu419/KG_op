"""Select compact saved summary fields for review; no recomputation."""
import json
from pathlib import Path
import time


ROOT = Path.cwd()
OUT = ROOT / 'paper_artifacts/mainline_decision_path_attribution_v1'
AFTER = 'legacy_role_after_init_point_only'


def without_vectors(value):
    if isinstance(value, dict):
        return {k: without_vectors(v) for k, v in value.items()
                if k not in ('point', 'policies', 'query_comparisons')
                and not (isinstance(v, list) and len(v) == 1000
                         and all(isinstance(x, (int, float)) for x in v))}
    if isinstance(value, list):
        return [without_vectors(v) for v in value]
    return value


def main():
    cpu = time.process_time()
    summary = json.loads((OUT / 'summary.json').read_text())
    after = next(g for g in summary['groups'] if g['arm'] == AFTER and g['protocol'] == 'old')
    overview = {k: summary[k] for k in ('status', 'statistical_unit', 'registered_geometries',
        'registered_streams', 'registered_task_streams', 'registered_controls', 'registered_outcomes',
        'analysis_failures', 'outcome_failures', 'false_certificates', 'costs', 'limits')}
    overview['AFTER_old_group'] = after
    overview['controls'] = [{**{k: g[k] for k in ('arm', 'protocol', 'outcome_status_counts',
        'analysis_status_counts', 'certificate_categories', 'policy_path_counts', 'nesting_issue_cases')},
        'metrics': {m: {k: g['metrics'][m][k] for k in ('task_stream_description', 'geometry_statistics')}
                    for m in ('C', 'G')}} for g in summary['groups']]
    overview['primary'] = {k: summary['primary'][k] for k in ('left', 'right', 'protocol', 'metrics',
        'discovery_quality_gain_difference', 'by_family')}
    overview['AFTER_old_task_streams'] = [{k: v for k, v in row.items()
        if k not in ('scientific_reported_deployed', 'scientific_certified_deployed')}
        for row in summary['cells'] if row['arm'] == AFTER and row['protocol'] == 'old']
    for row in overview['AFTER_old_task_streams']:
        row['discovery'] = {k: v for k, v in row['discovery'].items() if k != 'query_chains'}
    focal = []
    for task in summary['focal_cases']:
        entry = {k: task[k] for k in ('seed', 'stream', 'domain', 'status', 'baseline', 'coverage')}
        entry['arms'] = []
        for arm in task['arms']:
            selected = {protocol.get('deployed_policy_id') for protocol in arm['protocols'].values()}
            part = {k: arm[k] for k in ('arm', 'history_arm', 'terminal_backend', 'search_status',
                'sets', 'discovery', 'protocols', 'recording')}
            part['discovery'] = {k: v for k, v in part['discovery'].items() if k != 'query_chains'}
            part['observed'] = [row for row in arm['observed'] if row['improving_safe'] is True
                or row['policy_id'] in selected or any(v is not None and v > 1e-12
                    for v in row['quality_minus_deployed'].values())]
            entry['arms'].append(part)
        focal.append(entry)
    outputs = {'review_summary.json': overview, 'focal_extract.json': focal}
    for name, value in outputs.items():
        (OUT / name).write_text(json.dumps(without_vectors(value), indent=2) + '\n')
    record = dict(status='complete', cpu_seconds=time.process_time() - cpu,
        files={name: (OUT / name).stat().st_size for name in outputs},
        new_scientific_calls=0, new_truth_calls=0, new_fits=0, new_posterior_restorations=0)
    (OUT / 'cluster_control/extract_account.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
