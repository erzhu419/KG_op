"""Paired attribution from the two frozen joint-prior forecast packages."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save


def run():
    spec = read('performance/manifests/legacy_joint_population_f_inner_joint_compression_tail_attribution_execution_v1.json')
    result = read(Path(spec['output']) / 'results.json'); reference = read(spec['tail_reference_results'])
    assert result['diagnostic_interpreted'] and reference['diagnostic_interpreted']
    pairs = []
    for row in result['rows']:
        old = next(r for r in reference['rows'] if (r['layout'], r['query_task'], r['arm']) == (row['layout'], row['query_task'], row['arm']))
        assert (row['paid_prefix'], row['scoring_sites']) == (old['paid_prefix'], old['scoring_sites'])
        pairs.append({k: row[k] for k in ('layout', 'query_task', 'arm')} | {'without_minus_with_tail':
            {m: row['scores'][m] - old['scores'][m] for m in spec['metrics']}})
    assert len(pairs) == 20
    result['tail_contrasts'] = pairs
    result['tail_summary'] = {layout: {arm: {m: float(np.mean([p['without_minus_with_tail'][m] for p in pairs
        if p['layout'] == layout and p['arm'] == arm])) for m in spec['metrics']} for arm in spec['arms']} for layout in spec['layouts']}
    save(ROOT / spec['output'] / 'results.json', result)


if __name__ == '__main__':
    run()
