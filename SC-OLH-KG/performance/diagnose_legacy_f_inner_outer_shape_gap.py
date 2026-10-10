"""Cached level/shape attribution of the inner-versus-outer prediction gap."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_outer_shape_gap_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION)
    inner = read(spec['inner_results']); outer = read(spec['outer_results']); previous = read(spec['outer_reference_breakdown'])
    assert inner['all_ten_inner_folds_complete'] and outer['scientific_gate_interpreted']
    coverage, _ = source_paths(read(outer['execution']['source_context_execution'])); rows = []
    for row in outer['rows']:
        layout, query, design = row['layout'], row['deleted_task'], row['design']
        # This label block already scored all frozen Stage151/158 forecasts.
        labels = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')['source_tasks'][query]
        assert row['scoring_sites'] == coverage['layouts'][layout][query]
        observed = np.asarray(labels['response'])[:, 0]; candidate = decompose(row['prediction']['mean'], observed)
        np.testing.assert_allclose(candidate['mse'], row['scores']['f_mse'], atol=1e-12, rtol=1e-10)
        saved = next(r for r in previous['rows'] if r['layout'] == layout)['breakdown']
        comparisons = {}
        for arm in ('source_variance_mixture', 'source_gp'):
            reference = saved[f'{arm}/{design}']
            difference = {key: candidate[key]-reference[key] for key in ('mse', 'level_mse', 'centered_shape_mse')}
            np.testing.assert_allclose(difference['mse'], difference['level_mse']+difference['centered_shape_mse'], atol=1e-12, rtol=1e-10)
            np.testing.assert_allclose(difference['mse'], outer['gates'][layout][design][arm]['candidate_minus_reference']['f_mse'], atol=1e-12, rtol=1e-10)
            comparisons[arm] = {'candidate_minus_reference': difference,
                'level_fraction_of_positive_mse_gap': difference['level_mse']/difference['mse']}
        rows.append({'layout': layout, 'design': design, 'candidate_breakdown': candidate, 'comparisons': comparisons})
    source_changes = {layout: {contract: {key: group['shape_minus_native'][key] for key in ('f_mse', 'level_mse', 'centered_shape_mse')}
        for contract, group in groups.items()} for layout, groups in inner['summary'].items()}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'outer_rows': rows, 'inner_mean_changes': source_changes,
        'cost': {'new_source_fits': 0, 'new_source_integrations': 0, 'new_raw_forecasts': 0,
            'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started, 'no_new_unscored_outcomes_read': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'no_model_adopted': True, 'observed_label_decomposition_not_oracle_recovery': True,
        'interpretation': 'Inner mean gains persist with honest source4 raw updates;outer task1 losses remain at two other frozen four-point designs. Level/shape arithmetic locates error,but task shift,source4-versus-source5 and anchor-versus-off-source query contracts are not causally separated.'}
    save(out/'results.json', result)
    print('cached outer gap attribution', [{'layout': r['layout'], 'design': r['design'], 'comparisons': r['comparisons']} for r in rows], flush=True)


if __name__ == '__main__':
    run()
