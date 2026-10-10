"""Source-predictive allocation of diagonal low-frequency shape budgets."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_shape_geometry import condition
from performance.diagnose_legacy_f_inner_task_level_shape import prepare
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_shape_allocation_execution_v1.json'


def shape_system(system):
    direction = np.diag(np.diag(system['direction'])); direction[0, 0] = 0.
    d = system['d']; local = [d[j:j+12] for j in range(0, len(d), 12)]
    assert len(local) == 4
    np.testing.assert_array_equal(direction[0], 0.)
    assert np.all(np.diag(np.diag(system['residual'])+direction) >= 0.)
    return {**system, 'direction': direction, 'cq': block_diag(*[z @ direction @ z.T for z in local])}


def run():
    started = time.process_time(); spec = read(EXECUTION); joint = read(spec['joint_results']); attribution = read(spec['attribution_results'])
    assert not joint['source_predictive_joint_direction_supported']
    assert all(group['existing_task_shape_marginals']['predictive_q_gradient_per_label']
        +group['new_task_group_marginals']['predictive_q_gradient_per_label'] < 0 for group in attribution['summary'].values())
    inner = read(joint['execution']['inner_results']); coverage, _ = source_paths(read(inner['execution']['source_context_execution']))
    rows = []
    for record in inner['rows']:
        layout, query = record['layout'], record['inner_query_task']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{spec['outer_deleted_task']}"/'inputs.json')
        saved = read(record['source_fit']); assert query not in saved['inner_training_task_names']
        system = shape_system(prepare(inputs, saved['fits'], query)); baseline = condition(system, 0.)
        probe = condition(system, spec['fixed_probe_q'])
        reference = next(r for r in attribution['rows'] if r['layout'] == layout and r['inner_query_task'] == query)
        expected = sum(reference['component_attribution'][key]['predictive_q_gradient_per_label']
                       for key in ('existing_task_shape_marginals', 'new_task_group_marginals'))
        np.testing.assert_allclose(baseline['predictive_q_gradient_per_label'], expected, atol=1e-9, rtol=1e-8)
        verification = {'original_task_intercept_and_no_correlation_preserved': True,
                        'shape_allocation_derivative_matches_saved_attribution': True}
        if query == spec['gradient_verification_task']:
            point, step = spec['gradient_point'], spec['gradient_step']
            finite = (condition(system, point+step)['predictive_nll_per_label']-condition(system, point-step)['predictive_nll_per_label'])/(2*step)
            expected = condition(system, point)['predictive_q_gradient_per_label']
            np.testing.assert_allclose(finite, expected, atol=1e-7, rtol=1e-4)
            verification['predictive_gradient_finite_difference_error'] = float(abs(finite-expected))
        def scalar(value):
            return {k: v for k, v in value.items() if isinstance(v, float)}
        rows.append({'layout': layout, 'inner_query_task': query, 'baseline': scalar(baseline), 'fixed_probe': scalar(probe),
            'fixed_probe_predictive_density_gain_per_label': baseline['predictive_nll_per_label']-probe['predictive_nll_per_label'],
            'verification': verification})
    summary = {layout: {'mean_predictive_q_gradient': float(np.mean([r['baseline']['predictive_q_gradient_per_label'] for r in rows if r['layout'] == layout])),
        'mean_training_q_gradient': float(np.mean([r['baseline']['training_q_gradient_per_label'] for r in rows if r['layout'] == layout])),
        'mean_fixed_probe_predictive_density_gain': float(np.mean([r['fixed_probe_predictive_density_gain_per_label'] for r in rows if r['layout'] == layout])),
        'fixed_probe_improved_inner_tasks': sum(r['fixed_probe_predictive_density_gain_per_label'] > 0 for r in rows if r['layout'] == layout)}
        for layout in spec['layouts']}
    supported = all(v['mean_predictive_q_gradient'] < 0 and v['mean_fixed_probe_predictive_density_gain'] > 0 for v in summary.values())
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'source_predictive_shape_allocation_supported': supported,
        'cpu_seconds': time.process_time()-started, 'cost': spec['cost'], 'no_model_adopted': True,
        'outer_query_paid_or_score_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'direction_supported': supported, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
