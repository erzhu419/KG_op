"""Algebraic predictive-score attribution of the stopped joint task direction."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_inner_task_level_shape import prepare
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_inner_joint_attribution_execution_v1.json'


def parts(system):
    direction = system['direction']; shape = np.diag(np.diag(direction)); shape[0, 0] = 0.
    level = np.zeros_like(direction); level[0, 0] = direction[0, 0]
    coupling = np.zeros_like(direction); coupling[0, 1:] = direction[0, 1:]; coupling[1:, 0] = direction[1:, 0]
    active = np.r_[False, system['residual'][1:] > system['geometry']['independent_tail_per_shape_coefficient']]
    existing = np.diag(np.diag(shape)*active)
    result = {'level_variance': level, 'existing_task_shape_marginals': existing,
        'new_task_group_marginals': shape-existing, 'level_shape_correlation': coupling,
        'shape_correlation': direction-shape-level-coupling}
    np.testing.assert_allclose(sum(result.values()), direction, atol=1e-14, rtol=1e-12)
    return result


def attribute(system, directions):
    d, a, error = system['d'], system['a'], system['error']; n = len(error)
    factor = cho_factor(system['c'], lower=True); alpha = cho_solve(factor, error)
    solved = cho_solve(factor, d @ a); inverse = cho_solve(factor, np.eye(n))
    beta = system['m0']+a @ d.T @ alpha; h = a-a @ d.T @ solved
    z = system['query_design']; mean = z @ beta
    k = z @ (h+np.diag(system['residual'])) @ z.T+np.diag(system['query_noise'])
    query_factor = cho_factor(k, lower=True); residual = system['query_labels']-mean
    query_alpha = cho_solve(query_factor, residual); count = len(mean)
    local = [d[j:j+12] for j in range(0, n, 12)]
    assert n == 48 and len(local) == 4
    weights = np.ones(count)/count; result = {}
    for name, direction in directions.items():
        source_derivative = block_diag(*[design @ direction @ design.T for design in local])
        mean_derivative = -z @ solved.T @ source_derivative @ alpha
        common_derivative = z @ solved.T @ source_derivative @ solved @ z.T
        task_derivative = z @ direction @ z.T; derivative = common_derivative+task_derivative
        volume = .5*np.trace(cho_solve(query_factor, derivative))/count
        residual_covariance = -.5*query_alpha @ derivative @ query_alpha/count
        mean_term = -mean_derivative @ query_alpha/count
        result[name] = {'predictive_q_gradient_per_label': float(volume+residual_covariance+mean_term),
            'predictive_covariance_volume_term': float(volume),
            'predictive_residual_covariance_term': float(residual_covariance), 'predictive_mean_term': float(mean_term),
            'training_q_gradient_per_label': float(.5*(np.sum(inverse*source_derivative)-alpha @ source_derivative @ alpha)/n),
            'held_task_average_mean_derivative': float(weights @ mean_derivative),
            'held_task_average_predictive_variance_derivative': float(weights @ derivative @ weights),
            'common_conditioning_level_variance_derivative': float(weights @ common_derivative @ weights),
            'direct_task_level_variance_derivative': float(weights @ task_derivative @ weights)}
    return result


def run():
    started = time.process_time(); spec = read(EXECUTION); joint = read(spec['joint_results'])
    assert not joint['source_predictive_joint_direction_supported']
    inner = read(joint['execution']['inner_results']); coverage, _ = source_paths(read(inner['execution']['source_context_execution']))
    rows = []
    for record in inner['rows']:
        layout, query = record['layout'], record['inner_query_task']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/f"task{joint['execution']['outer_deleted_task']}"/'inputs.json')
        fits = read(record['source_fit']); system = prepare(inputs, fits['fits'], query)
        decomposition = attribute(system, parts(system))
        saved = next(r for r in joint['rows'] if r['layout'] == layout and r['inner_query_task'] == query)
        for key in ('predictive_q_gradient_per_label', 'training_q_gradient_per_label'):
            np.testing.assert_allclose(sum(v[key] for v in decomposition.values()), saved['baseline'][key], atol=1e-9, rtol=1e-8)
        rows.append({'layout': layout, 'inner_query_task': query, 'component_attribution': decomposition,
            'additive_parts_and_saved_predictive_training_gradients_reproduced': True})
    summary = {layout: {name: {key: float(np.mean([r['component_attribution'][name][key] for r in rows if r['layout'] == layout]))
                              for key in rows[0]['component_attribution'][name]}
                        for name in rows[0]['component_attribution']} for layout in joint['execution']['layouts']}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'summary': summary, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'attribution_only_no_component_probe_or_new_candidate': True, 'outer_query_paid_or_score_outcomes_read': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'summary': {layout: {name: {key: values[key] for key in ['predictive_q_gradient_per_label',
        'predictive_covariance_volume_term', 'predictive_residual_covariance_term', 'predictive_mean_term']}
        for name, values in group.items()} for layout, group in summary.items()}, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
