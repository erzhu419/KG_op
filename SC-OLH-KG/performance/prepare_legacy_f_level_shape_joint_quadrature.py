"""Positive full-tail source rule for the coherent two-variance reference law."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_shape_uncertainty import source_system
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_joint_quadrature_execution_v1.json'


def transform(t, signal):
    return signal*(t/(1-t))**2, np.log(2*signal)+np.log(t)-3*np.log1p(-t)


def tensor_rule(joint, size, reference, native, log_scale):
    nodes, weights = roots_legendre(size); coordinates = (nodes+1)/2; weights = weights/2
    variances, log_jacobians = transform(coordinates, joint.signal)
    second_v = reference[0]['mean']**2+reference[0]['standard_deviation']**2
    second_a = reference[1]['mean']**2+reference[1]['standard_deviation']**2
    units = np.array([1., reference[0]['mean'], reference[1]['mean'], second_v, second_a,
        reference[0]['mean']*reference[1]['mean']])
    dimension = len(joint.beta00); accumulated = np.zeros(6+dimension+2*dimension**2)
    physical_nodes, unnormalized_weights = [], []
    for i, v in enumerate(variances):
        for j, a in enumerate(variances):
            value = joint.evaluate(float(v), float(a))
            mass = float(weights[i]*weights[j]*np.exp(-value['total_source_nll']+native['total_source_nll']
                +value['joint_jeffreys_log_measure']+log_jacobians[i]+log_jacobians[j]-log_scale))
            delta = (value['beta']-native['beta'])/np.sqrt(joint.signal)
            moment = np.r_[np.array([1., v, a, v*v, a*a, v*a])/units, delta,
                (value['h']/joint.signal).ravel(), np.outer(delta, delta).ravel()]
            accumulated += mass*moment
            physical_nodes.append((float(v), float(a))); unnormalized_weights.append(mass)
    assert np.isfinite(accumulated).all() and accumulated[0] > 0
    normalized = accumulated/accumulated[0]
    mean_delta = normalized[6:6+dimension]
    average_h = joint.signal*normalized[6+dimension:6+dimension+dimension**2].reshape(dimension, dimension)
    delta_covariance = normalized[-dimension**2:].reshape(dimension, dimension)-np.outer(mean_delta, mean_delta)
    integrated_h = average_h+joint.signal*delta_covariance; integrated_h = .5*(integrated_h+integrated_h.T)
    assert np.linalg.eigvalsh(integrated_h).min() >= -1e-10
    mean_v, mean_a, second_v, second_a, cross = normalized[1:6]*units[1:6]
    assert second_v >= mean_v**2 and second_a >= mean_a**2
    sd_v, sd_a = np.sqrt(second_v-mean_v**2), np.sqrt(second_a-mean_a**2)
    moments = {'level_variance_mean': float(mean_v), 'shape_variance_mean': float(mean_a),
        'level_variance_standard_deviation': float(sd_v), 'shape_variance_standard_deviation': float(sd_a),
        'level_shape_variance_posterior_correlation': float((cross-mean_v*mean_a)/(sd_v*sd_a)),
        'integrated_common_coefficient_mean': (native['beta']+np.sqrt(joint.signal)*mean_delta).tolist(),
        'integrated_common_coefficient_covariance': integrated_h.tolist()}
    return {'scaled_normalizer': float(accumulated[0]), 'normalized_scaled_moments': normalized,
        'moments': moments, 'physical_nodes': np.asarray(physical_nodes),
        'posterior_weights': np.asarray(unnormalized_weights)/accumulated[0]}


def run():
    started = time.process_time(); spec = read(EXECUTION); geometry = read(spec['joint_execution'])
    shape_spec = read(geometry['shape_execution']); coverage, profile = source_paths(read(shape_spec['source_context_execution']))
    shapes = read(geometry['shape_results'])['rows']; levels = read(geometry['intercept_results'])['rows']
    previous_geometry = read(Path(read(spec['joint_execution'])['output'])/'results.json')['rows']
    source_laws = read(spec['joint_laws'])['rows']; rows, rule_metadata = [], []
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    for layout in geometry['layouts']:
        for deleted in geometry['training_tasks']:
            row_started = time.process_time(); suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            system = source_system(inputs, head, deleted); joint = TaskLevelShapeReference(system)
            native = joint.evaluate(joint.native_level, joint.native_shape)
            np.testing.assert_allclose(native['total_source_nll']/60, system['native']['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(native['beta'], system['native']['coefficient_mean'], atol=1e-8, rtol=1e-8)
            np.testing.assert_allclose(native['h'], system['native']['population_mean_covariance'], atol=1e-8, rtol=1e-8)
            previous = next(r for r in previous_geometry if r['layout'] == layout and r['deleted_task'] == deleted)
            for point in previous['source_diagnostic_points']:
                evaluated = joint.evaluate(point['level_variance'], point['shape_variance'], moments=False)
                np.testing.assert_allclose(evaluated['total_source_nll'], point['total_source_nll'], atol=1e-8, rtol=1e-8)
                np.testing.assert_allclose(evaluated['joint_to_product_conditional_jeffreys_factor'],
                    point['joint_to_product_conditional_jeffreys_factor'], atol=1e-8, rtol=1e-8)
            verification = {'native_and_previous_joint_density_points_reproduced': True}
            if deleted == geometry['dense_verification_deleted_task']:
                interior = geometry['dense_verification_variance_over_signal']*joint.signal
                verification.update(joint.dense_verification(interior, interior))
            level = next(r for r in levels if r['layout'] == layout and r['deleted_task'] == deleted)['reference_posterior']
            shape = next(r for r in shapes if r['layout'] == layout and r['deleted_task'] == deleted)['reference_posterior']
            median_v, median_a = (r['quantiles']['0.5'] for r in (level, shape))
            t, s = (np.sqrt(v/joint.signal)/(1+np.sqrt(v/joint.signal)) for v in (median_v, median_a))
            point = joint.evaluate(median_v, median_a, moments=False)
            log_scale = -point['total_source_nll']+native['total_source_nll']+point['joint_jeffreys_log_measure']
            log_scale += transform(t, joint.signal)[1]+transform(s, joint.signal)[1]
            trials, old_rule, selected = [], None, None
            for size in spec['resolutions']:
                rule = tensor_rule(joint, size, (level, shape), native, log_scale)
                errors = None if old_rule is None else {
                    'normalizer_relative_difference': float(abs(rule['scaled_normalizer']/old_rule['scaled_normalizer']-1.)),
                    'scaled_moment_max_absolute_difference': float(np.max(np.abs(rule['normalized_scaled_moments']-old_rule['normalized_scaled_moments'])))}
                passed = errors is not None and all(errors[k] <= limit for k, limit in spec['acceptance'].items())
                trials.append({'nodes_per_axis': size, 'node_count': size*size, 'errors': errors, 'passed': passed})
                if passed:
                    selected = rule; break
                old_rule = rule
            row = {'layout': layout, 'deleted_task': deleted, 'trials': trials, 'passed': selected is not None,
                'last_resolution_moments': rule['moments'], 'verification': verification, 'cpu_seconds': time.process_time()-row_started}
            if selected is not None:
                path = out/f'layout{layout}_task{deleted}_source_rule.npz'
                np.savez_compressed(path, physical_variance_nodes=selected['physical_nodes'], posterior_weights=selected['posterior_weights'])
                source_law = next(r for r in source_laws if r['layout'] == layout and r['deleted_task'] == deleted)
                rule_metadata.append({'layout': layout, 'deleted_task': deleted, 'node_count': len(selected['posterior_weights']),
                    'source_rule_file': str(path.relative_to(ROOT)), 'source_reference_law': source_law,
                    'reference_log_normalizer_relative_to_native_nll': float(log_scale+np.log(selected['scaled_normalizer'])),
                    'integrated_source_moments': selected['moments'], 'averaged_gaussian_not_adopted': True})
                row['selected_node_count'] = len(selected['posterior_weights'])
            rows.append(row)
            print('joint source quadrature', layout, deleted, 'passed', row['passed'], 'nodes', size*size, 'CPU', row['cpu_seconds'], flush=True)
    complete = len(rule_metadata) == 12
    save(out/'source_quadrature.json', {'execution': spec, 'rows': rule_metadata, 'all12_source_rules_frozen_without_query_data': complete})
    result = {'execution': spec, 'rows': rows, 'all12_source_rules_verified': complete,
        'refinement_is_numerical_evidence_not_analytic_error_bound': True, 'cpu_seconds': time.process_time()-started,
        'cost': {**spec['cost'], 'actual_joint_density_quadrature_evaluations': sum(t['node_count'] for r in rows for t in r['trials'])},
        'compact_source_rule_bytes': sum((ROOT/r['source_rule_file']).stat().st_size for r in rule_metadata),
        'no_prior_or_model_adopted': True, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'all12_source_rules_verified': complete, 'cpu_seconds': result['cpu_seconds'],
        'compact_source_rule_bytes': result['compact_source_rule_bytes']}, flush=True)


if __name__ == '__main__':
    run()
