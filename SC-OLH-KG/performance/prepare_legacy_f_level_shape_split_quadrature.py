"""Repair the sole unresolved source rule; retain eleven already-passing rules."""

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
from performance.prepare_legacy_f_level_shape_compact_quadrature import moment_vector
from performance.prepare_legacy_f_level_shape_joint_quadrature import transform
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_split_quadrature_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); ref = read(spec['reference_execution'])
    geometry = read(ref['joint_execution']); shape = read(geometry['shape_execution'])
    coverage, profile = source_paths(read(shape['source_context_execution']))
    layout, deleted = spec['fold']['layout'], spec['fold']['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
    inputs = read(Path(coverage['output'])/suffix/'inputs.json')
    head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
    joint = TaskLevelShapeReference(source_system(inputs, head, deleted))
    old_result = next(r for r in read(spec['retained_results'])['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
    with np.load(ROOT/old_result['reference_moment_vector_file']) as data:
        context = {'joint': joint, 'native': joint.evaluate(joint.native_level, joint.native_shape),
            'vector_size': len(data['normalized_scaled_moments']), 'log_scale': float(data['source_log_scale']),
            'reference_normalizer': float(data['scaled_reference_normalizer']), 'reference_vector': data['normalized_scaled_moments'].copy(),
            'units': data['moment_units'].copy()}
        coordinate_units = data['source_coordinate_units'].copy()
    xv, wv = roots_legendre(128); v, jv = transform((xv+1)/2, coordinate_units[0]); wv = wv/2
    xa, wa = roots_legendre(16)
    shape_coordinates = np.r_[.25+.25*xa, .75+.25*xa]; shape_weights = np.r_[.25*wa, .25*wa]
    a, ja = transform(shape_coordinates, coordinate_units[1]); integral = np.zeros(context['vector_size'])
    nodes, masses = [], []
    for i, level in enumerate(v):
        for j, shape in enumerate(a):
            value = joint.evaluate(float(level), float(shape))
            mass = float(wv[i]*shape_weights[j]*np.exp(-value['total_source_nll']+context['native']['total_source_nll']
                +value['joint_jeffreys_log_measure']+jv[i]+ja[j]-context['log_scale']))
            integral += mass*moment_vector(value, level, shape, context)
            nodes.append((float(level), float(shape))); masses.append(mass)
    difference = integral/integral[0]-context['reference_vector']
    errors = {'normalizer_relative_difference_vs_stage137': float(abs(integral[0]/context['reference_normalizer']-1.)),
        'scaled_moment_max_absolute_difference_vs_stage137': float(np.max(np.abs(difference)))}
    passed = all(errors[k] <= spec['acceptance'][k] for k in errors)
    old_rules = read(spec['retained_rules'])['rows']; assert len(old_rules) == 11
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); new_rules = list(old_rules)
    if passed:
        path = out/f'layout{layout}_task{deleted}_source_rule.npz'
        weights = np.asarray(masses)/integral[0]; assert (weights > 0).all() and abs(weights.sum()-1.) < 1e-12
        np.savez_compressed(path, physical_variance_nodes=np.asarray(nodes), posterior_weights=weights, normalized_scaled_moments=integral/integral[0])
        original = next(r for r in read(read(spec['reference_execution'])['joint_laws'])['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        full_rules = read('paper_artifacts/legacy_joint_population_f_level_shape_joint_quadrature_v1/source_quadrature.json')['rows']
        old_rule = next(r for r in full_rules if r['layout'] == layout and r['deleted_task'] == deleted)
        new_rules.append({**old_rule, 'source_reference_law': original, 'source_rule_file': str(path.relative_to(ROOT)), 'node_count': len(weights),
            'source_coordinate_units': coordinate_units.tolist(), 'nodes_per_axis': [128, [16, 16]], 'shape_coordinate_intervals': [[0., .5], [.5, 1.]],
            'reference_moment_vector_file': old_result['reference_moment_vector_file']})
    save(out/'source_quadrature.json', {'execution': spec, 'rows': new_rules, 'all12_source_rules_frozen_without_query_data': passed and len(new_rules) == 12})
    result = {'execution': spec, 'layout': layout, 'deleted_task': deleted, 'passed': passed, 'errors': errors,
        'all12_compact_source_rules_verified': passed and len(new_rules) == 12, 'retained_rules_recomputed': 0,
        'largest_error_component': int(np.argmax(np.abs(difference))), 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'compact_source_rule_bytes': sum((ROOT/r['source_rule_file']).stat().st_size for r in new_rules),
        'no_prior_or_model_adopted': True, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print(result, flush=True)


if __name__ == '__main__':
    run()
