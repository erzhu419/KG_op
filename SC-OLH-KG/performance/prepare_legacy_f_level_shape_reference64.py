"""Complete source rules using the original already-verified64-node resolution."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_shape_uncertainty import source_system
from performance.prepare_legacy_f_level_shape_joint_quadrature import transform
from performance.prepare_legacy_f_level_shape_scaled_quadrature import scaled_rule
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_reference64_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); ref = read(spec['reference_execution'])
    geometry = read(ref['joint_execution']); shape = read(geometry['shape_execution'])
    coverage, profile = source_paths(read(shape['source_context_execution']))
    layout, deleted = spec['fold']['layout'], spec['fold']['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
    inputs = read(Path(coverage['output'])/suffix/'inputs.json')
    head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
    joint = TaskLevelShapeReference(source_system(inputs, head, deleted))
    old_result = next(r for r in read(spec['retained_results'])['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
    old_refinement = next(r for r in read(spec['reference_results'])['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)['trials'][-1]
    assert old_refinement['nodes_per_axis'] == 128 and old_refinement['passed']
    with np.load(ROOT/old_result['reference_moment_vector_file']) as data:
        context = {'joint': joint, 'native': joint.evaluate(joint.native_level, joint.native_shape),
            'vector_size': len(data['normalized_scaled_moments']), 'log_scale': float(data['source_log_scale']),
            'reference_normalizer': float(data['scaled_reference_normalizer']), 'reference_vector': data['normalized_scaled_moments'].copy(),
            'units': data['moment_units'].copy(), 'density_evaluations': 0}
        old_units = data['source_coordinate_units'].copy()
    anchor = np.asarray(old_units)*.5
    # Scaling may use any fixed positive source point; the physical density is unchanged.
    units = np.full(2, joint.signal); value = joint.evaluate(*anchor, moments=False)
    coordinates = np.sqrt(anchor/units)/(1+np.sqrt(anchor/units)); old_scale = context['log_scale']
    context['log_scale'] = -value['total_source_nll']+context['native']['total_source_nll']+value['joint_jeffreys_log_measure']
    context['log_scale'] += sum(transform(t, unit)[1] for t, unit in zip(coordinates, units))
    context['reference_normalizer'] *= np.exp(old_scale-context['log_scale'])
    rule, errors = scaled_rule(context, units, tuple(spec['nodes_per_axis']))
    passed = all(errors[k] <= spec['acceptance'][k] for k in errors)
    retained = read(spec['retained_rules'])['rows']; assert len(retained) == 11
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); rules = list(retained)
    if passed:
        assert (rule['posterior_weights'] > 0).all() and abs(rule['posterior_weights'].sum()-1.) < 1e-12
        path = out/f'layout{layout}_task{deleted}_source_rule.npz'; np.savez_compressed(path, **rule)
        original = read(Path(ref['output'])/'source_quadrature.json')['rows']
        old_rule = next(r for r in original if r['layout'] == layout and r['deleted_task'] == deleted)
        rules.append({**old_rule, 'source_rule_file': str(path.relative_to(ROOT)), 'node_count': len(rule['posterior_weights']),
            'source_coordinate_units': units.tolist(), 'nodes_per_axis': spec['nodes_per_axis'],
            'reference_moment_vector_file': old_result['reference_moment_vector_file']})
    complete = passed and len(rules) == 12
    save(out/'source_quadrature.json', {'execution': spec, 'rows': rules, 'all12_source_rules_frozen_without_query_data': complete})
    result = {'execution': spec, 'layout': layout, 'deleted_task': deleted, 'passed': passed, 'errors': errors,
        'original_reference_refinement': old_refinement, 'all12_compact_source_rules_verified': complete,
        'retained_rules_recomputed': 0, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'compact_source_rule_bytes': sum((ROOT/r['source_rule_file']).stat().st_size for r in rules),
        'component_counts': [r['node_count'] for r in rules], 'no_prior_or_model_adopted': True,
        'query_paid_outcomes_read': False, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({k:result[k] for k in ('passed', 'errors', 'all12_compact_source_rules_verified', 'cpu_seconds', 'component_counts', 'compact_source_rule_bytes')}, flush=True)


if __name__ == '__main__':
    run()
