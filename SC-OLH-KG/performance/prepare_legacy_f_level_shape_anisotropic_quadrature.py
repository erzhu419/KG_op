"""Resolve the source intercept second moment without oversampling shape."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_shape_uncertainty import source_system
from performance.prepare_legacy_f_level_shape_compact_quadrature import source_context
from performance.prepare_legacy_f_level_shape_joint_quadrature import transform
from performance.prepare_legacy_f_level_shape_scaled_quadrature import scaled_rule
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_anisotropic_quadrature_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); reference = read(spec['reference_execution'])
    geometry = read(reference['joint_execution']); shape_spec = read(geometry['shape_execution'])
    coverage, profile = source_paths(read(shape_spec['source_context_execution']))
    shapes = read(geometry['shape_results'])['rows']; levels = read(geometry['intercept_results'])['rows']
    saved_vectors = read(spec['scaled_results'])['rows']; old_rules = read(spec['reference_rules'])
    assert old_rules['all12_source_rules_frozen_without_query_data']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    order = [(layout, 1) for layout in geometry['layouts']]+[(layout, task) for layout in geometry['layouts'] for task in geometry['training_tasks'] if task != 1]
    rows, rules = [], []
    for layout, deleted in order:
        row_started = time.process_time()
        shape = next(r for r in shapes if r['layout'] == layout and r['deleted_task'] == deleted)['reference_posterior']
        level = next(r for r in levels if r['layout'] == layout and r['deleted_task'] == deleted)['reference_posterior']
        old_rule = next(r for r in old_rules['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        if deleted == 1:
            suffix = Path(f'layout{layout}')/f'task{deleted}'
            inputs = read(Path(coverage['output'])/suffix/'inputs.json')
            head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
            joint = TaskLevelShapeReference(source_system(inputs, head, deleted))
            saved = next(r for r in saved_vectors if r['layout'] == layout and r['deleted_task'] == deleted)
            with np.load(ROOT/saved['reference_moment_vector_file']) as data:
                context = {'joint': joint, 'native': joint.evaluate(joint.native_level, joint.native_shape),
                    'vector_size': len(data['normalized_scaled_moments']), 'log_scale': float(data['source_log_scale']),
                    'reference_normalizer': float(data['scaled_reference_normalizer']), 'reference_vector': data['normalized_scaled_moments'].copy(),
                    'units': data['moment_units'].copy(), 'density_evaluations': 0, 'reference_moment_reconstruction_evaluations': 0}
        else:
            context = source_context(layout, deleted, coverage, profile, shape, level, old_rule)
        units = [context['joint'].signal, shape['quantiles']['0.5']]
        medians = [level['quantiles']['0.5'], shape['quantiles']['0.5']]
        coordinate_medians = [np.sqrt(v/unit)/(1+np.sqrt(v/unit)) for v, unit in zip(medians, units)]
        value = context['joint'].evaluate(*medians, moments=False); old_scale = context['log_scale']
        context['log_scale'] = -value['total_source_nll']+context['native']['total_source_nll']+value['joint_jeffreys_log_measure']
        context['log_scale'] += sum(transform(t, unit)[1] for t, unit in zip(coordinate_medians, units))
        context['reference_normalizer'] *= np.exp(old_scale-context['log_scale'])
        reference_path = out/f'layout{layout}_task{deleted}_reference_moments.npz'
        np.savez_compressed(reference_path, normalized_scaled_moments=context['reference_vector'], source_coordinate_units=np.asarray(units),
            source_log_scale=np.asarray(context['log_scale']), scaled_reference_normalizer=np.asarray(context['reference_normalizer']), moment_units=context['units'])
        trials, selected = [], None
        for shape_size in spec['shape_nodes_per_axis_resolutions']:
            sizes = (spec['level_nodes_per_axis'], shape_size)
            rule, errors = scaled_rule(context, units, sizes)
            passed = all(errors[k] <= spec['acceptance'][k] for k in errors)
            trials.append({'level_nodes': sizes[0], 'shape_nodes': sizes[1], 'node_count': sizes[0]*sizes[1], 'errors': errors, 'passed': passed})
            if passed:
                selected = rule; break
        row = {'layout': layout, 'deleted_task': deleted, 'passed': selected is not None, 'trials': trials,
            'source_coordinate_units': units, 'reference_moment_vector_file': str(reference_path.relative_to(ROOT)),
            'reference_moment_reconstruction_evaluations': context['reference_moment_reconstruction_evaluations'],
            'density_evaluations': context['density_evaluations'], 'both_full_variance_tails_retained': True,
            'cpu_seconds': time.process_time()-row_started}
        rows.append(row)
        if selected is not None:
            assert (selected['posterior_weights'] > 0).all() and abs(selected['posterior_weights'].sum()-1.) < 1e-12
            path = out/f'layout{layout}_task{deleted}_source_rule.npz'; np.savez_compressed(path, **selected)
            rules.append({**old_rule, 'source_rule_file': str(path.relative_to(ROOT)), 'node_count': len(selected['posterior_weights']),
                'reference_source_rule_file': old_rule['source_rule_file'], 'source_coordinate_units': units,
                'nodes_per_axis': list(sizes), 'reference_moment_vector_file': str(reference_path.relative_to(ROOT))})
        print('anisotropic joint source rule', layout, deleted, 'passed', row['passed'], 'nodes', sizes[0]*sizes[1],
            'errors', trials[-1]['errors'], 'CPU', row['cpu_seconds'], flush=True)
        if len(rows) == 2 and not all(r['passed'] for r in rows):
            break
    complete = len(rules) == 12
    save(out/'source_quadrature.json', {'execution': spec, 'rows': rules, 'all12_source_rules_frozen_without_query_data': complete})
    result = {'execution': spec, 'rows': rows, 'all12_compact_source_rules_verified': complete,
        'pilot_both_task1_passed': len(rows) >= 2 and all(r['passed'] for r in rows[:2]), 'cpu_seconds': time.process_time()-started,
        'cost': {**spec['cost'], 'actual_reference_reconstruction_evaluations': sum(r['reference_moment_reconstruction_evaluations'] for r in rows),
            'actual_density_evaluations': sum(r['density_evaluations'] for r in rows)},
        'compact_source_rule_bytes': sum((ROOT/r['source_rule_file']).stat().st_size for r in rules),
        'no_prior_or_model_adopted': True, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'all12_compact_source_rules_verified': complete, 'pilot_both_task1_passed': result['pilot_both_task1_passed'],
        'cpu_seconds': result['cpu_seconds'], 'compact_source_rule_bytes': result['compact_source_rule_bytes']}, flush=True)


if __name__ == '__main__':
    run()
