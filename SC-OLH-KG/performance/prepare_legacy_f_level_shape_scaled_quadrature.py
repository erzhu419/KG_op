"""Positive joint cubature using source posterior medians as coordinate units."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_level_shape_compact_quadrature import moment_vector, source_context
from performance.prepare_legacy_f_level_shape_joint_quadrature import transform
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_scaled_quadrature_execution_v1.json'


def scaled_rule(context, coordinate_units, axis_sizes):
    rules = [roots_legendre(size) for size in axis_sizes]
    weights = [rule[1]/2 for rule in rules]
    physical = [transform((rule[0]+1)/2, unit) for rule, unit in zip(rules, coordinate_units)]
    integral = np.zeros(context['vector_size']); variances, masses = [], []
    for i, v in enumerate(physical[0][0]):
        for j, a in enumerate(physical[1][0]):
            value = context['joint'].evaluate(float(v), float(a))
            mass = float(weights[0][i]*weights[1][j]*np.exp(-value['total_source_nll']+context['native']['total_source_nll']
                +value['joint_jeffreys_log_measure']+physical[0][1][i]+physical[1][1][j]-context['log_scale']))
            integral += mass*moment_vector(value, v, a, context)
            variances.append((float(v), float(a))); masses.append(mass)
    context['density_evaluations'] += axis_sizes[0]*axis_sizes[1]
    assert np.isfinite(integral).all() and integral[0] > 0
    errors = {'normalizer_relative_difference_vs_stage137': float(abs(integral[0]/context['reference_normalizer']-1.)),
        'scaled_moment_max_absolute_difference_vs_stage137': float(np.max(np.abs(integral/integral[0]-context['reference_vector'])))}
    return {'physical_variance_nodes': np.asarray(variances), 'posterior_weights': np.asarray(masses)/integral[0],
        'normalized_scaled_moments': integral/integral[0]}, errors


def run():
    started = time.process_time(); spec = read(EXECUTION); reference_spec = read(spec['reference_execution'])
    geometry = read(reference_spec['joint_execution']); shape_spec = read(geometry['shape_execution'])
    coverage, profile = source_paths(read(shape_spec['source_context_execution']))
    shapes = read(geometry['shape_results'])['rows']; levels = read(geometry['intercept_results'])['rows']
    reference = read(spec['reference_rules']); assert reference['all12_source_rules_frozen_without_query_data']
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    order = [(layout, 1) for layout in geometry['layouts']]+[(layout, task) for layout in geometry['layouts'] for task in geometry['training_tasks'] if task != 1]
    rows, rules = [], []
    for layout, deleted in order:
        row_started = time.process_time()
        shape = next(r for r in shapes if r['layout'] == layout and r['deleted_task'] == deleted)['reference_posterior']
        level = next(r for r in levels if r['layout'] == layout and r['deleted_task'] == deleted)['reference_posterior']
        old_rule = next(r for r in reference['rows'] if r['layout'] == layout and r['deleted_task'] == deleted)
        context = source_context(layout, deleted, coverage, profile, shape, level, old_rule)
        coordinate_units = [r['quantiles']['0.5'] for r in (level, shape)]
        median = context['joint'].evaluate(*coordinate_units, moments=False)
        old_scale = context['log_scale']
        context['log_scale'] = -median['total_source_nll']+context['native']['total_source_nll']+median['joint_jeffreys_log_measure']
        context['log_scale'] += sum(transform(.5, unit)[1] for unit in coordinate_units)
        context['reference_normalizer'] *= np.exp(old_scale-context['log_scale'])
        reference_path = out/f'layout{layout}_task{deleted}_reference_moments.npz'
        np.savez_compressed(reference_path, normalized_scaled_moments=context['reference_vector'],
            source_coordinate_units=np.asarray(coordinate_units), source_log_scale=np.asarray(context['log_scale']),
            scaled_reference_normalizer=np.asarray(context['reference_normalizer']), moment_units=context['units'])
        trials, selected = [], None
        for size in spec['resolutions']:
            rule, errors = scaled_rule(context, coordinate_units, (size, size))
            passed = all(errors[k] <= spec['acceptance'][k] for k in errors)
            trials.append({'nodes_per_axis': size, 'node_count': size*size, 'errors': errors, 'passed': passed})
            if passed:
                selected = rule; break
        row = {'layout': layout, 'deleted_task': deleted, 'passed': selected is not None, 'trials': trials,
            'source_coordinate_units': coordinate_units, 'source_signal': context['joint'].signal,
            'reference_moment_vector_file': str(reference_path.relative_to(ROOT)),
            'reference_moment_reconstruction_evaluations': context['reference_moment_reconstruction_evaluations'],
            'density_evaluations': context['density_evaluations'], 'both_full_variance_tails_retained': True,
            'cpu_seconds': time.process_time()-row_started}
        rows.append(row)
        if selected is not None:
            assert (selected['posterior_weights'] > 0).all() and abs(selected['posterior_weights'].sum()-1.) < 1e-12
            path = out/f'layout{layout}_task{deleted}_source_rule.npz'; np.savez_compressed(path, **selected)
            rules.append({**old_rule, 'source_rule_file': str(path.relative_to(ROOT)), 'node_count': len(selected['posterior_weights']),
                'reference_source_rule_file': old_rule['source_rule_file'], 'source_coordinate_units': coordinate_units,
                'nodes_per_axis': size, 'reference_moment_vector_file': str(reference_path.relative_to(ROOT))})
        print('median-scaled joint rule', layout, deleted, 'passed', row['passed'], 'nodes', size*size,
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
