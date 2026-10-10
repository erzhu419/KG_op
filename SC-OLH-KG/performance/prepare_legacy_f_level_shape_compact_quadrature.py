"""Source-only positive adaptive cubature for the frozen joint variance law."""

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
from performance.prepare_legacy_f_level_shape_joint_quadrature import transform
from performance.run_legacy_related_inventory_source import save
from representation.task_level_shape_reference import TaskLevelShapeReference

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_shape_compact_quadrature_execution_v1.json'


def moment_vector(value, v, a, context):
    joint, native = context['joint'], context['native']
    delta = (value['beta']-native['beta'])/np.sqrt(joint.signal)
    return np.r_[np.array([1., v, a, v*v, a*a, v*a])/context['units'], delta,
        (value['h']/joint.signal).ravel(), np.outer(delta, delta).ravel()]


def rectangle_rule(bounds, size, context):
    x, w = roots_legendre(size); joint = context['joint']; native = context['native']
    coordinates = [.5*(lo+hi)+.5*(hi-lo)*x for lo, hi in bounds]
    weights = [.5*(hi-lo)*w for lo, hi in bounds]
    physical = [transform(t, joint.signal) for t in coordinates]
    integral = np.zeros(context['vector_size']); nodes, masses = [], []
    for i, v in enumerate(physical[0][0]):
        for j, a in enumerate(physical[1][0]):
            value = joint.evaluate(float(v), float(a))
            mass = float(weights[0][i]*weights[1][j]*np.exp(-value['total_source_nll']+native['total_source_nll']
                +value['joint_jeffreys_log_measure']+physical[0][1][i]+physical[1][1][j]-context['log_scale']))
            integral += mass*moment_vector(value, v, a, context)
            nodes.append((float(v), float(a))); masses.append(mass)
    context['density_evaluations'] += size*size
    return {'integral': integral, 'nodes': np.asarray(nodes), 'masses': np.asarray(masses)}


def rectangle(bounds, context, spec):
    low = rectangle_rule(bounds, spec['low_nodes_per_axis'], context)
    high = rectangle_rule(bounds, spec['high_nodes_per_axis'], context)
    return {'bounds': bounds, 'high': high, 'absolute_error_vector': np.abs(high['integral']-low['integral'])}


def source_context(layout, deleted, coverage, profile, shape, level, reference_rule):
    suffix = Path(f'layout{layout}')/f'task{deleted}'
    inputs = read(Path(coverage['output'])/suffix/'inputs.json')
    head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
    system = source_system(inputs, head, deleted); joint = TaskLevelShapeReference(system)
    native = joint.evaluate(joint.native_level, joint.native_shape)
    median = [r['quantiles']['0.5'] for r in (level, shape)]
    splits = [float(np.sqrt(v/joint.signal)/(1+np.sqrt(v/joint.signal))) for v in median]
    value = joint.evaluate(*median, moments=False)
    log_scale = -value['total_source_nll']+native['total_source_nll']+value['joint_jeffreys_log_measure']
    log_scale += sum(transform(t, joint.signal)[1] for t in splits)
    second = [r['mean']**2+r['standard_deviation']**2 for r in (level, shape)]
    context = {'joint': joint, 'native': native, 'splits': splits, 'log_scale': log_scale, 'density_evaluations': 0,
        'units': np.array([1., level['mean'], shape['mean'], *second, level['mean']*shape['mean']]),
        'vector_size': 6+len(joint.beta00)+2*len(joint.beta00)**2}
    reference_vector = np.zeros(context['vector_size'])
    with np.load(ROOT/reference_rule['source_rule_file']) as saved:
        for (v, a), weight in zip(saved['physical_variance_nodes'], saved['posterior_weights']):
            value = joint.evaluate(float(v), float(a))
            reference_vector += weight*moment_vector(value, v, a, context)
        context['reference_moment_reconstruction_evaluations'] = len(saved['posterior_weights'])
    context['reference_normalizer'] = float(np.exp(reference_rule['reference_log_normalizer_relative_to_native_nll']-log_scale))
    context['reference_vector'] = reference_vector
    return context


def compact_rule(context, spec):
    axes = [[(0., split), (split, 1.)] for split in context['splits']]
    rectangles = [rectangle([v, a], context, spec) for v in axes[0] for a in axes[1]]
    history = []; limits = spec['acceptance']; passed = False
    while True:
        integral = sum(r['high']['integral'] for r in rectangles)
        errors = {'normalizer_relative_difference_vs_stage137': float(abs(integral[0]/context['reference_normalizer']-1.)),
            'scaled_moment_max_absolute_difference_vs_stage137': float(np.max(np.abs(integral/integral[0]-context['reference_vector']))),
            'sum_of_rectangle_high_low_scaled_errors': float(np.max(sum(r['absolute_error_vector'] for r in rectangles))/integral[0])}
        node_count = len(rectangles)*spec['high_nodes_per_axis']**2
        passed = all(errors[k] <= limits[k] for k in errors)
        history.append({'rectangle_count': len(rectangles), 'positive_node_count': node_count, 'errors': errors, 'passed': passed})
        if passed or node_count+spec['high_nodes_per_axis']**2 > limits['maximum_positive_nodes_per_fold']:
            break
        worst = int(np.argmax([np.max(r['absolute_error_vector']) for r in rectangles]))
        old = rectangles.pop(worst); bounds = old['bounds']; axis = int(np.argmax([hi-lo for lo, hi in bounds]))
        lo, hi = bounds[axis]; midpoint = .5*(lo+hi)
        for replacement in [(lo, midpoint), (midpoint, hi)]:
            child = list(bounds); child[axis] = replacement
            rectangles.append(rectangle(child, context, spec))
    rule = {'physical_variance_nodes': np.vstack([r['high']['nodes'] for r in rectangles]),
        'posterior_weights': np.concatenate([r['high']['masses'] for r in rectangles])/integral[0]}
    assert (rule['posterior_weights'] > 0).all() and abs(rule['posterior_weights'].sum()-1.) < 1e-12
    return rule, {'passed': passed, 'history': history, 'positive_node_count': node_count,
        'rectangles': [r['bounds'] for r in rectangles], 'reference_moment_reconstruction_evaluations': context['reference_moment_reconstruction_evaluations'],
        'adaptive_density_evaluations': context['density_evaluations'], 'both_full_variance_tails_retained': True}


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
        rule, record = compact_rule(context, spec)
        row = {'layout': layout, 'deleted_task': deleted, **record, 'cpu_seconds': time.process_time()-row_started}
        rows.append(row)
        if record['passed']:
            path = out/f'layout{layout}_task{deleted}_source_rule.npz'; np.savez_compressed(path, **rule)
            rules.append({**old_rule, 'source_rule_file': str(path.relative_to(ROOT)), 'node_count': record['positive_node_count'],
                'reference_source_rule_file': old_rule['source_rule_file'], 'full_domain_rectangles': record['rectangles']})
        print('compact joint source rule', layout, deleted, 'passed', record['passed'], 'nodes', record['positive_node_count'],
            'errors', record['history'][-1]['errors'], 'CPU', row['cpu_seconds'], flush=True)
        if len(rows) == 2 and not all(r['passed'] for r in rows):
            break
    complete = len(rules) == 12
    save(out/'source_quadrature.json', {'execution': spec, 'rows': rules, 'all12_source_rules_frozen_without_query_data': complete})
    result = {'execution': spec, 'rows': rows, 'all12_compact_source_rules_verified': complete,
        'pilot_both_task1_passed': len(rows) >= 2 and all(r['passed'] for r in rows[:2]),
        'cpu_seconds': time.process_time()-started, 'cost': {**spec['cost'],
            'actual_reference_reconstruction_evaluations': sum(r['reference_moment_reconstruction_evaluations'] for r in rows),
            'actual_adaptive_density_evaluations': sum(r['adaptive_density_evaluations'] for r in rows)},
        'compact_source_rule_bytes': sum((ROOT/r['source_rule_file']).stat().st_size for r in rules),
        'no_prior_or_model_adopted': True, 'query_paid_outcomes_read': False, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'all12_compact_source_rules_verified': complete, 'pilot_both_task1_passed': result['pilot_both_task1_passed'],
        'cpu_seconds': result['cpu_seconds'], 'compact_source_rule_bytes': result['compact_source_rule_bytes']}, flush=True)


if __name__ == '__main__':
    run()
