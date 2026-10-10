"""Learn one source-only covariance scale under the fixed few-shot contract."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score, raw_density
from performance.check_legacy_f_source_shape_scale_direction import scaled_source_law
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_f_task_level_reference_integration import diagnose
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.prepare_legacy_f_inner_raw_contract import finite_reference, finite_rule_errors
from performance.prepare_legacy_f_task_level_reference_quadrature import positive_base_rule, rule_on_intervals, approximation_errors, passes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_predictive_shape_scale_execution_v1.json'


def forecast(case, law, intervals, size):
    nodes, weights = rule_on_intervals(law, intervals, size)
    source_errors = finite_rule_errors(nodes, weights, law)
    assert all(source_errors[k] <= v for k, v in case['source_rule_tolerances'].items())
    field = mixture_field(case['basis'], case['psi'], law, nodes, weights, case['eta'])
    evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(case['paid']))
    state = compact(field, list(range(4, 12)), evidence); p = prediction(state)
    row = {'episode_count': 4, 'prediction': p}
    scores, bound = score(row, state, case['labels'])
    return scores, p, state, bound


def run():
    started = time.process_time(); spec = read(EXECUTION); direction = read(spec['direction_results'])
    assert direction['direction_interpreted'] and direction['source_coherent_shrink_direction_supported']
    context = read(spec['source_context_plan']); coverage, _ = source_paths(read(context['source_context_execution']))
    old = read(spec['reference_forecasts']); packages, rules = [], []
    for directory in old['execution']['source_packages']:
        packages += read(Path(directory)/'source_branches.json')['rows']
        rules += read(Path(directory)/'source_quadrature.json')['rows']
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data']); paid = {b['query_task']: [raw['tasks'][b['query_task']]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for b in packages}; del raw
    cases = []
    for branch in packages:
        layout, query, names = branch['layout'], branch['query_task'], branch['source_task_names']
        inputs = read(Path(coverage['output'])/f'layout{layout}'/'task1'/'inputs.json')
        kept = {**inputs, 'source_tasks': [t for t in inputs['source_tasks'] if t['name'] in names]}
        task = next(t for t in inputs['source_tasks'] if t['name'] == f'related_inventory_{query:02d}')
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][query])), bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
        selection = spec['paid_prefix']+branch['scoring_sites']
        rule = next(r for r in rules if r['layout'] == layout and r['arm'] == 'shape' and r['source_reference_law']['source_task_names'] == names)
        cases.append({'layout': layout, 'query_task': query, 'inputs': kept, 'frozen_law': rule['source_reference_law']['native_law'],
            'basis': OrthogonalLowOrderBasis(branch['public_mean_reference']), 'psi': psi[selection],
            'eta': (np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]),
            'paid': paid[query], 'labels': labels, 'source_rule_tolerances': spec['source_rule_tolerances']})
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    records, final_rules, final_forecasts = [], [], []; counters = {'source4_integrations': 0, 'cv_forecasts': 0, 'refinement_forecasts': 0}
    for layout in spec['layouts']:
        members = [c for c in cases if c['layout'] == layout]; history = []
        baseline = [r for r in old['rows'] if r['layout'] == layout and r['design'] == 'fixed_public_prefix' and r['arm'] == 'shape_mixture']
        baseline_density = float(np.mean([r['scores']['f_raw_batch_density'] for r in baseline]))
        def objective(scale):
            if time.process_time()-started > spec['maximum_single_thread_cpu_seconds']: raise RuntimeError('source scalar learning reached its registered CPU budget')
            scores = []
            for case in members:
                native = scaled_source_law(case['inputs'], case['frozen_law'], float(scale))
                record, law = finite_reference(case['inputs'], native, context); counters['source4_integrations'] += 1
                values, _, _, _ = forecast(case, law, record['adaptive_u_intervals'], spec['learning_nodes_per_interval']); counters['cv_forecasts'] += 1
                scores.append(values)
            value = float(np.mean([s['f_raw_batch_density'] for s in scores]))
            history.append({'scale': float(scale), 'raw_density': value, 'mse': float(np.mean([s['f_mse'] for s in scores]))})
            save(out/f'layout{layout}_learning_history.json', {'execution': spec, 'rows': history})
            print('source predictive scale', layout, len(history), float(scale), value, flush=True)
            return -value
        result = minimize_scalar(objective, bounds=spec['scale_bounds'], method='bounded',
            options={'xatol': spec['optimizer_xatol'], 'maxiter': spec['optimizer_max_iterations']})
        # Endpoints are part of this single bounded fit,not extra starts.
        options = [(float(result.x), float(result.fun)), (0., objective(0.)), (1., -baseline_density)]
        selected_scale, loss = min(options, key=lambda p: p[1]); selected_rows, checks, source_rows = [], [], []
        for case in members:
            native = scaled_source_law(case['inputs'], case['frozen_law'], selected_scale)
            record, law = finite_reference(case['inputs'], native, context); counters['source4_integrations'] += 1
            actual, p, state, bound = forecast(case, law, record['adaptive_u_intervals'], spec['learning_nodes_per_interval']); counters['cv_forecasts'] += 1
            refined, rp, _, rb = forecast(case, law, record['adaptive_u_intervals'], spec['final_refinement_nodes_per_interval']); counters['refinement_forecasts'] += 1
            differences = errors(p, rp); density_error = abs(actual['f_raw_batch_density']-refined['f_raw_batch_density'])
            passed = all(differences[k] <= limit for k, limit in spec['prediction_tolerances'].items()) and density_error <= spec['density_refinement_tolerance']
            checks.append({'query_task': case['query_task'], 'errors': differences, 'raw_density_refinement_error': density_error, 'passed': passed})
            filename = f"layout{layout}_task{case['query_task']}_selected.npz"; np.savez_compressed(out/filename, **state)
            selected_rows.append({'layout': layout, 'query_task': case['query_task'], 'scale': selected_scale, 'prediction': p,
                'scores': actual, 'refined_scores': refined, 'component_file': filename, 'density_integration_error_bound': bound})
            final_rules.append({'layout': layout, 'query_task': case['query_task'], 'scale': selected_scale,
                'source_reference_law': law, 'adaptive_u_intervals': record['adaptive_u_intervals']})
            source_rows.append(record)
        final_density = float(np.mean([r['scores']['f_raw_batch_density'] for r in selected_rows]))
        np.testing.assert_allclose(final_density, -loss, atol=1e-8, rtol=1e-8)
        record = {'layout': layout, 'scale': selected_scale, 'optimizer_converged': bool(result.success), 'optimizer_message': str(result.message),
            'bounded_optimizer_evaluations': int(result.nfev), 'actual_objective_evaluations': len(history),
            'baseline_scores': {k: float(np.mean([r['scores'][k] for r in baseline])) for k in baseline[0]['scores']},
            'selected_scores': {k: float(np.mean([r['scores'][k] for r in selected_rows])) for k in selected_rows[0]['scores']},
            'numerical_checks': checks, 'proper_density_gain': final_density-baseline_density,
            'scale_learned_from_source_only': True}
        records.append(record); final_forecasts += selected_rows
        save(out/'partial_results.json', {'execution': spec, 'rows': records, 'cost_counters': counters})
    verified = all(r['optimizer_converged'] and all(c['passed'] for c in r['numerical_checks']) for r in records)
    save(out/'inner_source_rules.json', {'execution': spec, 'rows': final_rules, 'all_selected_source4_rules_complete': verified})
    save(out/'inner_predictions.json', {'execution': spec, 'rows': final_forecasts, 'selection_training_only': True})
    full_records, full_rules = [], []
    if verified:
        original = read(spec['full_source_rules']); old_reference = read(spec['full_source_reference_results'])
        reference_spec = read(spec['full_source_reference_execution'])
        for fit in records:
            layout = fit['layout']; old_rule = next(r for r in original['rows'] if r['layout'] == layout)
            inputs = read(Path(coverage['output'])/f'layout{layout}'/'task1'/'inputs.json')
            native = scaled_source_law(inputs, old_rule['source_reference_law']['native_law'], fit['scale'])
            head = {'regularizer': {'coefficient_mean': native['empirical_center_coefficient_mean']}, 'joint_structural': native}
            previous = next(r for r in old_reference['rows'] if r['layout'] == layout)
            previous = {'conditional_source_likelihood_support_variances': [native['residual_coefficient_diagonal'][0], previous['stage105_slice_upper_variance']]}
            record, law = diagnose(inputs, head, previous, 1, reference_spec)
            _, _, _, intervals = positive_base_rule(law, record, reference_spec)
            selected = None
            for size in (4, 8, 16):
                nodes, weights = rule_on_intervals(law, intervals, size)
                error = approximation_errors(nodes, weights, law, record)
                if passes(error, law, record, reference_spec):
                    selected = {'layout': layout, 'deleted_task': 1, 'scale': fit['scale'], 'source_reference_law': law,
                        'variance_nodes': nodes.tolist(), 'posterior_weights': (weights/weights.sum()).tolist(), 'adaptive_u_intervals': intervals,
                        'nodes_per_interval': size, 'errors': error}; break
            full_records.append({'layout': layout, 'scale': fit['scale'], **record, 'rule_complete': selected is not None})
            if selected: full_rules.append(selected)
    complete = verified and len(full_rules) == 2
    save(out/'full_source_rules.json', {'execution': spec, 'both_source5_laws_frozen_without_task1_values': complete, 'rows': full_rules})
    result = {'execution': spec, 'rows': records, 'full_source_rows': full_records, 'all_source_predictive_fits_and_rules_complete': complete,
        'cost': {**spec['cost'], 'actual_source_predictive_scalar_fits': len(records), **counters, 'source5_integrations': len(full_records)},
        'cpu_seconds': time.process_time()-started, 'no_model_adopted': True, 'outer_task1_query_outcomes_used': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'training_diagnostic_only': True, 'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))}
    save(out/'results.json', result)
    print('source scale learner complete', [{'layout': r['layout'], 'scale': r['scale'], 'proper_density_gain': r['proper_density_gain']} for r in records], complete, 'CPU', result['cpu_seconds'], flush=True)


if __name__ == '__main__':
    run()
