"""Matched raw-prefix pilots retaining source task-amplitude mixtures."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction, errors, score
from performance.check_legacy_f_source_task_amplitude_uncertainty import amplitude_rule, source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_level_shape_mean_error import decompose
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.public_control_mean import public_control_mean_coordinates
from representation.vectorized_joint_function_risk import update_raw_observation

PLAN = 'performance/manifests/legacy_joint_population_f_inner_task_amplitude_raw_plan_v1.json'


def amplitude_field(design, rule, white, eta):
    k, n = len(rule['states']), len(design); means = np.empty((k, n, 2)); covariance = np.zeros((k, n, 2, n, 2))
    means[:, :, 1] = eta[0]; covariance[:, :, 1, :, 1] = eta[1]
    for j, state in enumerate(rule['states']):
        means[j, :, 0] = design @ state['source_posterior_coefficient_mean']
        covariance[j, :, 0, :, 0] = design @ (state['source_posterior_common_covariance'] + state['task_coefficient_covariance']) @ design.T + white * np.eye(n)
    field = JointFunctionRiskPosterior.from_moments(means[0], covariance[0])
    field.means, field.covariances = means.reshape(k, 2 * n), covariance.reshape(k, 2 * n, 2 * n)
    field.log_weights = np.log(rule['posterior_weights']); field.component_pairs = [(0, 0)] * k
    return field


def run():
    started = time.process_time(); spec = read(PLAN); source = read(spec['source_amplitude_results']); assert source['diagnostic_interpreted']
    amplitude_spec = read(spec['source_amplitude_plan']); model_spec = read(amplitude_spec['source_probe_plan'])
    controls, point = read(spec['control_results']), read(spec['point_control_results'])
    assert controls['diagnostic_interpreted'] and point['diagnostic_interpreted']
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in controls['execution']['source_packages']], [])
    coverage, _ = source_paths(read(spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); fields, frozen = [], []
    for layout in spec['layouts']:
        for query in spec['query_tasks']:
            model, law, _ = source_model(model_spec, layout, query)
            prior = next(r for r in source['rows'] if (r['layout'], r['query_task']) == (layout, query)); assert prior['numerical_passed']
            assert prior['finite_moment_verification']['posterior_amplitude_second_moment_finite']
            branch = next(b for b in branches if (b['layout'], b['query_task']) == (layout, query)); assert branch['source_task_names'] == law['source_task_names']
            inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
            psi = (public_control_mean_coordinates(no_outcomes(problem_for(public['plan'], public['tasks'][query])), bank)
                - inputs['mean_coordinate_center']) / inputs['mean_coordinate_scale']
            np.testing.assert_allclose(psi[coverage['layouts'][layout][query]], branch['query_mean_psi'], atol=1e-12, rtol=1e-12)
            sites = branch['scoring_sites']; selection = spec['paid_prefix'] + sites
            design = np.column_stack([np.ones(len(selection)), model.basis.transform(psi[selection])])
            eta = np.asarray(branch['eta_mean'])[selection], np.asarray(branch['eta_covariance'])[np.ix_(selection, selection)]
            for saved in prior['rules']:
                rule = amplitude_rule(model, law['parameters'], saved['order_per_interval'])
                for key in ('amplitude_nodes', 'source_nll_per_label', 'posterior_weights'):
                    np.testing.assert_allclose(rule[key], saved[key], atol=1e-12, rtol=1e-10)
                row = {'layout': layout, 'query_task': query, 'arm': 'task_amplitude_mixture', 'design': 'fixed_public_prefix',
                    'episode_count': 4, 'paid_prefix': spec['paid_prefix'], 'scoring_sites': sites, 'order_per_interval': saved['order_per_interval'],
                    'component_count': len(rule['states'])}
                fields.append({'row': row, 'field': amplitude_field(design, rule, law['white_discrepancy'], eta)})
                frozen.append({**row, 'source_task_names': law['source_task_names'], 'point_parameters': law['parameters'],
                    'source_log_evidence': rule['log_evidence'], 'source_likelihoods_and_weights_reproduced': True})
    assert len(fields) == 2 * len(spec['layouts']) * len(spec['query_tasks'])
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': frozen, 'all_priors_frozen_before_paid_values': True})
    raw = read(coverage['source_data'])
    paid = {q: [raw['tasks'][q]['Y_replicates'][j][0][0] for j in spec['paid_prefix']] for q in spec['query_tasks']}; del raw
    rows, numerical = [], []
    for work in fields:
        row, field = work['row'], work.pop('field')
        evidence = sum(update_raw_observation(field, j, float(y)) for j, y in enumerate(paid[row['query_task']]))
        state = compact(field, list(range(4, 12)), evidence)
        filename = f"layout{row['layout']}_task{row['query_task']}_order{row['order_per_interval']}.npz"
        np.savez_compressed(out / filename, **state)
        row.update(prediction=prediction(state), component_file=filename, max_raw_integration_error=field.max_integration_relative_error); rows.append(row)
    for layout in spec['layouts']:
        for query in spec['query_tasks']:
            pair = [r for r in rows if (r['layout'], r['query_task']) == (layout, query)]
            e = errors(pair[0]['prediction'], pair[1]['prediction'])
            numerical.append({'layout': layout, 'query_task': query, 'prediction_errors': e,
                'prediction_passed': all(e[k] <= v for k, v in spec['prediction_tolerances'].items())})
    save(out / 'predictions.json', {'execution': spec, 'rows': rows, 'all_forecasts_frozen_before_score_labels': True, 'numerical_checks': numerical})
    base = {'execution': spec, 'rows': rows, 'numerical_checks': numerical, 'cost': spec['cost'], 'no_model_adopted': True,
        'point_candidate_remains_stopped': True, 'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    if not all(r['prediction_passed'] for r in numerical):
        save(out / 'results.json', {**base, 'diagnostic_interpreted': False, 'forecasts_scored': False, 'cpu_seconds': time.process_time() - started}); return
    for row in rows:
        if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']:
            save(out / 'results.json', {**base, 'diagnostic_interpreted': False, 'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
        inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
        task = next(t for t in inputs['source_tasks'] if t['name'] == f"related_inventory_{row['query_task']:02d}")
        labels = {k: np.asarray(task[k])[4:] for k in ('response', 'noise')}
        with np.load(out / row['component_file']) as state: row['scores'], row['density_integration_error_bound'] = score(row, state, labels)
        row['mean_error'] = decompose(row['prediction']['mean'], labels['response'][:, 0])
    for check in numerical:
        pair = [r for r in rows if (r['layout'], r['query_task']) == (check['layout'], check['query_task'])]
        e = {k: abs(pair[0]['scores'][k] - pair[1]['scores'][k]) for k in spec['metrics']}
        check.update(score_errors=e, score_passed=all(e[k] <= v for k, v in spec['score_tolerances'].items()))
    if not all(c['score_passed'] for c in numerical):
        save(out / 'results.json', {**base, 'diagnostic_interpreted': False, 'forecasts_scored': True, 'cpu_seconds': time.process_time() - started}); return
    selected = [r for r in rows if r['order_per_interval'] == 16]
    for row in selected:
        row['comparisons'] = {}
        for control in spec['controls']:
            reference = next(r for r in (point['rows'] if control == 'joint_spectral_geometry' else controls['rows'])
                if (r['layout'], r['query_task'], r['arm'], r['design']) == (row['layout'], row['query_task'], control, row['design']))
            assert (reference['paid_prefix'], reference['scoring_sites']) == (row['paid_prefix'], row['scoring_sites'])
            delta = {k: row['scores'][k] - reference['scores'][k] for k in spec['metrics']}
            row['comparisons'][control] = {'differences': delta, 'mse_relative_change': delta['f_mse'] / reference['scores']['f_mse']}
    advances = all((r['comparisons']['joint_spectral_geometry']['differences']['f_mse'] < 0
        or r['comparisons']['joint_spectral_geometry']['differences']['f_raw_batch_density'] > 0)
        and not (r['comparisons']['joint_spectral_geometry']['differences']['f_mse'] > 0
            and r['comparisons']['joint_spectral_geometry']['differences']['f_raw_batch_density'] < 0
            and r['comparisons']['joint_spectral_geometry']['differences']['sequential_transcript_density'] < 0) for r in selected)
    save(out / 'results.json', {**base, 'diagnostic_interpreted': True, 'forecasts_scored': True, 'canonical_screening_passed': advances,
        'complete_tenfold_inner_comparison_registered': False, 'cpu_seconds': time.process_time() - started,
        'component_artifact_bytes': sum(p.stat().st_size for p in out.glob('*.npz'))})
    print('amplitude raw pilot screening', advances, [(r['layout'], r['comparisons']) for r in selected], flush=True)


if __name__ == '__main__':
    run()
