"""Source-gated four-field paid filtering for the frozen paired mean prior."""

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_counterbalanced_complete_mean_uncertainty import contrast
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.paired_function_risk import PairedFunctionRiskMixture, INTEGRATION_ORDERS, INTEGRATION_TOLERANCE
from representation.public_control_mean import source_prior_moments

EXECUTION = 'performance/manifests/legacy_joint_population_paired_function_risk_filter_execution_v2.json'


def paired_mean_moments(inputs, fit, task):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    psi = inputs['heldout_mean_psi'][str(task)]; design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    n, width = design.shape; coefficients = np.asarray(fit['coefficient_mean']).reshape(2, width)
    prior = np.asarray(fit['population_mean_covariance'])+np.asarray(fit['residual_coefficient_covariance'])
    mean = design@coefficients.T; covariance = np.empty((n, 2, n, 2))
    for h in range(2):
        for k in range(2):
            covariance[:, h, :, k] = design@prior[h*width:(h+1)*width, k*width:(k+1)*width]@design.T
            if h == k:
                covariance[:, h, :, k] += fit['discrepancy_by_head'][h]*np.eye(n)
    return mean, covariance


def independent_mean_moments(inputs, native, task):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    moments = [source_prior_moments(restore_joint(basis, h['joint_structural']), inputs['heldout_mean_psi'][str(task)])
        for h in native['heads']]
    n = len(moments[0][0]); covariance = np.zeros((n, 2, n, 2))
    for h in range(2):
        covariance[:, h, :, h] = moments[h][1]
    return np.column_stack([m[0] for m in moments]), covariance


def identity_model(mean_moments, members, task):
    fg_mean, fg_covariance = mean_moments; n = len(fg_mean); means, covariances = [], []
    for inputs, fit in members:
        mean = np.empty((n, 4)); covariance = np.zeros((n, 4, n, 4))
        mean[:, [0, 2]] = fg_mean
        for h in range(2):
            for k in range(2):
                covariance[:, 2*h, :, 2*k] = fg_covariance[:, h, :, k]
        basis = OrthogonalLowOrderBasis(inputs['public_risk_reference'])
        # Existing risk fits are eta_g then eta_f, rather than f/g order.
        for h, slot in ((0, 3), (1, 1)):
            m, c = source_prior_moments(restore_joint(basis, fit['heads'][h]['joint_structural']),
                inputs['heldout_risk_psi'][str(task)])
            mean[:, slot], covariance[:, slot, :, slot] = m, c
        means.append(mean); covariances.append(covariance)
    return PairedFunctionRiskMixture(['full144']+[f'drop{i}' for i in range(6)], means, covariances)


def forecast(query, model):
    mean, covariance = model.predict_joint(); n = len(mean)
    variance = np.diag(covariance.reshape(4*n, 4*n)).reshape(n, 4)
    return {'task_id': query['task_id'], 'task_index': query['task_index'], 'kind': 'source',
        'mean_arm': 'joint_structural', 'risk_case': 'paired_source_mean_with_complete_risk_identities',
        'head_order': ['f', 'g', 'eta_g', 'eta_f'], 'mean': mean[:, [0, 2, 3, 1]].tolist(),
        'variance': variance[:, [0, 2, 3, 1]].tolist(),
        'expected_episode_variance': model.expected_episode_variance().tolist(),
        'identity_names': model.names, 'identity_weights': model.weights.tolist(),
        'paid_predictive_log_score': float(sum(r['mixture_log_score'] for r in model.history)),
        'paid_history': list(model.history),
        'within_identity_fg_cross_covariance_norm': [float(np.linalg.norm(c[0::4, 2::4])) for c in model.covariances]}


def check_zero(query, model, reference):
    prior = forecast(query, model)
    for key in ('mean', 'variance', 'expected_episode_variance', 'identity_weights'):
        np.testing.assert_allclose(prior[key], reference['prior'][key], atol=1e-8, rtol=1e-8)
    for j, value in zip(query['candidate_indices'], query['paired_observations']):
        model.update_raw(j, [value[0], value[1]-query['tau']])
    actual = forecast(query, model)
    for key in ('mean', 'variance', 'expected_episode_variance', 'identity_weights', 'paid_predictive_log_score'):
        np.testing.assert_allclose(actual[key], reference[key], atol=1e-8, rtol=1e-8)


def score(row, labels, model):
    mean, variance = np.asarray(row['mean']), np.asarray(row['variance']); error = mean-np.asarray(labels['response'])
    observation_variance = variance.copy(); observation_variance[:, :2] += np.asarray(row['expected_episode_variance'])/3
    observation_variance[:, 2:] += labels['noise'][:, 2:]
    scores = [model.batch_log_scores(j, labels['response'][j, :2], labels['response'][j, [3, 2]], 3)
        for j in range(2, len(mean))]
    row['scores'] = {'observed_head_mse': (error[2:]**2).mean(axis=0).tolist(),
        'joint_batch_log_score': float(np.mean([s['joint'] for s in scores])),
        'response_batch_log_scores': np.mean([s['heads'] for s in scores], axis=0).tolist(),
        'predictive_moment_band_95_observed_coverage': (abs(error[2:])/np.sqrt(observation_variance[2:]) <= 1.96).mean(axis=0).tolist(),
        'predictive_standardized_residual_rms': np.sqrt((error[2:]**2/observation_variance[2:]).mean(axis=0)).tolist()}
    row['scoring_quadrature'] = {'max_integration_error': max(s['max_integration_error'] for s in scores),
        'orders': dict(Counter(o for s in scores for o in s['integration_orders']))}


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    assert spec['integration_orders'] == list(INTEGRATION_ORDERS)
    assert spec['integration_tolerance'] == INTEGRATION_TOLERANCE
    source = read(coverage['source_execution']); paired = read(spec['paired_source_execution'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    references, parents = read(spec['reference_profiles'])['rows'], read(spec['parent_reference_profiles'])['rows']
    data = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    rows, jobs, zeros = [], {}, []
    for layout in spec['layouts']:
        folder = Path(coverage['output'])/f'layout{layout}'
        inputs, native = read(folder/'inputs.json'), read(folder/'source_fit.json')
        fitted = read(Path(paired['output'])/f'layout{layout}'/'source_fit.json')
        old = Path(source['output'])/f"layout{spec['parent_layouts'][layout]}"
        members = [(read(old/'inputs.json'), read(old/'risk_source_fit.json'))]+[
            (read(old/f'task{i}'/'inputs.json'), read(old/f'task{i}'/'risk_source_fit.json')) for i in spec['training_tasks']]
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'tau': task['tau'],
                'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            reference = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_structural')
            check_zero(query, identity_model(independent_mean_moments(inputs, native, i), members, i), reference)
            zeros.append({'layout': layout, 'task_id': task['task_id'], 'passed': True})
            model = identity_model(paired_mean_moments(inputs, fitted, i), members, i)
            prior = forecast(query, model)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                model.update_raw(j, [value[0], value[1]-query['tau']])
            row = forecast(query, model); row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior)
            rows.append(row); jobs[(layout, i)] = model
            print('paired paid prediction frozen', layout, task['task_id'], flush=True)
    save(out/'source_predictions.json', {'execution': spec, 'all_forecasts_frozen': True,
        'zero_coupling_reproduction': zeros, 'rows': rows})
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    contrasts, parent_contrasts = [], []
    reused = read(spec['reused_scoring_progress'])['rows']
    for row in rows:
        layout, i = row['layout'], row['task_index']
        cached = next((r for r in reused if r['layout'] == layout and r['task_index'] == i), None)
        if cached is None:
            score(row, labels[i], jobs[(layout, i)])
        else:
            for key in ('mean', 'variance', 'expected_episode_variance', 'identity_weights', 'paid_predictive_log_score'):
                np.testing.assert_array_equal(row[key], cached[key])
            row['scores'], row['scoring_quadrature'] = cached['scores'], cached['scoring_quadrature']
            row['reused_score_execution'] = spec['repaired_execution']
        previous = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_structural')
        parent = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == 'joint_structural')
        contrasts.append(contrast(row, previous, 'same216_call_structural'))
        parent_contrasts.append(contrast(row, parent, 'original144_call_structural'))
        save(out/'scoring_progress.json', {'completed': len(contrasts), 'rows': rows[:len(contrasts)]})
        print('paired source score completed', layout, row['task_id'], flush=True)
    summary = {layout: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == layout], axis=0).tolist()
        for k in rows[0]['scores']} for layout in spec['layouts']}
    current_gate, parent_gate = all(c['passed'] for c in contrasts), all(c['passed'] for c in parent_contrasts)
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'same216_source_gate_passed': current_gate, 'parent144_source_gate_passed': parent_gate,
        'structural_advance_gate_passed': current_gate and parent_gate, 'zero_coupling_reproduction': zeros,
        'cost': {'new_source_fits': 0, 'new_risk_fits': 0, 'new_simulator_calls': 0,
            'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({k: result[k] for k in ('summary', 'contrasts', 'parent_contrasts', 'structural_advance_gate_passed', 'cpu_seconds')}, flush=True)


if __name__ == '__main__':
    run()
