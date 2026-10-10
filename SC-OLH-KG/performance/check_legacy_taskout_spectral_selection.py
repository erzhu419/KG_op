"""Whole-source-task predictive selection of a single common-spectrum ablation."""

import copy
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_taskout_spectral_selection_v1.json'


def fit_zero_common_spectrum(inputs, old, head, opt):
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
    prior = restore_regularizer(basis, old['regularizer'], tasks, head)
    likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, head), [t['noise'][:, head] for t in tasks])
    native = old['joint_structural']; initial = np.asarray(native['optimizer_parameters']).copy()
    np.testing.assert_allclose(likelihood.evaluate(initial)[0], native['source_negative_log_likelihood_per_label'], rtol=1e-8, atol=1e-8)
    count = len(basis.group_names); initial[count] = 0.
    bounds = [(0., None)]*count+[(0., 0.)]+[(0., None)]*(count+1)+[(0., 4.), (0., None)]
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True, method=opt['method'], bounds=bounds,
        options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    info = copy.deepcopy(native)
    info.update(coefficient_mean=state['mean'].tolist(), population_mean_covariance=state['mean_covariance'].tolist(),
        common_prior_coefficient_diagonal=state['common_diagonal'].tolist(), common_group_variances=state['common_group_variances'].tolist(),
        source_group_variances=state['group_variances'].tolist(), common_zero_group_count=int(np.count_nonzero(state['common_group_variances'] == 0)),
        source_zero_group_count=int(np.count_nonzero(state['group_variances'] == 0)), common_spectral_exponent=state['common_spectral_exponent'],
        spectral_exponent=state['spectral_exponent'], residual_coefficient_diagonal=state['residual_diagonal'].tolist(),
        discrepancy_variance=state['discrepancy'], optimizer_parameters=result.x.tolist(), optimizer_initial_parameters=initial.tolist(),
        optimizer_converged=bool(result.success), optimizer_iterations=int(result.nit), optimizer_message=str(result.message),
        source_negative_log_likelihood_per_label=value, initial_source_negative_log_likelihood_per_label=likelihood.evaluate(initial)[0],
        common_spectrum_fixed_for_ablation=0.)
    upper = np.full(len(result.x), np.inf); upper[count] = 0.; upper[-2] = 4.
    projected = np.where(((result.x <= 0) & (gradient > 0)) | ((result.x >= upper) & (gradient < 0)), 0., gradient)
    projected[count] = 0.
    info['optimizer_projected_gradient_max'] = float(np.max(np.abs(projected)))
    assert info['common_spectral_exponent'] == 0.
    return info


def gaussian_score(mean, covariance, label, noise):
    error = np.asarray(label)-np.asarray(mean); variance = np.asarray(covariance)+np.diag(noise)
    factor = cho_factor(variance, lower=True)
    return float(-.5*(len(error)*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@cho_solve(factor, error)))


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    deleted = read(spec['deleted_mean_execution']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, candidates, validation, full_inputs = {}, {}, [], {}; new_fits = 0
    for layout in spec['layouts']:
        folder = Path(source['output'])/f'layout{layout}'
        group = [(read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json'))]
        group += [(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
                   read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        inp, native, _ = group[0]; groups[layout] = group; full_inputs[layout] = inp
        branches = [(inp, native, 'full')]+[(g[0], read(Path(deleted['output'])/f'layout{layout}'/f'task{i}'/'source_fit.json'), str(i))
            for i, g in zip(source['training_tasks'], group[1:])]
        for branch_input, fit, identity in branches:
            path = out/f'layout{layout}'/identity; path.mkdir(parents=True, exist_ok=True)
            if (path/'source_fit.json').exists():
                candidate = read(path/'source_fit.json')
            else:
                candidate = copy.deepcopy(fit)
                candidate['heads'][1]['joint_structural'] = fit_zero_common_spectrum(branch_input, fit['heads'][1], 1, inp['execution']['optimizer'])
                new_fits += 1; save(path/'source_fit.json', candidate)
            if not candidate['heads'][1]['joint_structural']['optimizer_converged']:
                raise RuntimeError('constrained source fit incomplete; retain artifact and stop before selection')
            if identity == 'full':
                candidates[layout] = {'native': native, 'zero': candidate}; continue
            i = int(identity); position = source['training_tasks'].index(i); task = inp['source_tasks'][position]
            assert all(t['name'] != task['name'] for t in branch_input['source_tasks'])
            # Invert full public-policy normalization, then apply only the fold's own normalization.
            raw = np.asarray(task['psi'])*inp['mean_coordinate_scale']+inp['mean_coordinate_center']
            psi = (raw-branch_input['mean_coordinate_center'])/branch_input['mean_coordinate_scale']
            basis = OrthogonalLowOrderBasis(branch_input['public_mean_reference'])
            for name, fitted in (('native', fit), ('zero', candidate)):
                mean, covariance = source_prior_moments(restore_joint(basis, fitted['heads'][1]['joint_structural']), psi)
                validation.append({'layout': layout, 'task_index': i, 'task_id': task['name'], 'candidate': name,
                    'mean': mean.tolist(), 'covariance': covariance.tolist()})
    save(out/'taskout_predictions.json', {'execution': spec, 'rows': validation})
    for row in validation:
        position = source['training_tasks'].index(row['task_index']); task = full_inputs[row['layout']]['source_tasks'][position]
        row['log_density'] = gaussian_score(row['mean'], row['covariance'], np.asarray(task['response'])[:, 1], np.asarray(task['noise'])[:, 1])
    selections = {}; fits = {}
    for layout in spec['layouts']:
        scores = {name: sum(r['log_density'] for r in validation if r['layout'] == layout and r['candidate'] == name) for name in ('native', 'zero')}
        chosen = 'zero' if scores['zero'] > scores['native'] else 'native'
        selections[layout] = {'scores': scores, 'selected': chosen}; fits[layout] = candidates[layout][chosen]
    save(out/'taskout_scores_selection.json', {'execution': spec, 'rows': validation, 'selections': selections})
    # Neither heldout source task6/7 has been read or used by fits or selection above.
    data = read(source['source_data']); public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
    rows, jobs = [], {}
    for layout, group in groups.items():
        for i in source['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
                'tau': task['tau'], 'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            for arm, model in risk_models(group, fits[layout], group[0][0]['heldout_mean_psi'][str(i)],
                    [g[0]['heldout_risk_psi'][str(i)] for g in group], spec['mean_arms']):
                prior = forecast(query, arm, 'source_taskout_common_spectral_selection', model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, arm, 'source_taskout_common_spectral_selection', model)
                row.update(layout=layout, prior=prior); rows.append(row); jobs[(layout, i)] = model
    save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in source['heldout_tasks']}
    reference = read(spec['reference_profiles'])['rows']; contrasts = []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], jobs[(layout, i)], 3)
        old = next(r for r in reference if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_structural')
        c = {'layout': layout, 'task_id': row['task_id'],
            'g_mse_difference': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]}
        c['passed'] = c['g_mse_difference'] <= 0 and c['joint_batch_score_difference'] >= 0 and c['g_batch_score_difference'] > 0
        contrasts.append(c)
    summary = {layout: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == layout], axis=0).tolist()
        for k in rows[0]['scores']} for layout in spec['layouts']}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    result = {'execution': spec, 'selections': selections, 'summary': summary, 'contrasts': contrasts,
        'source_gate_passed': all(c['passed'] for c in contrasts),
        'cost': {'distinct_mean_fit_models': 14, 'mean_fits_this_run': new_fits, 'reused_completed_candidate_fits': 14-new_fits,
            'new_risk_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print(result, flush=True)


if __name__ == '__main__':
    run()
