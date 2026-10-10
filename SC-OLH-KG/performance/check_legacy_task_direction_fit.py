"""Source-fitted scalar task covariance factors with frozen risk controls."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_joint_population_matched_baselines import paired_fields
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_function_prior import SourceFunctionPrior, head_batches
from representation.source_task_covariance import SourceTaskDirectionLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_task_direction_fit_v1.json'


def fit_directions(inputs, native, directions):
    started = time.process_time(); basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); fitted = {'heads': []}
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    opt = inputs['execution']['optimizer']
    for h, head in enumerate(native['heads']):
        original = head['joint_structural']; direction = directions[h]['direction']
        prior = restore_regularizer(basis, head['regularizer'], tasks, h)
        model = SourceTaskDirectionLikelihood(prior, head_batches(tasks, h), [t['noise'][:, h] for t in tasks],
            original['common_prior_coefficient_diagonal'], original['residual_coefficient_diagonal'], original['discrepancy_variance'], direction)
        value, gradient, state = model.evaluate(0.)
        np.testing.assert_allclose(value, original['source_negative_log_likelihood_per_label'], atol=1e-8)
        np.testing.assert_allclose(state['mean'], original['coefficient_mean'], atol=1e-8)
        np.testing.assert_allclose(gradient[0], directions[h]['rank_one_variance_source_nll_derivative'], atol=1e-8)
        result = minimize(lambda p: model.evaluate(p[0])[:2], [0.], jac=True, method=opt['method'], bounds=[(0., None)],
            options={'maxiter': opt['max_iterations'], 'ftol': opt['ftol'], 'gtol': opt['gtol']})
        value, gradient, state = model.evaluate(result.x[0])
        fitted['heads'].append({**basis.diagnostics(), 'mode': 'source_group_spectrum_with_task_direction',
            'source_label_count': model.label_count, 'source_task_count': len(tasks),
            'common_prior_coefficient_diagonal': original['common_prior_coefficient_diagonal'],
            'base_residual_coefficient_diagonal': original['residual_coefficient_diagonal'],
            'base_common_group_variances': original['common_group_variances'], 'base_common_spectral_exponent': original['common_spectral_exponent'],
            'base_task_group_variances': original['source_group_variances'], 'base_task_spectral_exponent': original['spectral_exponent'],
            'task_direction': direction, 'task_direction_scale': float(result.x[0]),
            'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
            'residual_coefficient_covariance': state['residual_covariance'].tolist(), 'discrepancy_variance': state['discrepancy'],
            'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit), 'optimizer_message': str(result.message),
            'optimizer_gradient': float(gradient[0]), 'source_negative_log_likelihood_per_label': value,
            'native_source_negative_log_likelihood_per_label': original['source_negative_log_likelihood_per_label']})
    fitted['cpu_seconds'] = time.process_time()-started
    return fitted


def source_model(inputs, means, risk_members, task_index):
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); psi = inputs['heldout_mean_psi'][str(task_index)]
    moments = [source_prior_moments(SourceFunctionPrior(basis, np.asarray(h['coefficient_mean']),
        np.asarray(h['population_mean_covariance'])+np.asarray(h['residual_coefficient_covariance']),
        h['discrepancy_variance'], h), psi) for h in means['heads']]
    fields = []
    for view, risk in risk_members:
        rb = OrthogonalLowOrderBasis(view['public_risk_reference'])
        eta = [source_prior_moments(restore_joint(rb, h['joint_structural']), view['heldout_risk_psi'][str(task_index)]) for h in risk['heads']]
        fields.append(paired_fields(moments+eta))
    return JointResponseMixture(['full144']+[f'drop{i}' for i in range(6)], fields)


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution']); source = read(coverage['source_execution'])
    directions = read(spec['direction_results'])['rows']; out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    fits, inputs, risks = {}, {}, {}
    for layout in spec['layouts']:
        folder = Path(coverage['output'])/f'layout{layout}'; view = read(folder/'inputs.json')
        local = out/f'layout{layout}'; local.mkdir(parents=True, exist_ok=True)
        fit = fit_directions(view, read(folder/'source_fit.json'), [next(r for r in directions if r['layout'] == layout and r['head'] == h) for h in ('f', 'g')])
        save(local/'source_fit.json', fit)
        if not all(h['optimizer_converged'] for h in fit['heads']):
            raise RuntimeError('scalar source direction fit incomplete; no forecasts')
        inputs[layout], fits[layout] = view, fit
        old = Path(source['output'])/f"layout{spec['parent_layouts'][layout]}"
        risks[layout] = [(read(old/'inputs.json'), read(old/'risk_source_fit.json'))]+[
            (read(old/f'task{i}'/'inputs.json'), read(old/f'task{i}'/'risk_source_fit.json')) for i in spec['training_tasks']]
    data = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    rows, jobs = [], {}
    for layout in spec['layouts']:
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            model = source_model(inputs[layout], fits[layout], risks[layout], i)
            prior = forecast(query, 'joint_structural', 'source_task_direction_covariance', model)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                model.update_raw(j, [value[0], value[1]-query['tau']])
            row = forecast(query, 'joint_structural', 'source_task_direction_covariance', model)
            row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior)
            rows.append(row); jobs[(layout, i)] = model
    save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    references = read(spec['reference_profiles'])['rows']; labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    contrasts = []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], jobs[(layout, i)], 3)
        old = next(r for r in references if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == 'joint_structural')
        c = {'layout': layout, 'task_id': row['task_id'],
            'g_mse_difference': row['scores']['observed_head_mse'][1]-old['scores']['observed_head_mse'][1],
            'joint_batch_score_difference': row['scores']['joint_batch_log_score']-old['scores']['joint_batch_log_score'],
            'g_batch_score_difference': row['scores']['response_batch_log_scores'][1]-old['scores']['response_batch_log_scores'][1]}
        c['passed'] = c['g_mse_difference'] <= 0 and c['joint_batch_score_difference'] >= 0 and c['g_batch_score_difference'] > 0
        contrasts.append(c)
    summary = {layout: {k: np.mean([r['scores'][k] for r in rows if r['layout'] == layout], axis=0).tolist()
        for k in rows[0]['scores']} for layout in spec['layouts']}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'source_gate_passed': all(c['passed'] for c in contrasts),
        'fits': [{'layout': l, 'heads': [{k: h[k] for k in ('task_direction_scale', 'source_negative_log_likelihood_per_label',
            'native_source_negative_log_likelihood_per_label', 'optimizer_iterations', 'optimizer_converged')} for h in f['heads']]} for l, f in fits.items()],
        'cost': {'source_calls_per_model': 216, 'new_scalar_source_fits': 4, 'new_risk_fits': 0,
            'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0}, 'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print(result, flush=True)


if __name__ == '__main__':
    run()
