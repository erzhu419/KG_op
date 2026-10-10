"""Condition the source posterior on extra sites without covariance refitting."""

import copy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood

EXECUTION = 'performance/manifests/legacy_joint_population_counterbalanced_frozen_covariance_v1.json'


def condition_means(prefix_inputs, prefix_fit, augmented):
    basis = OrthogonalLowOrderBasis(prefix_inputs['public_mean_reference']); fitted = {'heads': []}; contraction = []
    tasks = lambda view: [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in view['source_tasks']]
    old_tasks, new_tasks = tasks(prefix_inputs), tasks(augmented)
    for h, head in enumerate(prefix_fit['heads']):
        prior = restore_regularizer(basis, head['regularizer'], old_tasks, h); original = head['joint_structural']
        parameters = np.asarray(original['optimizer_parameters'])
        old_likelihood = JointCovariancePopulationLikelihood(prior, head_batches(old_tasks, h), [t['noise'][:, h] for t in old_tasks])
        _, _, old = old_likelihood.evaluate(parameters)
        np.testing.assert_allclose(old['mean'], original['coefficient_mean'], atol=1e-8)
        np.testing.assert_allclose(old['mean_covariance'], original['population_mean_covariance'], atol=1e-8)
        likelihood = JointCovariancePopulationLikelihood(prior, head_batches(new_tasks, h), [t['noise'][:, h] for t in new_tasks])
        value, _, state = likelihood.evaluate(parameters)
        difference = old['mean_covariance']-state['mean_covariance']
        minimum = float(np.linalg.eigvalsh((difference+difference.T)/2).min())
        if minimum < -1e-10:
            raise RuntimeError('added same-law source observations increased common covariance; check prefix and parameter reconstruction')
        contraction.append({'head': h, 'minimum_covariance_reduction_eigenvalue': minimum,
            'common_covariance_trace_before': float(np.trace(old['mean_covariance'])),
            'common_covariance_trace_after': float(np.trace(state['mean_covariance']))})
        info = {**basis.diagnostics(), 'mode': 'frozen_source_covariance_sequential_conditioning',
            'population_model': 'original source common/task law conditioned on72 source records',
            'source_label_count': likelihood.label_count, 'source_task_count': len(new_tasks),
            'source_signal_scale': likelihood.signal, 'tail_variance_budget': .01*likelihood.signal,
            'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
            'common_prior_coefficient_diagonal': state['common_diagonal'].tolist(),
            'residual_coefficient_diagonal': state['residual_diagonal'].tolist(), 'discrepancy_variance': state['discrepancy'],
            'common_group_variances': state['common_group_variances'].tolist(), 'common_spectral_exponent': state['common_spectral_exponent'],
            'source_group_variances': state['group_variances'].tolist(), 'spectral_exponent': state['spectral_exponent'],
            'new_task_covariance_includes_population_mean_uncertainty': True, 'new_optimizer_invocations': 0,
            'source_negative_log_likelihood_per_label': value, 'prefix_hyperparameter_fit': original}
        fitted['heads'].append({'joint_structural': info})
    return fitted, contraction


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution']); coverage = read(spec['coverage_execution'])
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); groups, fits, reductions = {}, {}, []
    for layout in spec['layouts']:
        parent = spec['parent_layouts'][layout]; folder = Path(source['output'])/f'layout{parent}'
        inputs, means, risk = read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json')
        augmented = read(Path(coverage['output'])/f'layout{layout}'/'inputs.json')
        fit, reduction = condition_means(inputs, means, augmented)
        local = out/f'layout{layout}'; local.mkdir(parents=True, exist_ok=True); save(local/'source_fit.json', fit)
        fits[layout] = fit; reductions.extend(dict(r, layout=layout) for r in reduction)
        groups[layout] = [(inputs, means, risk)]+[(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
            read(folder/f'task{i}'/'risk_source_fit.json')) for i in spec['training_tasks']]
    data = read(spec['source_data']); public = read(Path(spec['source_data']).with_name('frozen_inputs.json'))
    rows, jobs = [], {}
    for layout, group in groups.items():
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
                'tau': task['tau'], 'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            for arm, model in risk_models(group, fits[layout], group[0][0]['heldout_mean_psi'][str(i)],
                    [g[0]['heldout_risk_psi'][str(i)] for g in group], spec['mean_arms']):
                prior = forecast(query, arm, 'frozen144_covariance_conditioned216_mean', model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, arm, 'frozen144_covariance_conditioned216_mean', model)
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
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'covariance_contractions': reductions,
        'source_gate_passed': all(c['passed'] for c in contrasts),
        'cost': {'source_calls_per_model': 216, 'new_source_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result); print(result, flush=True)


if __name__ == '__main__':
    run()
