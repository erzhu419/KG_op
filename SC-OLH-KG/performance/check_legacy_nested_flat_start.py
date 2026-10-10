"""Source-only native likelihood initialization check with frozen risk controls."""

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
from representation.source_joint_population import JointCovariancePopulationLikelihood, _solve_joint_common_population

EXECUTION = 'performance/manifests/legacy_joint_population_nested_flat_start_v1.json'


def flat_initialization(basis, flat):
    common, intercept, residual, discrepancy = flat['optimizer_parameters']
    fractions = np.bincount(basis.group_index)/basis.feature_dim
    return np.r_[common*fractions, 0., intercept, residual*fractions, 0., discrepancy]


def fit_layout(inputs, old, out):
    out.mkdir(parents=True, exist_ok=True); fitted = copy.deepcopy(old); diagnostics = []; started = time.process_time()
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); optimizer = inputs['execution']['optimizer']
    for h, head in enumerate(fitted['heads']):
        prior = restore_regularizer(basis, head['regularizer'], tasks, h)
        likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, h), [t['noise'][:, h] for t in tasks])
        flat = head['flat']; initial = flat_initialization(basis, flat)
        value, _, state = likelihood.evaluate(initial)
        # An incorrect embedding changes the covariance model before optimization.
        np.testing.assert_allclose(value, flat['source_negative_log_likelihood_per_label'], rtol=1e-8, atol=1e-8)
        for native, cached in (('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
                ('common_diagonal', 'common_prior_coefficient_diagonal'), ('residual_diagonal', 'residual_coefficient_diagonal')):
            np.testing.assert_allclose(state[native], flat[cached], rtol=1e-8, atol=1e-8)
        old_fit = head['joint_structural']
        np.testing.assert_allclose(likelihood.evaluate(np.asarray(old_fit['optimizer_parameters']))[0],
            old_fit['source_negative_log_likelihood_per_label'], rtol=1e-8, atol=1e-8)
        candidate = _solve_joint_common_population(likelihood, initial,
            old_fit['fixed_common_source_negative_log_likelihood_per_label'], optimizer).diagnostics
        if not candidate['optimizer_converged']:
            save(out/f'head{h}_before_completion.json', candidate)
            candidate = _solve_joint_common_population(likelihood, np.asarray(candidate['optimizer_parameters']),
                old_fit['fixed_common_source_negative_log_likelihood_per_label'], optimizer).diagnostics
        if not candidate['optimizer_converged']:
            save(out/f'head{h}_failed.json', candidate)
            raise RuntimeError('native source fit incomplete; no forecasts')
        improvement = old_fit['source_negative_log_likelihood_per_label']-candidate['source_negative_log_likelihood_per_label']
        selected = improvement > optimizer['ftol']
        if selected:
            head['joint_structural'] = candidate
        diagnostics.append({'head': h, 'original': old_fit, 'flat_initialized_fit': candidate,
            'source_nll_improvement_per_label': improvement, 'selected_alternative': selected})
    save(out/'source_fit.json', fitted); save(out/'fit_attribution.json', {'heads': diagnostics, 'cpu_seconds': time.process_time()-started})
    return fitted, diagnostics


def run():
    started = time.process_time(); spec = read(EXECUTION); source = read(spec['source_execution'])
    data = read(source['source_data']); public = read(Path(source['source_data']).with_name('frozen_inputs.json'))
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); groups, fits, attribution = {}, {}, []
    for layout in spec['layouts']:
        folder = Path(source['output'])/f'layout{layout}'
        group = [(read(folder/'inputs.json'), read(folder/'source_fit.json'), read(folder/'risk_source_fit.json'))]
        group += [(read(folder/f'task{i}'/'inputs.json'), read(folder/f'task{i}'/'gp_source_fit.json'),
                   read(folder/f'task{i}'/'risk_source_fit.json')) for i in source['training_tasks']]
        fit, diagnostics = fit_layout(group[0][0], group[0][1], out/f'layout{layout}')
        groups[layout], fits[layout] = group, fit
        attribution.extend(dict(d, layout=layout) for d in diagnostics)
    rows, jobs = [], {}
    for layout, group in groups.items():
        for i in source['heldout_tasks']:
            task = data['tasks'][i]; query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source',
                'tau': task['tau'], 'candidate_indices': [0, 1], 'paired_observations': [task['Y_replicates'][j][0] for j in (0, 1)]}
            for arm, model in risk_models(group, fits[layout], group[0][0]['heldout_mean_psi'][str(i)],
                    [g[0]['heldout_risk_psi'][str(i)] for g in group], spec['mean_arms']):
                prior = forecast(query, arm, 'nested_flat_start_source_nll_selection', model)
                for j, value in zip(query['candidate_indices'], query['paired_observations']):
                    model.update_raw(j, [value[0], value[1]-query['tau']])
                row = forecast(query, arm, 'nested_flat_start_source_nll_selection', model)
                row.update(layout=layout, prior=prior); rows.append(row); jobs[(layout, i, arm)] = model
    # All selected fits and both layouts' predictions are saved before any score is evaluated.
    save(out/'source_predictions.json', {'execution': spec, 'rows': rows})
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in source['heldout_tasks']}
    reference = read(spec['reference_profiles'])['rows']; contrasts = []
    for row in rows:
        layout, i, arm = row['layout'], row['task_index'], row['mean_arm']
        score_source(row, labels[i], jobs[(layout, i, arm)], 3)
        old = next(r for r in reference if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == arm)
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
        'source_gate_passed': all(c['passed'] for c in contrasts), 'fit_attribution': attribution,
        'cost': {'new_mean_fits': 4+sum((out/f'layout{l}'/f'head{h}_before_completion.json').exists() for l in spec['layouts'] for h in (0, 1)),
            'new_risk_fits': 0, 'new_simulator_calls': 0, 'new_verification_calls': 0, 'new_terminal_integrations': 0},
        'cpu_seconds': time.process_time()-started}
    save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'summary': summary, 'contrasts': contrasts,
        'nll_improvements': [(d['layout'], d['head'], d['source_nll_improvement_per_label'], d['selected_alternative']) for d in attribution],
        'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
