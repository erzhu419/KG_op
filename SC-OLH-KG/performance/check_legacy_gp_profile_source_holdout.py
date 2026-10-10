"""Frozen native source-NLL start selection on the existing source holdouts."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_counterbalanced_complete_mean_uncertainty import load_group, contrast
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_gp_profile_start import initial_profile
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.run_legacy_related_inventory_source import save, training_view
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood, _solve_joint_common_population

EXECUTION = 'performance/manifests/legacy_joint_population_gp_profile_source_holdout_execution_v1.json'
ARM = 'joint_structural'


def refit(inputs, means, folder):
    folder.mkdir(parents=True, exist_ok=True)
    tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}}
        for t in inputs['source_tasks']]
    basis = OrthogonalLowOrderBasis(inputs['public_mean_reference']); psi = np.asarray(inputs['public_mean_reference'])
    design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
    native = means['heads'][1][ARM]
    prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
    likelihood = JointCovariancePopulationLikelihood(prior, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
    np.testing.assert_allclose(likelihood.evaluate(np.asarray(native['optimizer_parameters']))[0],
        native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    initial, profile = initial_profile(design, psi, basis, means['heads'][1]['joint_gp'], native, likelihood)
    optimizer = inputs['execution']['optimizer']; count = 1
    candidate = _solve_joint_common_population(likelihood, initial,
        native['fixed_common_source_negative_log_likelihood_per_label'], optimizer).diagnostics
    if not candidate['optimizer_converged']:
        save(folder/'before_completion.json', candidate)
        candidate = _solve_joint_common_population(likelihood, np.asarray(candidate['optimizer_parameters']),
            native['fixed_common_source_negative_log_likelihood_per_label'], optimizer).diagnostics
        count += 1
    save(folder/'candidate_fit.json', candidate)
    if not candidate['optimizer_converged']:
        raise RuntimeError('full-source native g fit incomplete; no holdout forecasts')
    improvement = native['source_negative_log_likelihood_per_label']-candidate['source_negative_log_likelihood_per_label']
    selected = improvement > optimizer['ftol']; fitted = deepcopy(means)
    if selected:
        fitted['heads'][1][ARM] = candidate
    save(folder/'selected_source_fit.json', fitted)
    info = {'initial_profile': profile, 'native': native, 'candidate': candidate,
        'source_nll_improvement_per_label': improvement, 'selected_gp_start': bool(selected), 'new_g_source_fits': count}
    save(folder/'fit_attribution.json', info)
    return fitted, info


def reproduce(actual, cached):
    for key in ('mean', 'variance', 'expected_episode_variance', 'identity_weights'):
        np.testing.assert_allclose(actual[key], cached[key], atol=1e-8, rtol=1e-8)
    assert actual['identity_names'] == cached['identity_names']
    np.testing.assert_allclose(actual['paid_predictive_log_score'], cached['paid_predictive_log_score'], atol=1e-8)


def run():
    started = time.process_time(); spec = read(EXECUTION); coverage = read(spec['coverage_execution'])
    parent = read(coverage['source_execution']); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    groups, fits, attribution = {}, {}, []
    for layout in spec['layouts']:
        group = load_group(spec, coverage, parent, layout)
        fitted, info = refit(group[0][0], group[0][1], out/f'layout{layout}')
        groups[layout], fits[layout] = group, fitted; attribution.append(dict(info, layout=layout))
        print('full-source g fit frozen', layout, info['selected_gp_start'], info['source_nll_improvement_per_label'], flush=True)
    # Fit/start selection is complete before indexing either holdout's paid observations.
    references = read(spec['reference_profiles'])['rows']
    data = read(coverage['source_data']); public = read(Path(coverage['source_data']).with_name('frozen_inputs.json'))
    rows, models, controls = [], {}, []
    for layout, group in groups.items():
        for i in spec['heldout_tasks']:
            task = data['tasks'][i]
            query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': spec['paid_candidates'],
                'paired_observations': [task['Y_replicates'][j][0] for j in spec['paid_candidates']]}
            mq = np.asarray(group[0][0]['heldout_mean_psi'][str(i)])
            rq = [np.asarray(g[0]['heldout_risk_psi'][str(i)]) for g in group]
            baseline = risk_models(group, group[0][1], mq, rq, [ARM])[0][1]
            model = risk_models(group, fits[layout], mq, rq, [ARM])[0][1]
            old = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
            prior = forecast(query, ARM, 'native_source_nll_start_selection', model)
            original_prior = forecast(query, ARM, 'native_source_control', baseline)
            reproduce(original_prior, old['prior'])
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                centered = [value[0], value[1]-query['tau']]
                model.update_raw(j, centered); baseline.update_raw(j, centered)
            original = forecast(query, ARM, 'native_source_control', baseline); reproduce(original, old)
            row = forecast(query, ARM, 'native_source_nll_start_selection', model)
            row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior)
            rows.append(row); models[(layout, i)] = model
            controls.append(dict(original, layout=layout, prior=original_prior))
    save(out/'source_predictions.json', {'execution': spec, 'all4_candidate_forecasts_frozen': True,
        'native_controls_all73_prior_paid_moments_reproduced': True, 'rows': rows, 'controls': controls})
    # Unobserved labels are scored only after all four candidate forecasts are on disk.
    labels = {i: training_view(data, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    parents = read(spec['parent_reference_profiles'])['rows']; contrasts, parent_contrasts, gp_contrasts = [], [], []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], models[(layout, i)], 3)
        old = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
        original = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == ARM)
        gp = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_gp')
        contrasts.append(contrast(row, old, 'same216_call_native_structural'))
        parent_contrasts.append(contrast(row, original, 'original144_call_structural'))
        gp_contrasts.append(contrast(row, gp, 'same216_call_gp'))
    metrics = ('observed_head_mse', 'joint_batch_log_score', 'response_batch_log_scores',
        'predictive_moment_band_95_observed_coverage', 'predictive_standardized_residual_rms')
    summary = {layout: {case: {key: np.mean([r['scores'][key] for r in members if r['layout'] == layout], axis=0).tolist()
        for key in metrics} for case, members in (
            ('candidate', rows), ('same216_native', [r for r in references if r['mean_arm'] == ARM]),
            ('same216_gp', [r for r in references if r['mean_arm'] == 'joint_gp']))} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'gp_contrasts': gp_contrasts, 'fit_attribution': attribution,
        'same_information_source_gate_passed': all(c['passed'] for c in contrasts),
        'original144_source_gate_passed': all(c['passed'] for c in parent_contrasts),
        'source_gate_passed': all(c['passed'] for c in contrasts+parent_contrasts),
        'cost': {**spec['cost'], 'actual_new_g_source_fits': sum(d['new_g_source_fits'] for d in attribution)},
        'cpu_seconds': time.process_time()-started, 'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'summary': summary, 'contrasts': contrasts,
        'parent_contrasts': parent_contrasts, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
