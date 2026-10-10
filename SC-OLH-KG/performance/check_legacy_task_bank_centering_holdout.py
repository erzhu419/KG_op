"""Frozen public-bank task level/shape family on existing source6/7 checks."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_counterbalanced_complete_mean_uncertainty import load_group, contrast
from performance.check_legacy_crossfit_variance_noise import risk_models
from performance.check_legacy_gp_profile_source_holdout import reproduce
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read, forecast, score_source
from performance.check_legacy_task_transfer_holdout import calibrated_model
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save, training_view
from representation.joint_response_mixture import JointResponseMixture
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood
from representation.task_bank_centering import (CenteredTaskPopulationLikelihood,
    centered_g_field, fit_centered_task_population, task_coefficient_covariance)

EXECUTION = 'performance/manifests/legacy_joint_population_task_bank_centering_holdout_execution_v1.json'
ARM = 'joint_structural'


def centered_model(template, design, law, center):
    components = [(deepcopy(f), centered_g_field(g, design, law, center)) for f, g in template.components]
    for (f, g), (original_f, original_g) in zip(components, template.components):
        for actual, expected in zip(f.predict(full_cov=True), original_f.predict(full_cov=True)):
            np.testing.assert_array_equal(actual, expected)
        mean, covariance = g.predict(full_cov=True); expected_mean, expected_cov = original_g.predict(full_cov=True)
        np.testing.assert_array_equal(mean[:, 1], expected_mean[:, 1])
        np.testing.assert_array_equal(covariance[:, 1, :, 1], expected_cov[:, 1, :, 1])
    return JointResponseMixture(template.names, components)


def run():
    started = time.process_time(); spec = read(EXECUTION)
    training = read(spec['training_results']); assert training['centering_preferred'] and training['all_fits_complete']
    training_spec = read(spec['training_execution']); native_spec = read(spec['native_execution'])
    coverage = read(spec['coverage_execution']); parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw_coordinates = {public['tasks'][i]['task_id']: public_control_mean_coordinates(
        no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank) for i in spec['training_tasks']}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); groups, means, laws = {}, {}, {}
    for layout in spec['layouts']:
        group = load_group(spec, coverage, parent, layout); inputs = group[0][0]
        native_fit = read(Path(native_spec['output'])/f'layout{layout}'/'selected_source_fit.json')
        native = native_fit['heads'][1][ARM]; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
        tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
        centers = {name: basis.transform((coordinates-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']).mean(axis=0)
            for name, coordinates in raw_coordinates.items()}
        prior = restore_regularizer(basis, native_fit['heads'][1]['regularizer'], tasks, 1)
        batches, noise = head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks]
        likelihood = CenteredTaskPopulationLikelihood(prior, batches, noise, [centers[t['name']] for t in tasks])
        original = JointCovariancePopulationLikelihood(prior, batches, noise)
        np.testing.assert_allclose(original.evaluate(np.asarray(native['optimizer_parameters']))[0],
            native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        assert inputs['execution']['optimizer'] == training_spec['optimizer']
        law = fit_centered_task_population(likelihood, native, training_spec['optimizer'])
        law['source_bank_centers'] = {name: value.tolist() for name, value in centers.items()}
        folder = out/f'layout{layout}'; folder.mkdir(parents=True, exist_ok=True); save(folder/'source_fit.json', law)
        groups[layout], means[layout], laws[layout] = group, native_fit, law
        print('full-six-source centered g fit frozen', layout, law['optimizer_converged'], law['optimizer_iterations'], flush=True)
    save(out/'source_laws.json', {'execution': spec, 'both_laws_frozen_before_heldout_paid_observations': True,
        'rows': [{'layout': layout, 'law': law} for layout, law in laws.items()]})
    cost = {**spec['cost'], 'actual_new_g_source_fits': len(laws)}
    if not all(law['optimizer_converged'] for law in laws.values()):
        result = {'execution': spec, 'source_gate_passed': False, 'all_fits_complete': False, 'forecasts_scored': False,
            'cost': cost, 'cpu_seconds': time.process_time()-started, 'independent_confirmation': False, 'target_promoted': False}
        save(out/'results.json', result); print('incomplete full-source fit; no scoring or continuation', flush=True); return
    cached = read(Path(native_spec['output'])/'calibration_profiles.json')['rows']
    raw = read(coverage['source_data']); rows, jobs = [], {}
    for layout, group in groups.items():
        basis = OrthogonalLowOrderBasis(group[0][0]['public_mean_reference'])
        for i in spec['heldout_tasks']:
            task = raw['tasks'][i]
            query = {'task_id': task['task_id'], 'task_index': i, 'kind': 'source', 'tau': task['tau'],
                'candidate_indices': spec['paid_candidates'], 'paired_observations': [task['Y_replicates'][j][0] for j in spec['paid_candidates']]}
            psi = np.asarray(group[0][0]['heldout_mean_psi'][str(i)])
            design = np.column_stack([np.ones(len(psi)), basis.transform(psi)]); center = design[:, 1:].mean(axis=0)
            risk_queries = [np.asarray(g[0]['heldout_risk_psi'][str(i)]) for g in group]
            template = risk_models(group, means[layout], psi, risk_queries, [ARM])[0][1]
            diagonal = np.asarray(laws[layout]['residual_coefficient_diagonal']); assert np.all(diagonal[1:] >= .01*laws[layout]['source_signal_scale']/basis.feature_dim)
            coefficient_covariance = np.asarray(laws[layout]['population_mean_covariance'])+task_coefficient_covariance(diagonal, center)
            np.testing.assert_allclose(coefficient_covariance, coefficient_covariance.T, atol=1e-10, rtol=1e-10)
            assert np.linalg.eigvalsh(coefficient_covariance).min() >= -1e-10
            original = calibrated_model(template, design, means[layout]['heads'][1][ARM], [1., 1.])
            model = centered_model(template, design, laws[layout], center)
            old = next(r for r in cached if r['layout'] == layout and r['task_index'] == i)
            reproduce(forecast(query, ARM, 'native_control', original), old['prior'])
            prior = forecast(query, ARM, 'bank_centered_task_population', model)
            for j, value in zip(query['candidate_indices'], query['paired_observations']):
                centered = [value[0], value[1]-query['tau']]
                original.update_raw(j, centered); model.update_raw(j, centered)
            reproduce(forecast(query, ARM, 'native_control', original), old)
            row = forecast(query, ARM, 'bank_centered_task_population', model)
            row.update(layout=layout, parent_layout=spec['parent_layouts'][layout], prior=prior, query_bank_center=center.tolist())
            rows.append(row); jobs[(layout, i)] = model
            print('centered task prior/paid source forecast frozen', layout, i, flush=True)
    save(out/'source_predictions.json', {'execution': spec, 'all4_forecasts_frozen_before_scoring': True,
        'native_all73_four_head_prior_paid_reproduced': True, 'f_eta_prior_laws_unchanged': True, 'rows': rows})
    labels = {i: training_view(raw, [i], public['plan']['task_distribution']['L'])[0] for i in spec['heldout_tasks']}
    references = read(spec['reference_profiles'])['rows']; parents = read(spec['parent_reference_profiles'])['rows']
    contrasts, parent_contrasts, selected_contrasts, gp_contrasts = [], [], [], []
    for row in rows:
        layout, i = row['layout'], row['task_index']; score_source(row, labels[i], jobs[(layout, i)], 3)
        same = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == ARM)
        old = next(r for r in parents if r['layout'] == row['parent_layout'] and r['task_index'] == i and r['mean_arm'] == ARM)
        selected = next(r for r in cached if r['layout'] == layout and r['task_index'] == i)
        gp = next(r for r in references if r['layout'] == layout and r['task_index'] == i and r['mean_arm'] == 'joint_gp')
        contrasts.append(contrast(row, same, 'same216_call_native_structural'))
        parent_contrasts.append(contrast(row, old, 'original144_call_structural'))
        selected_contrasts.append(contrast(row, selected, 'same216_call_selected_native'))
        gp_contrasts.append(contrast(row, gp, 'same216_call_gp'))
    metrics = ('observed_head_mse', 'joint_batch_log_score', 'response_batch_log_scores')
    summary = {layout: {case: {key: np.mean([r['scores'][key] for r in members if r['layout'] == layout], axis=0).tolist()
        for key in metrics} for case, members in (('centered', rows), ('native', cached),
        ('gp', [r for r in references if r['mean_arm'] == 'joint_gp']))} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'parent_contrasts': parent_contrasts,
        'selected_contrasts': selected_contrasts, 'gp_contrasts': gp_contrasts,
        'same_information_source_gate_passed': all(c['passed'] for c in contrasts),
        'original144_source_gate_passed': all(c['passed'] for c in parent_contrasts),
        'source_gate_passed': all(c['passed'] for c in contrasts+parent_contrasts),
        'all_fits_complete': True, 'forecasts_scored': True, 'native_all73_four_head_prior_paid_reproduced': True,
        'cost': cost, 'cpu_seconds': time.process_time()-started, 'independent_confirmation': False, 'target_promoted': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'source_gate_passed': result['source_gate_passed'], 'summary': summary, 'contrasts': contrasts,
        'parent_contrasts': parent_contrasts, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
