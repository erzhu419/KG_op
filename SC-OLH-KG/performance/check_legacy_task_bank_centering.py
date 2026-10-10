"""Source-only task-level/shape contrast in the unchanged orthogonal basis."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.calibrate_legacy_task_transfer import NATIVE
from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.check_legacy_task_spectral_moments import score
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood
from representation.task_bank_centering import (CenteredTaskPopulationLikelihood,
    centered_g_field, fit_centered_task_population, task_coefficient_covariance)

EXECUTION = 'performance/manifests/legacy_joint_population_task_bank_centering_execution_v1.json'
CENTERED = 'public_bank_centered_task_population'


def verify_dense(likelihood, parameters):
    """Independent dense marginal/posterior/gradient for actual source blocks."""
    value, gradient, state = likelihood.evaluate(parameters)
    count = len(likelihood.basis.group_names); group_index = likelihood.basis.group_index
    common_weights, common_derivative = spectral_weights(likelihood.basis, parameters[count])
    task_weights, task_derivative = spectral_weights(likelihood.basis, parameters[-2])
    common = np.diag(state['common_diagonal']); residual = state['residual_diagonal']
    design = np.vstack([b[0] for b in likelihood.blocks]); error = np.concatenate([b[1] for b in likelihood.blocks])
    blocks = [(z*residual)@z.T+np.diag(b[2]+state['discrepancy'])
        for b, z in zip(likelihood.blocks, likelihood.task_designs)]
    covariance = design@common@design.T+block_diag(*blocks)
    factor = cho_factor(covariance, lower=True); alpha = cho_solve(factor, error)
    n = likelihood.label_count
    dense = .5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha)/n
    expected_mean = likelihood.prior.mean+common@design.T@alpha
    expected_cov = common-common@design.T@cho_solve(factor, design)@common
    np.testing.assert_allclose(value, dense, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean'], expected_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(state['mean_covariance'], expected_cov, atol=1e-8, rtol=1e-8)
    influence = cho_solve(factor, np.eye(n))-np.outer(alpha, alpha)
    derivatives = []
    for k in range(count):
        diagonal = np.r_[0., likelihood.signal*common_weights*(group_index == k)]
        derivatives.append((design*diagonal)@design.T)
    diagonal = np.r_[0., state['common_group_variances'][group_index]*common_derivative]
    derivatives.append((design*diagonal)@design.T)
    derivatives.append(block_diag(*[likelihood.signal*np.outer(z[:, 0], z[:, 0]) for z in likelihood.task_designs]))
    for k in range(count):
        diagonal = np.r_[0., likelihood.signal*task_weights*(group_index == k)]
        derivatives.append(block_diag(*[(z*diagonal)@z.T for z in likelihood.task_designs]))
    diagonal = np.r_[0., state['group_variances'][group_index]*task_derivative]
    derivatives.append(block_diag(*[(z*diagonal)@z.T for z in likelihood.task_designs]))
    derivatives.append(likelihood.signal*np.eye(n))
    expected_gradient = np.array([.5*np.sum(influence*d.T)/n for d in derivatives])
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-8, rtol=1e-7)
    return {'nll_absolute_error': float(abs(value-dense)),
        'gradient_max_absolute_error': float(np.max(np.abs(gradient-expected_gradient))),
        'common_covariance_max_absolute_error': float(np.max(np.abs(state['mean_covariance']-expected_cov)))}


def prepare(spec):
    profile = read(spec['profile_execution']); coverage = read(spec['coverage_execution'])
    projection = read(profile['projection_execution']); gain = read(projection['gain_execution'])
    adaptation = read(gain['adaptation_execution']); parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    public_coordinates = {public['tasks'][i]['task_id']: public_control_mean_coordinates(
        no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank) for i in spec['training_tasks']}
    branches = {}
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            inputs = read(Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'/'inputs.json')
            means = read(Path(profile['output'])/f'layout{layout}'/f'task{deleted}'/'selected_source_fit.json')
            native = means['heads'][1]['joint_structural']; basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            mean_psi = {name: (raw-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
                for name, raw in public_coordinates.items()}
            designs = {name: np.column_stack([np.ones(len(bank)), basis.transform(psi)]) for name, psi in mean_psi.items()}
            centers = {name: d[:, 1:].mean(axis=0) for name, d in designs.items()}
            prior = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            batches, noises = head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks]
            likelihood = CenteredTaskPopulationLikelihood(prior, batches, noises, [centers[t['name']] for t in tasks])
            for t, (d, _, _), z in zip(tasks, likelihood.blocks, likelihood.task_designs):
                residual = np.asarray(native['residual_coefficient_diagonal']); center = centers[t['name']]
                np.testing.assert_allclose((z*residual)@z.T, d@task_coefficient_covariance(residual, center)@d.T, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose((designs[t['name']][:, 1:]-center).mean(axis=0), 0., atol=1e-12)
            branches[(layout, deleted)] = {'inputs': inputs, 'means': means, 'native': native,
                'likelihood': likelihood, 'tasks': tasks, 'basis': basis, 'centers': centers, 'designs': designs,
                'risk_path': Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json',
                'task_name': public['tasks'][deleted]['task_id'], 'mean_psi': mean_psi[public['tasks'][deleted]['task_id']],
                'mean_psi_by_task': mean_psi,
                'sites': coverage['layouts'][layout][deleted],
                'layout': layout, 'deleted_task': deleted, 'paid_sites': spec['paid_candidates']}
    return coverage, branches


def run():
    started = time.process_time(); spec = read(EXECUTION); out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    coverage, branches = prepare(spec); verification = []
    for query in spec['verification_branches']:
        b = branches[(query['layout'], query['deleted_task'])]; likelihood = b['likelihood']; native = b['native']
        parameters = np.asarray(native['optimizer_parameters'])
        dense = verify_dense(likelihood, parameters)
        zero = CenteredTaskPopulationLikelihood(likelihood.prior, head_batches(b['tasks'], 1),
            [t['noise'][:, 1] for t in b['tasks']], [np.zeros(b['basis'].feature_dim) for _ in b['tasks']])
        original = JointCovariancePopulationLikelihood(likelihood.prior, head_batches(b['tasks'], 1), [t['noise'][:, 1] for t in b['tasks']])
        a, ga, sa = zero.evaluate(parameters); expected, ge, se = original.evaluate(parameters)
        np.testing.assert_allclose(a, expected, atol=1e-9, rtol=1e-9)
        np.testing.assert_allclose(expected, native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(ga, ge, atol=1e-8, rtol=1e-8)
        for key in se:
            np.testing.assert_allclose(sa[key], se[key], atol=1e-9, rtol=1e-9)
        verification.append({**query, **dense, 'zero_center_native_law_reproduced': True})
    save(out/'source_verification.json', {'execution': spec, 'rows': verification, 'source_covariance_transform_equivalence_passed': True})
    print('centered source likelihood/posterior/gradient, zero-center native law and covariance transform verified', flush=True)
    laws = []
    for (layout, deleted), b in branches.items():
        folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
        optimizer = b['inputs']['execution']['optimizer']; assert optimizer == spec['optimizer']
        fit_started = time.process_time(); law = fit_centered_task_population(b['likelihood'], b['native'], optimizer)
        b['law'] = law; law['fit_cpu_seconds'] = time.process_time()-fit_started
        law['source_bank_centers'] = {t['name']: b['centers'][t['name']].tolist() for t in b['tasks']}
        law['source_task_names'] = [t['name'] for t in b['tasks']]
        laws.append({'layout': layout, 'deleted_task': deleted, 'law': law})
        save(folder/'source_fit.json', law)
        print('centered g source fit', layout, deleted, law['optimizer_converged'], law['optimizer_iterations'],
            law['source_negative_log_likelihood_per_label'], flush=True)
    save(out/'source_laws.json', {'execution': spec, 'all12_laws_frozen_before_forecasts': True, 'rows': laws})
    cost = {**spec['cost'], 'actual_new_g_source_fits': len(laws)}
    if not all(r['law']['optimizer_converged'] for r in laws):
        result = {'execution': spec, 'all_fits_complete': False, 'forecasts_scored': False,
            'centering_preferred': False, 'verification': verification, 'cost': cost,
            'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
            'target_promoted': False, 'independent_confirmation': False}
        save(out/'results.json', result); print('incomplete fit retained; no scoring or continuation', flush=True); return
    native_rows = [r for r in read(spec['native_profiles'])['rows'] if r['case'] == NATIVE]
    raw = read(coverage['source_data'])
    paid_values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in spec['paid_candidates']]
        for i in spec['training_tasks']}; del raw
    rows, fields = [], {}
    for (layout, deleted), b in branches.items():
        template = components(b['inputs'], b['means'], read(b['risk_path']), b['mean_psi'],
            np.asarray(b['inputs']['public_risk_reference']))[0][1]
        law, design, center = b['law'], b['designs'][b['task_name']], b['centers'][b['task_name']]
        diagonal = np.asarray(law['residual_coefficient_diagonal']); tail = .01*law['source_signal_scale']/b['basis'].feature_dim
        assert np.all(diagonal[1:] >= tail)
        coefficient_covariance = np.asarray(law['population_mean_covariance'])+task_coefficient_covariance(diagonal, center)
        np.testing.assert_allclose(coefficient_covariance, coefficient_covariance.T, atol=1e-10, rtol=1e-10)
        assert np.linalg.eigvalsh(coefficient_covariance).min() >= -1e-10
        prior = centered_g_field(template, design, law, center); paid = deepcopy(prior); evidence = 0.
        old_mean, old_cov = template.predict(full_cov=True); new_mean, new_cov = prior.predict(full_cov=True)
        np.testing.assert_array_equal(new_mean[:, 1], old_mean[:, 1]); np.testing.assert_array_equal(new_cov[:, 1, :, 1], old_cov[:, 1, :, 1])
        for j, value in zip(b['paid_sites'], paid_values[deleted]):
            evidence += paid.update_batch(j, float(value), None, 1)
        prediction = {}
        for stage, field in zip(('prior', 'raw_paid'), (prior, paid)):
            mean, covariance = field.predict(full_cov=True)
            prediction[stage] = g_prediction(mean[:, 0], covariance[:, 0, :, 0], b['sites'])
        prediction['raw_paid']['paid_predictive_log_score'] = float(evidence)
        rows.append({'layout': layout, 'deleted_task': deleted, 'case': CENTERED,
            'scoring_sites': b['sites'], 'predictions': prediction, 'query_bank_center': center.tolist()})
        fields[(layout, deleted)] = (prior, paid)
    save(out/'predictions.json', {'execution': spec, 'all12_candidate_and_native_forecasts_frozen_before_scoring': True,
        'eta_laws_unchanged': True, 'native_frozen_profile_reference': spec['native_profiles'], 'rows': rows+native_rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        key = (row['layout'], row['deleted_task']); b = branches[key]
        labels = source[key[0]]['source_tasks'][key[1]]
        b['observed'], b['noise'] = np.asarray(labels['response']), np.asarray(labels['noise'])[:, 1]
        row['scores'] = score(row, b, fields[key])
    summary = {layout: {case: {stage: {metric: float(np.mean([r['scores'][stage][metric] for r in members if r['layout'] == layout]))
        for metric in members[0]['scores'][stage]} for stage in ('prior', 'raw_paid')}
        for case, members in ((NATIVE, native_rows), (CENTERED, rows))} for layout in spec['layouts']}
    preferred = {}; contrasts = []
    for layout in spec['layouts']:
        new, old = (summary[layout][case]['raw_paid'] for case in (CENTERED, NATIVE))
        preferred[layout] = new['g_mse'] <= old['g_mse'] and new['g_raw_batch_density'] > old['g_raw_batch_density'] and new['sequential_transcript_density'] >= old['sequential_transcript_density']
    for row in rows:
        old = next(r for r in native_rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'])
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'centered_minus_native': {stage: {metric: row['scores'][stage][metric]-old['scores'][stage][metric] for metric in row['scores'][stage]}
                for stage in ('prior', 'raw_paid')}})
    previous = read('paper_artifacts/legacy_joint_population_task_spectral_moments_v1/results.json')
    gp_spec = read('performance/manifests/legacy_joint_population_training_paid_adaptation_execution_v1.json')
    gp_summary = read(Path(gp_spec['output'])/'results.json')['summary']
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'benchmarks': {layout: {**previous['benchmarks'][layout],
            'stage85_integrated': previous['summary'][layout]['conditional_task_spectral_moments']['raw_paid'],
            'same_source_gp': gp_summary[layout]['joint_gp']}
            for layout in spec['layouts']}, 'preferred_by_layout': preferred, 'centering_preferred': all(preferred.values()),
        'all_fits_complete': True, 'forecasts_scored': True, 'verification': verification, 'cost': cost,
        'cpu_seconds': time.process_time()-started, 'source_holdouts_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows+native_rows}); save(out/'results.json', result)
    print({'centering_preferred': result['centering_preferred'], 'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
