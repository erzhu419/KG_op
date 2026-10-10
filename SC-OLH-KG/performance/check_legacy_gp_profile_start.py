"""Refit the unchanged native source objective from a GP-derived variance profile."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_joint_common_covariance import restore_regularizer
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_function_prior import head_batches
from representation.source_joint_population import JointCovariancePopulationLikelihood, _solve_joint_common_population
from representation.source_kernel_population import rbf_cross, shared_kernel

EXECUTION = 'performance/manifests/legacy_joint_population_gp_profile_start_execution_v1.json'


def initial_profile(design, psi, basis, fit, native, likelihood):
    operator = np.linalg.lstsq(design, np.eye(len(design)), rcond=None)[0]
    scale = fit['anchor']['normalization']['scale']**2
    common = scale*shared_kernel(psi, psi, fit['anchor'])
    amplitude, intercept, discrepancy = np.asarray(fit['residual_parameters'])*scale
    task = amplitude*rbf_cross(psi, psi, fit['anchor']['lengthscale'])+intercept
    common_diagonal = np.diag(operator@common@operator.T)
    task_diagonal = np.diag(operator@task@operator.T)
    sizes = np.bincount(basis.group_index)
    common_mass = np.bincount(basis.group_index, weights=common_diagonal[1:])
    task_mass = np.bincount(basis.group_index, weights=task_diagonal[1:])
    common_groups = np.maximum(common_mass-sizes*likelihood.tail, 0.)
    task_groups = np.maximum(task_mass-sizes*likelihood.tail, 0.)
    initial = np.r_[common_groups/likelihood.signal, native['common_spectral_exponent'],
        task_diagonal[0]/likelihood.signal, task_groups/likelihood.signal,
        native['spectral_exponent'], discrepancy/likelihood.signal]
    return initial, {'projected_common_prior_group_mass': common_mass.tolist(),
        'projected_task_group_mass': task_mass.tolist(),
        'projected_task_intercept_variance': float(task_diagonal[0]),
        'common_intercept_variance_retained': float(likelihood.prior.covariance[0, 0])}


def run():
    started = time.process_time(); spec = read(EXECUTION); projection = read(spec['projection_execution'])
    gain = read(projection['gain_execution']); adaptation = read(gain['adaptation_execution'])
    coverage = read(adaptation['coverage_execution']); deletion = read(adaptation['deleted_mean_execution'])
    parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    references = read(Path(read(gain['adaptation_results'])['execution']['output'])/'predictions.json')['rows']
    raw = read(coverage['source_data']); paid_sites = spec['paid_candidates']
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in paid_sites]
        for i in spec['training_tasks']}; del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    rows, fits, models, fit_count = [], [], {}, 0
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs, gp = read(branch/'inputs.json'), read(branch/'gp_source_fit.json')
            means = read(Path(deletion['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            means = dict(means, heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(means['heads'], gp['heads'])])
            native = means['heads'][1]['joint_structural']
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            tasks = [{**t, **{k: np.asarray(t[k]) for k in ('psi', 'features', 'response', 'noise')}} for t in inputs['source_tasks']]
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            regularizer = restore_regularizer(basis, means['heads'][1]['regularizer'], tasks, 1)
            likelihood = JointCovariancePopulationLikelihood(regularizer, head_batches(tasks, 1), [t['noise'][:, 1] for t in tasks])
            np.testing.assert_allclose(likelihood.evaluate(np.asarray(native['optimizer_parameters']))[0],
                native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
            initial, profile = initial_profile(design, psi, basis, gp['heads'][1]['joint_gp'], native, likelihood)
            optimizer = inputs['execution']['optimizer']; folder = out/f'layout{layout}'/f'task{deleted}'; folder.mkdir(parents=True, exist_ok=True)
            candidate = _solve_joint_common_population(likelihood, initial,
                native['fixed_common_source_negative_log_likelihood_per_label'], optimizer).diagnostics
            fit_count += 1
            if not candidate['optimizer_converged']:
                save(folder/'before_completion.json', candidate)
                candidate = _solve_joint_common_population(likelihood, np.asarray(candidate['optimizer_parameters']),
                    native['fixed_common_source_negative_log_likelihood_per_label'], optimizer).diagnostics
                fit_count += 1
            save(folder/'candidate_fit.json', candidate)
            if not candidate['optimizer_converged']:
                raise RuntimeError(f'native g source fit incomplete at {layout}/{deleted}; no scores')
            improvement = native['source_negative_log_likelihood_per_label']-candidate['source_negative_log_likelihood_per_label']
            selected = improvement > optimizer['ftol']
            selected_means = deepcopy(means)
            if selected:
                selected_means['heads'][1]['joint_structural'] = candidate
            save(folder/'selected_source_fit.json', selected_means)
            fits.append({'layout': layout, 'deleted_task': deleted, 'source_nll_improvement_per_label': improvement,
                'selected_gp_start': bool(selected), 'initial_profile': profile,
                'native': native, 'candidate': candidate})
            sites = coverage['layouts'][layout][deleted]
            for case, fitted in zip(spec['cases'], (means, selected_means)):
                prior = components(inputs, fitted, risk, psi, np.asarray(inputs['public_risk_reference']))[0][1]
                pm, pc = prior.predict(full_cov=True); paid = deepcopy(prior)
                for j, y in zip(paid_sites, values[deleted]):
                    paid.update_batch(j, float(y), None, 1)
                qm, qc = paid.predict(full_cov=True)
                if case == 'native_structural':
                    reference = next(r for r in references if r['layout'] == layout and r['deleted_task'] == deleted and r['mean_arm'] == 'joint_structural')
                    for actual_m, actual_c, cached in ((pm, pc, reference['prior']), (qm, qc, reference)):
                        np.testing.assert_allclose(actual_m, np.asarray(cached['mean'])[:, [1, 2]], atol=1e-8)
                        variance = np.diag(actual_c.reshape(2*len(psi), 2*len(psi))).reshape(len(psi), 2)
                        np.testing.assert_allclose(variance, np.asarray(cached['variance'])[:, [1, 2]], atol=1e-8)
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': sites,
                    'predictions': {'prior': g_prediction(pm[:, 0], pc[:, 0, :, 0], sites),
                        'raw_paid': g_prediction(qm[:, 0], qc[:, 0, :, 0], sites)}})
                models[(layout, deleted, case)] = prior, paid
            print('native GP-profile source fit and forecasts frozen', layout, deleted, selected, improvement, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all24_branches_frozen': True,
        'native_objectives_and_all73_prior_paid_moments_reproduced': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']
        labels = source[layout]['source_tasks'][deleted]; observed = np.asarray(labels['response']); noise = np.asarray(labels['noise'])[:, 1]
        row['scores'] = {}
        for stage, model in zip(('prior', 'raw_paid'), models[(layout, deleted, case)]):
            prediction = row['predictions'][stage]; predicted = np.asarray(prediction['mean']); covariance = np.asarray(prediction['covariance'])
            row['scores'][stage] = {'g_mse': float(np.mean((predicted-observed[:, 1])**2)),
                'g_gaussian_mean_density': gaussian_score(predicted, covariance+np.diag(noise), observed[:, 1])/len(predicted),
                'g_raw_batch_density': float(np.mean([joint_batch_log_score(model.local_components(j), value[1], value[2], 3)
                    for j, value in zip(row['scoring_sites'], observed)]))}
    summary = {layout: {case: {stage: {k: np.mean([r['scores'][stage][k] for r in rows if r['layout'] == layout and r['case'] == case]).item()
        for k in rows[0]['scores'][stage]} for stage in ('prior', 'raw_paid')} for case in spec['cases']} for layout in spec['layouts']}
    contrasts = []
    for row in rows:
        if row['case'] == 'native_structural':
            continue
        reference = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == 'native_structural')
        contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'],
            'selected_minus_native': {stage: {k: row['scores'][stage][k]-reference['scores'][stage][k]
                for k in row['scores'][stage]} for stage in ('prior', 'raw_paid')}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts, 'fit_attribution': fits,
        'cost': {**spec['cost'], 'actual_new_g_source_fits': fit_count}, 'cpu_seconds': time.process_time()-started,
        'independent_source_gate_evaluated': False}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'selected_gp_starts': sum(d['selected_gp_start'] for d in fits),
        'new_g_source_fits': fit_count, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
