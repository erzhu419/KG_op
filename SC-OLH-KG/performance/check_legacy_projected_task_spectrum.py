"""Task covariance compression with a fixed projected source common posterior."""

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.check_legacy_public_mean_source_half import components
from performance.check_legacy_source_mean_covariance_swap import g_prediction
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.joint_function_risk import JointFunctionRiskPosterior, joint_batch_log_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis, spectral_weights
from representation.orthogonal_gp_projection import project_gaussian_field
from representation.public_control_mean import public_control_mean_coordinates
from representation.source_kernel_population import rbf_cross

EXECUTION = 'performance/manifests/legacy_joint_population_projected_task_spectrum_execution_v1.json'


def run():
    started = time.process_time(); spec = read(EXECUTION); projection = read(spec['projection_execution'])
    gain = read(projection['gain_execution']); adaptation = read(gain['adaptation_execution'])
    coverage = read(adaptation['coverage_execution']); deletion = read(adaptation['deleted_mean_execution']); parent = read(coverage['source_execution'])
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    old = read(Path(read(spec['projection_results'])['execution']['output'])/'calibration_profiles.json')['rows']
    raw = read(coverage['source_data']); paid_sites = spec['paid_candidates']
    values = {i: [raw['tasks'][i]['Y_replicates'][j][0][1]-raw['tasks'][i]['tau'] for j in paid_sites] for i in spec['training_tasks']}; del raw
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True); rows, models = [], {}
    for layout in spec['layouts']:
        for deleted in spec['training_tasks']:
            branch = Path(coverage['output'])/f'layout{layout}'/f'task{deleted}'
            inputs, gp = read(branch/'inputs.json'), read(branch/'gp_source_fit.json')
            means = read(Path(deletion['output'])/f'layout{layout}'/f'task{deleted}'/'source_fit.json')
            native = means['heads'][1]['joint_structural']
            means = dict(means, heads=[dict(h, joint_gp=g['joint_gp']) for h, g in zip(means['heads'], gp['heads'])])
            risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
            provider = no_outcomes(problem_for(public['plan'], public['tasks'][deleted]))
            psi = (public_control_mean_coordinates(provider, bank)-inputs['mean_coordinate_center'])/inputs['mean_coordinate_scale']
            field = components(inputs, means, risk, psi, np.asarray(inputs['public_risk_reference']))[2][1]
            m, c = field.predict(full_cov=True); n = len(m)
            basis = OrthogonalLowOrderBasis(inputs['public_mean_reference'])
            design = np.column_stack([np.ones(n), basis.transform(psi)])
            operator = np.linalg.lstsq(design, np.eye(n), rcond=None)[0]
            fit = gp['heads'][1]['joint_gp']; scale = fit['anchor']['normalization']['scale']**2
            amplitude, intercept, discrepancy = np.asarray(fit['residual_parameters'])*scale
            task = amplitude*rbf_cross(psi, psi, fit['anchor']['lengthscale'])+intercept
            task_coefficients = operator@task@operator.T; task_coefficients = .5*(task_coefficients+task_coefficients.T)
            common = operator@(c[:, 0, :, 0]-task-discrepancy*np.eye(n))@operator.T; common = .5*(common+common.T)
            projected = project_gaussian_field(design, m[:, 0], c[:, 0, :, 0], discrepancy)
            np.testing.assert_allclose(common+task_coefficients, projected['coefficient_covariance'], atol=1e-10)
            tail = .01*native['source_signal_scale']/basis.feature_dim
            diagonal = np.diag(task_coefficients).copy(); sizes = np.bincount(basis.group_index)
            group_mass = np.bincount(basis.group_index, weights=diagonal[1:])
            excess = np.maximum(group_mass-sizes*tail, 0.)
            weights, _ = spectral_weights(basis, native['spectral_exponent'])
            structured = np.r_[diagonal[0], excess[basis.group_index]*weights+tail]
            variants = {'projected_full_task': task_coefficients, 'diagonal_task': np.diag(diagonal), 'native_group_spectrum_tail': np.diag(structured)}
            mean = projected['mean']; sites = coverage['layouts'][layout][deleted]
            for case, task_covariance in variants.items():
                covariance = design@(common+task_covariance)@design.T+discrepancy*np.eye(n)
                fm = np.column_stack([mean, m[:, 1]]); fc = np.zeros((n, 2, n, 2))
                fc[:, 0, :, 0], fc[:, 1, :, 1] = covariance, c[:, 1, :, 1]
                prior = JointFunctionRiskPosterior.from_moments(fm, fc); paid = deepcopy(prior)
                for j, y in zip(paid_sites, values[deleted]):
                    paid.update_batch(j, float(y), None, 1)
                pm, pc = paid.predict(full_cov=True)
                if case == 'projected_full_task':
                    bm = np.column_stack([projected['mean'], m[:, 1]]); bc = np.zeros_like(fc)
                    bc[:, 0, :, 0], bc[:, 1, :, 1] = projected['covariance'], c[:, 1, :, 1]
                    baseline = JointFunctionRiskPosterior.from_moments(bm, bc)
                    for j, y in zip(paid_sites, values[deleted]):
                        baseline.update_batch(j, float(y), None, 1)
                    expected_m, expected_c = baseline.predict(full_cov=True)
                    np.testing.assert_allclose(fm, bm, atol=1e-10); np.testing.assert_allclose(fc, bc, atol=1e-10)
                    np.testing.assert_allclose(pm, expected_m, atol=1e-8); np.testing.assert_allclose(pc, expected_c, atol=1e-8)
                    cached = next(r for r in old if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == 'full_dictionary')
                    for stage, predicted_m, predicted_c in (('prior', fm[:, 0], fc[:, 0, :, 0]), ('raw_paid', pm[:, 0], pc[:, 0, :, 0])):
                        np.testing.assert_allclose(predicted_m[sites], cached['predictions'][stage]['mean'], atol=1e-8)
                        np.testing.assert_allclose(predicted_c[np.ix_(sites, sites)], cached['predictions'][stage]['covariance'], atol=1e-8)
                off = task_coefficients-np.diag(diagonal)
                diagnostics = {'task_intercept_variance': float(diagonal[0]), 'native_task_intercept_variance': native['residual_coefficient_diagonal'][0],
                    'task_coefficient_off_diagonal_squared_mass_fraction': float(np.sum(off**2)/np.sum(task_coefficients**2)),
                    'native_task_spectral_exponent': native['spectral_exponent'], 'native_tail': float(tail),
                    'projected_task_group_mass': group_mass.tolist(), 'native_spectral_task_group_mass': (excess+sizes*tail).tolist(),
                    'task_trace_added_by_native_tail': float(np.sum(excess+sizes*tail-group_mass)),
                    'mean_paid_variance_reduction_fraction': float(np.mean(1-np.diag(pc[:, 0, :, 0])[sites]/np.diag(covariance)[sites]))}
                rows.append({'layout': layout, 'deleted_task': deleted, 'case': case, 'scoring_sites': sites, 'diagnostics': diagnostics,
                    'predictions': {'prior': g_prediction(mean, covariance, sites), 'raw_paid': g_prediction(pm[:, 0], pc[:, 0, :, 0], sites)}})
                models[(layout, deleted, case)] = prior, paid
        print('projected task covariance forecasts frozen', layout, flush=True)
    save(out/'predictions.json', {'execution': spec, 'all24_compressed_branches_frozen': True, 'projected_baseline_reproduced': True, 'rows': rows})
    source = {layout: read(Path(coverage['output'])/f'layout{layout}'/'inputs.json') for layout in spec['layouts']}
    for row in rows:
        layout, deleted, case = row['layout'], row['deleted_task'], row['case']
        if case == 'projected_full_task':
            cached = next(r for r in old if r['layout'] == layout and r['deleted_task'] == deleted and r['case'] == 'full_dictionary')
            row['scores'] = cached['scores']; continue
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
        if row['case'] == 'projected_full_task':
            continue
        reference = next(r for r in rows if r['layout'] == row['layout'] and r['deleted_task'] == row['deleted_task'] and r['case'] == 'projected_full_task')
        for stage in ('prior', 'raw_paid'):
            a, b = row['scores'][stage], reference['scores'][stage]
            contrasts.append({'layout': row['layout'], 'deleted_task': row['deleted_task'], 'case': row['case'], 'stage': stage,
                'compressed_minus_full': {k: a[k]-b[k] for k in a}})
    result = {'execution': spec, 'summary': summary, 'contrasts': contrasts,
        'diagnostics': [{k: r[k] for k in ('layout', 'deleted_task', 'case', 'diagnostics')} for r in rows],
        'projected_baseline_reproduction_passed': True, 'no_model_selected': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started}
    save(out/'calibration_profiles.json', {'execution': spec, 'rows': rows}); save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
