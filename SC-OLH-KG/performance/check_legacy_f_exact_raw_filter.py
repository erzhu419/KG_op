"""Separate four-observation Gaussian projection error from frozen prior error."""

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import errors
from performance.check_legacy_f_source_information_design import gaussian_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.exact_gaussian_raw_posterior import quadrature, moments, future_batch_density
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from performance.validate_legacy_paired_taskout import gaussian_score
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population
from representation.vectorized_joint_function_risk import update_raw_observation

EXECUTION = 'performance/manifests/legacy_joint_population_f_exact_raw_filter_execution_v1.json'


def prediction(m, grid):
    return {'mean': m['f_mean'][4:].tolist(), 'covariance': m['f_covariance'][4:, 4:].tolist(),
        'paid_predictive_log_score': grid['evidence']}


def prepare_cases(spec):
    control = read(spec['control_results']); assert control['scientific_gate_interpreted']
    coverage, profile = source_paths(read(spec['source_context_execution']))
    parent = read(coverage['source_execution']); old = read(spec['retained_source_execution'])
    adaptation = read(read(old['reference_execution'])['adaptation_execution'])
    designs = read(spec['design_results'])
    cases, frozen = [], []
    for layout in spec['layouts']:
        inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
        head = read(Path(profile['output']) / f'layout{layout}' / 'task1' / 'selected_source_fit.json')['heads'][0]
        psi = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(psi)
        rpsi = np.asarray(inputs['public_risk_reference']); rbasis = OrthogonalLowOrderBasis(rpsi)
        risk = read(Path(parent['output']) / f"layout{adaptation['parent_layouts'][layout]}" / 'task1' / 'risk_source_fit.json')
        rlaw = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
        eta = source_prior_moments(restore_joint(rbasis, rlaw), rpsi)
        design = next(r for r in designs['rows'] if r['layout'] == layout)
        for name in spec['designs']:
            prefix, sites = design['frozen_prefixes'][name], design['scoring_site_indices']
            indices = prefix + sites
            for arm in spec['arms']:
                f = predict_unequal_site_kernel_population(psi[indices], head['joint_gp']) if arm == 'source_gp' else source_prior_moments(restore_joint(basis, head['joint_structural']), psi[indices])
                prior = {'f_mean': f[0], 'f_covariance': f[1], 'eta_mean': eta[0][indices], 'eta_covariance': eta[1][np.ix_(indices, indices)]}
                baseline = next(r for r in control['rows'] if r['layout'] == layout and r['design'] == name and r['arm'] == arm)
                assert prefix == baseline['paid_prefix'] and sites == baseline['scoring_sites']
                identity = {'layout': layout, 'design': name, 'arm': arm, 'paid_prefix': prefix, 'scoring_sites': sites}
                cases.append({'identity': identity, 'prior': prior, 'baseline': baseline})
                frozen.append(identity)
    return cases, frozen, coverage


def run():
    started = time.process_time(); spec = read(EXECUTION)
    assert read(spec['source_package_results'])['diagnostic_interpreted']
    cases, frozen, coverage = prepare_cases(spec)
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    base = {'execution': spec, 'no_model_adopted': True, 'source6_7_evaluated': False,
        'target_promoted': False, 'independent_confirmation': False,
        'no_new_unscored_outcomes_read': True, 'training_diagnostic_only': True, 'cost': spec['cost']}
    save(out / 'frozen_inputs.json', {'execution': spec, 'all_eight_gaussian_priors_frozen_before_paid_values': True, 'rows': frozen})
    raw = read(coverage['source_data'])
    paid = {(c['identity']['layout'], c['identity']['design']): [raw['tasks'][1]['Y_replicates'][j][0][0] for j in c['identity']['paid_prefix']] for c in cases}
    del raw
    rows = []
    for case in cases:
        identity, prior = case['identity'], case['prior']; values = paid[(identity['layout'], identity['design'])]
        one_grid = quadrature(prior, values[:1], spec['orders'][-1]); one = moments(prior, one_grid)
        field = gaussian_field((prior['f_mean'], prior['f_covariance']), (prior['eta_mean'], prior['eta_covariance']))
        one_evidence = update_raw_observation(field, 0, values[0])
        combined_mean = np.column_stack([one['f_mean'], one['eta_mean']]).ravel()
        combined_covariance = np.empty((16, 2, 16, 2))
        combined_covariance[:, 0, :, 0] = one['f_covariance']; combined_covariance[:, 1, :, 1] = one['eta_covariance']
        combined_covariance[:, 0, :, 1] = one['cross_covariance']; combined_covariance[:, 1, :, 0] = one['cross_covariance'].T
        combined_covariance = combined_covariance.reshape(32, 32)
        one_errors = {'mean_max_absolute': float(np.max(np.abs(combined_mean - field.means[0]))),
            'covariance_relative_frobenius': float(np.linalg.norm(combined_covariance - field.covariances[0]) / np.linalg.norm(field.covariances[0])),
            'paid_log_evidence_absolute': abs(one_grid['evidence'] - one_evidence)}
        assert all(one_errors[k] <= v for k, v in spec['prediction_tolerances'].items())
        previous = None; history = []; selected = None
        for order in spec['orders']:
            if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise RuntimeError('registered exact-filter CPU budget reached')
            grid = quadrature(prior, values, order); m = moments(prior, grid); p = prediction(m, grid)
            history.append({'order': order, 'prediction': p})
            if previous is not None:
                difference = errors(previous[2], p)
                if all(difference[k] <= v for k, v in spec['prediction_tolerances'].items()):
                    selected = previous[0], previous[1], previous[2], grid, m, p, difference; break
            previous = grid, m, p
        row = {**identity, 'single_observation_errors': one_errors, 'quadrature_history': history,
            'moment_quadrature_verified': selected is not None}
        if selected is not None:
            case['integrals'] = selected
            row.update(prediction=selected[2], refined_prediction=selected[5], quadrature_errors=selected[6],
                selected_order=selected[0]['order'], refined_order=selected[3]['order'])
        rows.append(row)
        save(out / 'partial_results.json', {**base, 'rows': rows})
        print('exact filter moments', identity, 'complete', selected is not None, flush=True)
    complete = all(r['moment_quadrature_verified'] for r in rows)
    save(out / 'predictions.json', {**base, 'all_eight_moments_frozen_before_score_labels': True, 'rows': rows})
    if not complete:
        save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started}); return
    for row, case in zip(rows, cases):
        labels = read(Path(coverage['output']) / f"layout{row['layout']}" / 'inputs.json')['source_tasks'][1]
        observed = np.asarray(labels['response']); noise = np.asarray(labels['noise'])[:, 0]
        densities = []
        for grid, future_order in [(case['integrals'][0], spec['future_orders'][0]), (case['integrals'][3], spec['future_orders'][1])]:
            densities.append(future_batch_density(case['prior'], grid, observed, future_order))
        error = float(np.max(np.abs(densities[0] - densities[1])))
        row['density_refinement_error'] = error; row['density_quadrature_verified'] = error <= spec['density_refinement_tolerance']
        p = row['prediction']; moment = gaussian_score(np.asarray(p['mean']), np.asarray(p['covariance']) + np.diag(noise), observed[:, 0]) / 12
        scores = {'f_mse': float(np.mean((np.asarray(p['mean']) - observed[:, 0]) ** 2)), 'f_raw_batch_density': float(np.mean(densities[0])),
            'f_gaussian_mean_density': moment, 'raw_paid_log_evidence': p['paid_predictive_log_score'],
            'sequential_transcript_density': (p['paid_predictive_log_score'] + 12 * moment) / 16}
        row['scores'] = scores; row['exact_minus_projection'] = {k: v - case['baseline']['scores'][k] for k, v in scores.items()}
        row['prediction_difference_from_projection'] = errors(p, case['baseline']['prediction'])
        print('exact filter contrast', row['layout'], row['design'], row['arm'], row['exact_minus_projection'], 'density verified', row['density_quadrature_verified'], flush=True)
    save(out / 'results.json', {**base, 'rows': rows, 'diagnostic_interpreted': all(r['density_quadrature_verified'] for r in rows),
        'cpu_seconds': time.process_time() - started, 'interpretation': 'Exact under each frozen independent Gaussian f/eta prior,within the demonstrated quadrature error. Includes native point and GP controls,not the retained source-variance mixture. Source model and physical risk recovery are not established by this numerical diagnostic.'})


if __name__ == '__main__':
    run()
