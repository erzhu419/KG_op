"""Zero-shot task0 pilot of the raw-source positive function/risk mixture."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import digamma, logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_f_reference_training_budget import compact, prediction
from performance.check_legacy_f_task_level_reference_development import mixture_field
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.prepare_legacy_f_task_level_reference_quadrature import rule_on_intervals
from performance.prepare_legacy_f_variance_reference import VarianceReferencePosterior
from performance.run_legacy_related_inventory_source import save
from performance.vectorized_joint_batch_density import component_batch_log_evidence
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_kernel_population import rbf_cross

PLAN = 'performance/manifests/legacy_joint_population_f_raw_source_predictive_mixture_plan_v1.json'


def source_projection(field, index, law, risk, psi, risk_psi, reference, components, gp):
    retained = np.asarray(components['eta_states_by_chain'])
    eta_states = retained.reshape(-1, 48); count = len(eta_states); n = len(psi)
    rd = np.column_stack([np.ones(n), OrthogonalLowOrderBasis(risk['public_reference_psi']).transform(risk_psi)])
    rm = np.asarray(risk['prior_mean']); gain = np.asarray(risk['source_eta_gain'])
    risk_mean = (rd @ rm)[None, :] + (eta_states - field['eta_mean']) @ gain.T @ rd.T
    risk_covariance = rd @ (np.asarray(risk['conditional_covariance']) + np.asarray(risk['task_coefficient_covariance'])) @ rd.T
    risk_covariance += risk['white_discrepancy'] * np.eye(n); np.linalg.cholesky(risk_covariance)
    if index == 0:
        design = np.column_stack([np.ones(n), OrthogonalLowOrderBasis(reference).transform(psi)])
        cross = design @ np.asarray(field['common_latent_to_source_f_covariances'][index])
        qcov = design @ (np.asarray(law['common_coefficient_covariance']) + np.asarray(law['task_coefficient_covariance'])) @ design.T
        # The original discrepancy is fixed and included once for the new task.
        qcov += law['white_discrepancy'] * np.eye(n)
        prior_mean = design @ np.asarray(field['common_latent_prior_means'][index])
    else:
        theta = law['learned_parameters']; anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
        xs = np.asarray(gp['source_features'])
        cross = scale2 * (theta[0] * rbf_cross(psi, xs, anchor['lengthscale']) + anchor['constant_variance'])
        kernel = rbf_cross(psi, psi, anchor['lengthscale'])
        qcov = scale2 * ((theta[0] + theta[1]) * kernel + anchor['constant_variance'] + theta[2] + theta[3] * np.eye(n))
        prior_mean = np.full(n, field['f_means'][index][0])
    model = VarianceReferencePosterior(field['f_means'][index], field['f_covariances'][index], field['batch_means'], field['sample_variances'])
    return project_source_components(model, eta_states, prior_mean, qcov, cross, risk_mean, risk_covariance)


def project_source_components(model, eta_states, prior_mean, qcov, cross, risk_mean, risk_covariance):
    """Historical exact Gaussian completion, independent of the chosen basis."""
    count, n = len(eta_states), len(prior_mean)
    means, covariances = [], []
    for eta in eta_states:
        _, factor, alpha = model.evaluate(eta)
        covariance = qcov - cross @ cho_solve(factor, cross.T); covariance = .5 * (covariance + covariance.T)
        np.linalg.cholesky(covariance)
        means.append(prior_mean + cross @ alpha); covariances.append(covariance)
    means, covariances = np.asarray(means), np.asarray(covariances)
    local = np.zeros((count, n, 2, 2)); local[:, :, 0, 0] = np.diagonal(covariances, axis1=1, axis2=2)
    local[:, :, 1, 1] = np.diag(risk_covariance)
    return {'component_means': np.stack([means, risk_mean], axis=2), 'component_local_covariances': local,
        'f_component_covariances': covariances,
        'risk_component_covariances': np.broadcast_to(risk_covariance, (count, n, n)),
        'log_weights': np.full(count, -np.log(count)), 'paid_log_evidence': np.asarray(0.)}


def subset(state, indices):
    return {k: (v[indices] if k != 'paid_log_evidence' else v) for k, v in state.items()}


def chain_half_indices(count):
    assert count % 8 == 0
    per_chain = count // 4; half = per_chain // 2
    return [np.concatenate([np.arange(per_chain * c + start, per_chain * c + start + half) for c in range(4)])
        for start in (0, half)]


def split_prediction_checks(state):
    full = prediction(state); mean = np.asarray(full['mean']); diagonal = np.diag(full['covariance']); errors = []
    for indices in chain_half_indices(len(state['log_weights'])):
        half = subset(state, indices); half['log_weights'] = np.full(len(indices), -np.log(len(indices)))
        p = prediction(half)
        errors.append({'mean_rms_difference_in_full_sd': float(np.sqrt(np.mean((np.asarray(p['mean']) - mean) ** 2 / diagonal))),
            'maximum_variance_relative_difference': float(np.max(np.abs(np.diag(p['covariance']) / diagonal - 1)))})
    return errors


def run():
    started = time.process_time(); spec = read(PLAN)
    assert read(spec['shared_mixture_results'])['all_joint_mixture_gates_passed']
    fields = read(spec['source_inputs'])['rows']; risks = read(spec['reconstruction'])['rows']
    components = read(spec['eta_components'])['rows']; laws = read(spec['source_laws'])['rows']
    mean_spec = read(spec['source_mean_plan']); registry = read(mean_spec['source_registry'])['rows']
    original_laws = read(mean_spec['source_laws'])['rows']
    coverage, _ = source_paths(read(mean_spec['source_context_execution']))
    controls = read(spec['retained_control_results']); assert controls['diagnostic_interpreted']
    packages = controls['execution']['source_packages']
    branches = sum([read(Path(p) / 'source_branches.json')['rows'] for p in packages], [])
    rules = sum([read(Path(p) / 'source_quadrature.json')['rows'] for p in packages], [])
    known = read(read(spec['source_contract_plan'])['known_source_plan']); risk_execution = read(known['risk_source_execution'])
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True); work, frozen = [], []
    query = spec['source_validation_task']
    for field in fields:
        layout = field['layout']; branch = next(b for b in branches if (b['layout'], b['query_task']) == (layout, query))
        assert branch['source_task_names'] == [f'related_inventory_{i:02d}' for i in (2, 3, 4, 5)]
        sites = coverage['layouts'][layout][query]; assert len(sites) == 12
        psi = np.asarray(branch['query_mean_psi']); risk = next(r for r in risks if r['layout'] == layout)
        case = next(c for c in risk_execution['cases'] if (c['layout'], c['query_task']) == (layout, query))
        ri = read(Path(risk_execution['output']) / case['folder'] / 'risk_inputs.json')
        np.testing.assert_allclose(risk['public_reference_psi'], ri['public_reference_psi'], atol=1e-8, rtol=1e-8)
        risk_psi = np.asarray(ri['heldout_public_psi'])[sites]
        gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) == (layout, 'related_inventory_00')))['fits']['gp']
        for index, arm in enumerate(field['arms']):
            law = next(r for r in laws if (r['layout'], r['arm']) == (layout, arm))
            if index == 0:
                original = next(r for r in original_laws if (r['layout'], r['query_task']) == (layout, query))
                law = {**law, 'white_discrepancy': original['white_discrepancy']}
            component = next(r for r in components if (r['layout'], r['arm']) == (layout, arm))
            state = source_projection(field, index, law, risk, psi, risk_psi, branch['public_mean_reference'], component, gp)
            row = {'layout': layout, 'source_validation_task': query, 'arm': arm, 'sites': sites,
                'prediction': prediction(state), 'split_prediction_errors': split_prediction_checks(state)}
            work.append((row, state))
        rule = next(r for r in rules if r['layout'] == layout and r['arm'] == 'native'
            and r['source_reference_law']['source_task_names'] == branch['source_task_names'])
        eta = np.asarray(branch['eta_mean'])[sites], np.asarray(branch['eta_covariance'])[np.ix_(sites, sites)]
        basis = OrthogonalLowOrderBasis(branch['public_mean_reference'])
        refined_nodes, refined_weights = rule_on_intervals(rule['source_reference_law'], rule['adaptive_u_intervals'], 32)
        for resolution, nodes, weights in [('selected', rule['variance_nodes'], rule['posterior_weights']), ('refined', refined_nodes, refined_weights)]:
            native = mixture_field(basis, psi, rule['source_reference_law'], np.asarray(nodes), np.asarray(weights), eta)
            state = compact(native, list(range(12)), 0.)
            row = {'layout': layout, 'source_validation_task': query, 'arm': 'native_mixture', 'resolution': resolution,
                'sites': sites, 'prediction': prediction(state)}; work.append((row, state))
    for row, state in work:
        frozen.append({**row, 'component_means': state['component_means'].tolist(),
            'component_local_covariances': state['component_local_covariances'].tolist(), 'log_weights': state['log_weights'].tolist()})
    save(out / 'predictions.json', {'execution': spec, 'rows': frozen, 'all_eight_zero_shot_predictions_frozen_before_scores': True,
        'paid_target_observations': 0, 'joint_mixture_not_replaced_by_Gaussian': True})
    rows = []
    try:
        for row, state in work:
            inputs = read(Path(coverage['output']) / f"layout{row['layout']}" / 'task1' / 'inputs.json')
            task = next(t for t in inputs['source_tasks'] if t['name'] == 'related_inventory_00')
            labels = np.asarray(task['response']); noise = np.asarray(task['noise'])[:, 0]
            np.testing.assert_allclose(np.exp(labels[:, 3] + digamma(1.)) / 3, noise, atol=1e-12, rtol=1e-10)
            densities, integral_errors, split_errors = [], [], []
            for j, label in enumerate(labels):
                if time.process_time() - started >= spec['maximum_single_thread_cpu_seconds']: raise TimeoutError('registered CPU budget reached')
                logs, error = component_batch_log_evidence(state['component_means'][:, j], state['component_local_covariances'][:, j], label[0], label[3], 3)
                full = float(logsumexp(state['log_weights'] + logs)); densities.append(full); integral_errors.append(error)
                if row['arm'] != 'native_mixture':
                    indices = np.concatenate([np.arange(128 * c, 128 * c + 64) for c in range(4)])
                    complement = np.concatenate([np.arange(128 * c + 64, 128 * c + 128) for c in range(4)])
                    split_errors.extend(abs(float(logsumexp(logs[k]) - np.log(256)) - full) for k in (indices, complement))
            residual = np.asarray(row['prediction']['mean']) - labels[:, 0]
            row.update(scores={'f_mse': float(np.mean(residual ** 2)), 'f_raw_batch_density': float(np.mean(densities))},
                per_site_raw_batch_densities=densities, maximum_density_integration_relative_error_bound=max(integral_errors))
            if split_errors:
                row['maximum_split_raw_density_absolute_difference'] = max(split_errors)
                row['mixture_precision_gate_passed'] = (max(split_errors) <= .02
                    and all(e['mean_rms_difference_in_full_sd'] <= .1 and e['maximum_variance_relative_difference'] <= .15 for e in row['split_prediction_errors']))
            rows.append(row)
            print('zero-shot raw-source mixture', row['layout'], row['arm'], row.get('resolution', ''), row['scores'], flush=True)
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False, 'cpu_seconds': time.process_time() - started})
    except TimeoutError:
        save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
            'stopped_at_cpu_budget': True, 'cpu_seconds': time.process_time() - started}); return
    native_checks, comparisons, matched_gates = [], [], []
    for layout in spec['layouts']:
        selected, refined = [next(r for r in rows if r['layout'] == layout and r['arm'] == 'native_mixture' and r['resolution'] == res) for res in ('selected', 'refined')]
        a, b = selected['prediction'], refined['prediction']
        errors = {'mean_max_absolute': float(np.max(np.abs(np.asarray(a['mean']) - b['mean']))),
            'covariance_relative_frobenius': float(np.linalg.norm(np.asarray(a['covariance']) - b['covariance']) / np.linalg.norm(b['covariance'])),
            'raw_density_absolute': abs(selected['scores']['f_raw_batch_density'] - refined['scores']['f_raw_batch_density'])}
        native_checks.append({'layout': layout, 'errors': errors, 'passed': errors['mean_max_absolute'] <= 1e-7 and errors['covariance_relative_frobenius'] <= 1e-5 and errors['raw_density_absolute'] <= 1e-5})
        structure = next(r for r in rows if r['layout'] == layout and r['arm'] == 'joint_spectral_geometry')
        for arm, control in [('source_gp', next(r for r in rows if r['layout'] == layout and r['arm'] == 'source_gp')), ('native_mixture', selected)]:
            delta = {k: structure['scores'][k] - control['scores'][k] for k in spec['metrics']}
            comparisons.append({'layout': layout, 'reference': arm, 'structure_minus_reference': delta,
                'mse_relative_change': structure['scores']['f_mse'] / control['scores']['f_mse'] - 1})
            if arm == 'source_gp': matched_gates.append({'layout': layout, 'mse_nonincrease': delta['f_mse'] <= 0, 'raw_density_strict_improvement': delta['f_raw_batch_density'] > 0})
    numerical = all(r.get('mixture_precision_gate_passed', True) for r in rows) and all(r['passed'] for r in native_checks)
    predictive = all(g['mse_nonincrease'] and g['raw_density_strict_improvement'] for g in matched_gates) if numerical else None
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_eight_zero_shot_predictions_scored': True,
        'native_refinement_checks': native_checks, 'comparisons': comparisons, 'matched_GP_predictive_gates': matched_gates,
        'all_numerical_gates_passed': numerical, 'source_structure_predictive_gate_passed': predictive,
        'diagnostic_interpreted': numerical, 'cpu_seconds': time.process_time() - started, 'cost': spec['cost'],
        'no_model_adopted': True, 'independent_confirmation': False, 'outer_task1_outcomes_indexed': False,
        'interpretation': 'Zero-shot source4 development mixture comparison on12 task0 policies per layout. New raw arms share the original source likelihood and M step;native is the retained Gaussian-source intercept-mixture reference. No paid update,independent task,converged EM fit or decision advantage.'})
    print('raw-source zero-shot comparison', comparisons, 'numerical', numerical, 'predictive', predictive, flush=True)


if __name__ == '__main__':
    run()
