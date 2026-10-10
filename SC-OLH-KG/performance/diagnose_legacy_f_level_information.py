"""Source-only level information; no paid observations or forecast replay."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.run_legacy_joint_population_source_budget import restore_joint
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.public_control_mean import source_prior_moments
from representation.source_kernel_population import predict_unequal_site_kernel_population, rbf_cross, shared_kernel

EXECUTION = 'performance/manifests/legacy_joint_population_f_level_information_execution_v1.json'


def coefficient_field(design, mean, common, residual, discrepancy):
    parts = {'common_function': design @ common @ design.T,
             'task_function': (design * residual) @ design.T,
             'discrepancy': discrepancy * np.eye(len(design))}
    return design @ mean, sum(parts.values()), parts


def gp_field(psi, fit):
    mean, covariance = predict_unequal_site_kernel_population(psi, fit)
    anchor = fit['anchor']; xs = np.asarray(fit['source_features'])
    cross = shared_kernel(psi, xs, anchor); scale2 = anchor['normalization']['scale']**2
    amplitude, intercept, discrepancy = fit['residual_parameters']
    parts = {'common_function': scale2 * (shared_kernel(psi, psi, anchor)
        - cross @ np.asarray(fit['common_precision']) @ cross.T),
        'task_function': scale2 * (amplitude * rbf_cross(psi, psi, anchor['lengthscale']) + intercept),
        'discrepancy': scale2 * discrepancy * np.eye(len(psi))}
    np.testing.assert_allclose(sum(parts.values()), covariance, atol=1e-12, rtol=1e-10)
    return mean, covariance, parts


def level_information(field, weights, prefix, noise):
    mean, covariance, parts = field
    level_mean = float(weights @ mean); level_variance = float(weights @ covariance @ weights)
    cross = weights @ covariance[:, prefix]
    latent = covariance[np.ix_(prefix, prefix)]
    factor = cho_factor(latent + np.diag(noise), lower=True)
    gain = cho_solve(factor, cross)
    reduction = float(cross @ gain)
    centered = latent - cross[:, None] - cross[None, :] + level_variance
    coupling = cross - level_variance
    shape_correlation = np.divide(coupling, np.sqrt(level_variance * np.diag(centered)),
                                  out=np.zeros_like(coupling), where=np.diag(centered) > 0)
    return {'prior_mean': level_mean, 'prior_variance': level_variance,
        'variance_components': {name: float(weights @ value @ weights) for name, value in parts.items()},
        'level_to_prefix_covariance': cross.tolist(),
        'level_to_centered_prefix_shape_covariance': coupling.tolist(),
        'level_to_centered_prefix_shape_correlation': shape_correlation.tolist(),
        'linear_gain_weights': gain.tolist(), 'gain_sum': float(gain.sum()),
        'common_shift_retention': float(1-gain.sum()),
        'conditional_mean_intercept': float(level_mean-gain @ mean[prefix]),
        'conditional_level_variance': float(level_variance-reduction),
        'explained_level_variance_fraction': reduction/level_variance,
        'observation_noise_in_linear_estimator': float(np.sum(noise * gain**2)),
        'prefix_latent_variances': np.diag(latent).tolist(),
        'prefix_prior_means': mean[prefix].tolist()}


def design_support(design, source_design, prefix, rosters):
    prefix_design = design[prefix]
    projection = np.linalg.pinv(prefix_design) @ prefix_design
    return {'coefficient_dimension': design.shape[1],
        'source_design_rank': int(np.linalg.matrix_rank(source_design)),
        'public_design_rank': int(np.linalg.matrix_rank(design)),
        'prefix_design_rank': int(np.linalg.matrix_rank(prefix_design)),
        'prefix_policy_indices': prefix,
        'level_functional_outside_prefix_row_span': {name: float(np.linalg.norm(
            weights @ design @ (np.eye(design.shape[1])-projection))/np.linalg.norm(weights @ design))
            for name, weights in rosters.items()},
        'support_definition': 'Euclidean coefficient-space row-span diagnostic;not posterior information or a paid prediction score'}


def run():
    started = time.process_time(); spec = read(EXECUTION)
    development = read(spec['development_execution']); compact = read(development['compact_execution'])
    reference = read(compact['reference_execution']); geometry = read(reference['joint_execution'])
    shape = read(geometry['shape_execution']); coverage, profile = source_paths(read(shape['source_context_execution']))
    old = read(spec['retained_source_execution']); parent = read(coverage['source_execution'])
    adaptation = read(read(old['reference_execution'])['adaptation_execution'])
    intercept_laws = read(spec['intercept_source_laws'])['rows']; joint_laws = read(spec['joint_source_laws'])['rows']
    public_prefixes = read(old['public_prefixes'])['rows']; rows = []
    for layout in spec['layouts']:
        deleted = spec['deleted_task']; suffix = Path(f'layout{layout}')/f'task{deleted}'
        inputs = read(Path(coverage['output'])/suffix/'inputs.json')
        head = read(Path(profile['output'])/suffix/'selected_source_fit.json')['heads'][0]
        names = [task['name'] for task in inputs['source_tasks']]
        assert set(names) == {f'related_inventory_{i:02d}' for i in range(6) if i != deleted}
        assert names == [task['name'] for task in inputs['risk_source_tasks']]
        prefix = next(r['public_prefix'] for r in public_prefixes if r['layout'] == layout and r['deleted_task'] == deleted)[:4]
        sites = coverage['layouts'][layout][deleted]; assert not set(prefix).intersection(sites)
        psi = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(psi)
        design = np.column_stack([np.ones(len(psi)), basis.transform(psi)])
        native = head['joint_structural']; fields = {'native': coefficient_field(design,
            np.asarray(native['coefficient_mean']), np.asarray(native['population_mean_covariance']),
            np.asarray(native['residual_coefficient_diagonal']), native['discrepancy_variance'])}
        restored = source_prior_moments(restore_joint(basis, native), psi)
        for actual, expected in zip(fields['native'][:2], restored):
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-10)
        intercept = next(r for r in intercept_laws if r['layout'] == layout and r['deleted_task'] == deleted)
        joint = next(r for r in joint_laws if r['layout'] == layout and r['deleted_task'] == deleted)
        assert intercept['source_task_names'] == joint['source_reference_law']['source_task_names'] == names
        residual = np.asarray(native['residual_coefficient_diagonal']).copy()
        residual[0] = intercept['integrated_task_intercept_variance']
        fields['intercept_mixture'] = coefficient_field(design, np.asarray(intercept['integrated_common_coefficient_mean']),
            np.asarray(intercept['integrated_common_coefficient_covariance']), residual, native['discrepancy_variance'])
        moments = joint['integrated_source_moments']; source = joint['source_reference_law']
        unit = np.asarray(source['task_group_unit_coefficient_diagonal'])
        residual = np.asarray(source['residual_diagonal_without_selected_shape_excess_at_native_level']).copy()
        residual[0] = moments['level_variance_mean']; residual += moments['shape_variance_mean'] * unit
        native_residual = np.asarray(source['residual_diagonal_without_selected_shape_excess_at_native_level']).copy()
        native_residual[0] = source['native_level_variance']; native_residual += source['native_shape_variance'] * unit
        np.testing.assert_allclose(native_residual, native['residual_coefficient_diagonal'], atol=1e-12, rtol=1e-10)
        fields['joint_level_shape_mixture'] = coefficient_field(design,
            np.asarray(moments['integrated_common_coefficient_mean']), np.asarray(moments['integrated_common_coefficient_covariance']),
            residual, native['discrepancy_variance'])
        fields['source_gp'] = gp_field(psi, head['joint_gp'])
        risk = read(Path(parent['output'])/f"layout{adaptation['parent_layouts'][layout]}"/f'task{deleted}'/'risk_source_fit.json')
        eta_law = next(h['joint_structural'] for h in risk['heads'] if h['head'] == 3)
        risk_psi = np.asarray(inputs['public_risk_reference']); risk_basis = OrthogonalLowOrderBasis(risk_psi)
        eta_mean, eta_covariance = source_prior_moments(restore_joint(risk_basis, eta_law), risk_psi)
        noise = np.exp(eta_mean[prefix] + .5*np.diag(eta_covariance)[prefix])
        weights = np.zeros(len(psi)); weights[sites] = 1/len(sites)
        rosters = {'public_bank': np.ones(len(psi))/len(psi), 'fixed_score_roster': weights}
        information = {arm: {name: level_information(field, w, prefix, noise) for name, w in rosters.items()}
                       for arm, field in fields.items()}
        source_design = np.vstack([np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))])
                                  for t in inputs['source_tasks']])
        gp = head['joint_gp']; normalization = gp['anchor']['normalization']
        centers = {'structural_empirical_intercept': head['regularizer']['coefficient_mean'][0],
            'structural_common_intercept_prior_variance': native['common_intercept_variance_fixed'],
            'structural_common_intercept_posterior_mean': native['coefficient_mean'][0],
            'structural_common_intercept_posterior_variance': native['population_mean_covariance'][0][0],
            'structural_native_task_intercept_variance': native['residual_coefficient_diagonal'][0],
            'gp_empirical_constant': normalization['mean'] + normalization['scale'] * gp['anchor']['constant'],
            'gp_common_intercept_prior_variance': normalization['scale']**2 * gp['anchor']['constant_variance'],
            'gp_task_intercept_variance': normalization['scale']**2 * gp['residual_parameters'][1],
            'uncertainty_contract': 'Both condition on source-fitted empirical centers/hyperparameters;shared-function posterior uncertainty retained. Mixtures additionally integrate registered variance uncertainty.'}
        rows.append({'layout': layout, 'deleted_task': deleted, 'source_task_names': names,
            'native_prior_and_joint_native_residual_reproduced': True,
            'eta_expected_observation_variances_at_prefix': noise.tolist(), 'scoring_site_indices': sites,
            'information': information, 'empirical_center_audit': centers,
            'design_support': design_support(design, source_design, prefix, rosters)})
        print({'layout': layout, 'score_level_information': {arm: {
            key: values['fixed_score_roster'][key] for key in ['prior_mean', 'prior_variance', 'gain_sum',
                'explained_level_variance_fraction', 'conditional_mean_intercept']}
            for arm, values in information.items()}, 'centers': centers}, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'rows': rows, 'cpu_seconds': time.process_time()-started, 'cost': spec['cost'],
        'query_paid_outcomes_read': False, 'no_forecast_or_replacement_prior_adopted': True,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False,
        'linear_diagnostic_contract': 'Exact first/second source-prior moments plus E[exp(eta_f)] observation noise;linear projection information only,not the nonlinear raw moment/mixture filter.'}
    save(out/'results.json', result)


if __name__ == '__main__':
    run()
