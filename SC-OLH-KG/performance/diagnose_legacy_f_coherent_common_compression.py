"""Source GP common-prior compression before conditioning in the original span."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_kernel_population import rbf_cross, shared_kernel

EXECUTION = 'performance/manifests/legacy_joint_population_f_coherent_common_compression_execution_v1.json'


def condition(kernel, residual, error):
    factor = cho_factor(kernel+residual, lower=True); alpha = cho_solve(factor, error)
    n = len(error)
    nll = float(.5*(n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+error@alpha))
    return nll, alpha, factor


def diagnose(inputs, native, gp, spec):
    reference = np.asarray(inputs['public_mean_reference']); basis = OrthogonalLowOrderBasis(reference)
    design = np.column_stack([np.ones(len(reference)), basis.transform(reference)])
    tasks = inputs['source_tasks']; features = [np.asarray(t['features']) for t in tasks]
    source_features = np.vstack(features); np.testing.assert_array_equal(source_features, gp['source_features'])
    distance = np.sum((source_features[:, None, :]-reference[None, :, :])**2, axis=2)
    sites = np.argmin(distance, axis=1); np.testing.assert_allclose(reference[sites], source_features, atol=1e-12, rtol=0.)
    anchor = gp['anchor']; scale = anchor['normalization']['scale']
    constant = anchor['normalization']['mean']+scale*anchor['constant']
    response = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks]); error = response-constant
    amplitude, intercept, discrepancy = gp['residual_parameters']
    residual = block_diag(*[scale**2*(amplitude*rbf_cross(x, x, anchor['lengthscale'])+intercept)
        +np.diag(np.asarray(t['noise'])[:, 0]+scale**2*discrepancy) for t, x in zip(tasks, features)])
    kernel = scale**2*shared_kernel(reference, reference, anchor)
    source_kernel = kernel[np.ix_(sites, sites)]
    full_nll, alpha, factor = condition(source_kernel, residual, error)
    expected_nll = gp['source_negative_log_likelihood_per_label']+np.log(scale)
    np.testing.assert_allclose(full_nll/len(error), expected_nll, atol=1e-8, rtol=1e-8)
    cross = kernel[:, sites]
    common_mean = constant+cross@alpha
    common_h = kernel-cross@cho_solve(factor, cross.T); common_h = .5*(common_h+common_h.T)
    normalized_cross = shared_kernel(reference, source_features, anchor)
    cached_mean = constant+scale*normalized_cross@np.asarray(gp['common_alpha'])
    cached_h = kernel-scale**2*normalized_cross@np.asarray(gp['common_precision'])@normalized_cross.T
    np.testing.assert_allclose(common_mean, cached_mean, atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(common_h, cached_h, atol=1e-8, rtol=1e-8)
    native_mean = design@np.asarray(native['coefficient_mean'])
    native_h = design@np.asarray(native['population_mean_covariance'])@design.T
    active = np.asarray(native['common_group_variances']) > 0
    scopes = {'full_dictionary': list(range(design.shape[1])),
        'native_common_groups': [0]+[j+1 for j, g in enumerate(basis.group_index) if active[g]]}
    laws, records = [], []
    for scope, columns in scopes.items():
        d = design[:, columns]; source_d = d[sites]
        operator, _, rank, _ = np.linalg.lstsq(d, np.eye(len(d)), rcond=None)
        assert rank == len(columns)
        projector = d@operator; complement = np.eye(len(d))-projector
        a = operator@kernel@operator.T; a = .5*(a+a.T)
        eigenvalues = np.linalg.eigvalsh(a); assert eigenvalues.min() >= -1e-10
        prior_beta = np.zeros(len(columns)); prior_beta[0] = constant
        inside_kernel = projector@kernel@projector.T
        outside_kernel = complement@kernel@complement.T
        cross_kernel = projector@kernel@complement.T+complement@kernel@projector.T
        np.testing.assert_allclose(inside_kernel+outside_kernel+cross_kernel, kernel, atol=1e-10, rtol=1e-9)
        compressed_kernel = source_d@a@source_d.T
        nll, compressed_alpha, compressed_factor = condition(compressed_kernel, residual, error)
        beta = prior_beta+a@source_d.T@compressed_alpha
        h = a-a@source_d.T@cho_solve(compressed_factor, source_d)@a; h = .5*(h+h.T)
        assert np.linalg.eigvalsh(h).min() >= -1e-10
        coherent_mean = d@beta; coherent_h = d@h@d.T
        posthoc_beta = operator@common_mean; posthoc_h = operator@common_h@operator.T
        posthoc_mean = d@posthoc_beta; posthoc_covariance = d@posthoc_h@d.T
        difference = common_mean-native_mean
        inside_difference = projector@difference; outside_difference = complement@difference
        np.testing.assert_allclose(difference@difference, inside_difference@inside_difference+outside_difference@outside_difference,
            atol=1e-10, rtol=1e-9)
        bank_variance = float(np.trace(kernel)); observed_variance = float(np.trace(source_kernel))
        covariance_scale = np.linalg.norm(posthoc_covariance)
        records.append({'scope': scope, 'column_count': len(columns), 'columns': columns,
            'public_point_count': len(reference), 'source_record_count': len(error),
            'full_gp_source_nll_per_label_physical': full_nll/len(error),
            'compressed_gp_source_nll_per_label_physical': nll/len(error),
            'compression_total_source_nll_change': nll-full_nll,
            'common_prior_outside_scope_bank_variance_fraction': float(np.trace(outside_kernel)/bank_variance),
            'common_prior_outside_scope_source_variance_fraction': float(np.trace(outside_kernel[np.ix_(sites, sites)])/observed_variance),
            'common_prior_cross_scope_source_trace_fraction': float(np.trace(cross_kernel[np.ix_(sites, sites)])/observed_variance),
            'common_prior_cross_scope_relative_frobenius_norm': float(np.linalg.norm(cross_kernel)/np.linalg.norm(kernel)),
            'gp_posterior_common_mean_outside_scope_bank_rms': float(np.sqrt(np.mean((common_mean-posthoc_mean)**2))),
            'gp_posterior_common_mean_outside_scope_source_rms': float(np.sqrt(np.mean((common_mean[sites]-posthoc_mean[sites])**2))),
            'native_gp_common_mean_bank_gap_rms': float(np.sqrt(np.mean(difference**2))),
            'native_gp_common_mean_within_scope_bank_gap_rms': float(np.sqrt(np.mean(inside_difference**2))),
            'native_gp_common_mean_outside_scope_bank_gap_rms': float(np.sqrt(np.mean(outside_difference**2))),
            'native_common_mean_outside_scope_bank_rms': float(np.sqrt(np.mean((complement@native_mean)**2))),
            'coherent_minus_posthoc_common_mean_bank_rms': float(np.sqrt(np.mean((coherent_mean-posthoc_mean)**2))),
            'coherent_minus_posthoc_common_mean_source_rms': float(np.sqrt(np.mean((coherent_mean[sites]-posthoc_mean[sites])**2))),
            'coherent_minus_posthoc_common_covariance_relative_frobenius': float(np.linalg.norm(coherent_h-posthoc_covariance)/covariance_scale) if covariance_scale > 0 else None,
            'mean_bank_common_variance': {'full_gp': float(np.trace(common_h)/len(reference)),
                'posterior_projected_gp': float(np.trace(posthoc_covariance)/len(reference)),
                'prior_projected_conditioned_gp': float(np.trace(coherent_h)/len(reference)),
                'native_f': float(np.trace(native_h)/len(reference))},
            'native_minus_coherent_common_mean_bank_rms': float(np.sqrt(np.mean((native_mean-coherent_mean)**2)))})
        laws.append({'scope': scope, 'columns': columns, 'coefficient_names': [(['intercept']+basis.names)[j] for j in columns],
            'source_task_names': [t['name'] for t in tasks], 'prior_coefficient_mean': prior_beta.tolist(),
            'prior_coefficient_covariance': a.tolist(), 'prior_coefficient_covariance_eigenvalues': eigenvalues.tolist(),
            'coherent_posterior_coefficient_mean': beta.tolist(), 'coherent_posterior_coefficient_covariance': h.tolist(),
            'posthoc_projected_coefficient_mean': posthoc_beta.tolist(), 'posthoc_projected_coefficient_covariance': posthoc_h.tolist(),
            'gp_task_kernel_and_noise_preserved': True, 'native_task_spectral_tail_contract_preserved': False})
    return laws, records, {'full_gp_source_covariance_nll_and_common_moments_verified': True,
        'physical_nll_normalization_error': float(abs(full_nll/len(error)-expected_nll)),
        'cached_common_mean_max_error': float(np.max(np.abs(common_mean-cached_mean))),
        'cached_common_covariance_max_error': float(np.max(np.abs(common_h-cached_h))),
        'source_coordinates_match_public_bank': True}


def run():
    started = time.process_time(); spec = read(EXECUTION); centering = read(spec['centering_execution'])
    coverage = read(centering['coverage_execution']); profile = read(centering['profile_execution'])
    native_full = read(spec['native_full_execution']); laws, rows, verification = [], [], []
    for layout in spec['layouts']:
        folders = [(None, Path(coverage['output'])/f'layout{layout}', Path(native_full['output'])/f'layout{layout}')]
        folders += [(i, Path(coverage['output'])/f'layout{layout}'/f'task{i}', Path(profile['output'])/f'layout{layout}'/f'task{i}')
            for i in spec['training_tasks']]
        for deleted, inputs_folder, fit_folder in folders:
            inputs = read(inputs_folder/'inputs.json'); means = read(fit_folder/'selected_source_fit.json')
            gp_filename = 'source_fit.json' if deleted is None else 'gp_source_fit.json'
            gp = read(inputs_folder/gp_filename)['heads'][0]['joint_gp']
            branch_laws, branch_rows, checked = diagnose(inputs, means['heads'][0]['joint_structural'], gp, spec)
            laws += [{'layout': layout, 'deleted_task': deleted, **law} for law in branch_laws]
            rows += [{'layout': layout, 'deleted_task': deleted, **row} for row in branch_rows]
            verification.append({'layout': layout, 'deleted_task': deleted, **checked})
        print('coherent source f common-prior compression complete', layout, flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out/'source_laws.json', {'execution': spec, 'all28_finite_source_laws_frozen_without_paid_outcomes': True, 'rows': laws})
    fields = ('compression_total_source_nll_change', 'common_prior_outside_scope_bank_variance_fraction',
        'gp_posterior_common_mean_outside_scope_bank_rms', 'native_gp_common_mean_bank_gap_rms',
        'coherent_minus_posthoc_common_mean_bank_rms', 'coherent_minus_posthoc_common_covariance_relative_frobenius')
    summary = {layout: {scope: {key: float(np.mean([r[key] for r in rows if r['layout'] == layout and r['scope'] == scope]))
        for key in fields} for scope in spec['scopes']} for layout in spec['layouts']}
    result = {'execution': spec, 'summary': summary, 'rows': rows, 'verification': verification,
        'all14_full_gp_source_laws_and28_compressions_verified': len(verification) == 14 and len(rows) == 28,
        'no_model_adopted': True, 'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'query_paid_outcomes_read': False, 'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
