"""Fixed source-law attribution of finite common and task covariances."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_coherent_common_compression import condition
from performance.run_legacy_related_inventory_source import save
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_kernel_population import rbf_cross, shared_kernel

EXECUTION = 'performance/manifests/legacy_joint_population_f_coherent_covariance_attribution_execution_v1.json'


def diagnose(inputs, means, gp, compressed, spec):
    tasks = inputs['source_tasks']; reference = np.asarray(inputs['public_mean_reference'])
    basis = OrthogonalLowOrderBasis(reference)
    d = np.column_stack([np.ones(len(reference)), basis.transform(reference)])
    designs = [np.column_stack([np.ones(len(t['psi'])), basis.transform(np.asarray(t['psi']))]) for t in tasks]
    ds = np.vstack(designs); y = np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks])
    features = [np.asarray(t['features']) for t in tasks]
    np.testing.assert_array_equal(np.vstack(features), gp['source_features'])
    native = means['joint_structural']; m0 = np.asarray(means['regularizer']['coefficient_mean'])
    a0 = np.diag(native['common_prior_coefficient_diagonal']); r0 = np.asarray(native['residual_coefficient_diagonal'])
    v0 = block_diag(*[(z*r0)@z.T+np.diag(np.asarray(t['noise'])[:, 0]+native['discrepancy_variance'])
        for z, t in zip(designs, tasks)])
    anchor = gp['anchor']; scale = anchor['normalization']['scale']
    amplitude, intercept, discrepancy = gp['residual_parameters']
    v1 = block_diag(*[scale**2*(amplitude*rbf_cross(x, x, anchor['lengthscale'])+intercept)
        +np.diag(np.asarray(t['noise'])[:, 0]+scale**2*discrepancy) for t, x in zip(tasks, features)])
    a1 = np.asarray(compressed['prior_coefficient_covariance']); m1 = np.asarray(compressed['prior_coefficient_mean'])
    shape = a1.copy(); shape[0, :] = 0.; shape[:, 0] = 0.; shape[0, 0] = a0[0, 0]
    branches = {
        'native': (a0, v0, m0), 'native_common_gp_task': (a0, v1, m0),
        'gp_common_native_task': (a1, v0, m0), 'gp_common_gp_task': (a1, v1, m0),
        'gp_diagonal_native_task': (np.diag(np.diag(a1)), v0, m0),
        'gp_shape_native_intercept_native_task': (shape, v0, m0),
        'gp_shape_diagonal_native_intercept_native_task': (np.diag(np.diag(shape)), v0, m0),
        'native_gp_center': (a0, v0, m1), 'gp_common_gp_task_gp_center': (a1, v1, m1)}
    assert list(branches) == spec['branches']
    states = {}
    for name, (a, v, mean) in branches.items():
        value, alpha, factor = condition(ds@a@ds.T, v, y-ds@mean)
        beta = mean+a@ds.T@alpha
        h = a-a@ds.T@cho_solve(factor, ds)@a; h = .5*(h+h.T)
        states[name] = (value, beta, h, d@beta)
    value, beta, h, baseline = states['native']
    np.testing.assert_allclose(value/len(y), native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(beta, native['coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(h, native['population_mean_covariance'], atol=1e-8, rtol=1e-8)
    endpoint = states['gp_common_gp_task_gp_center']
    np.testing.assert_allclose(endpoint[1], compressed['coherent_posterior_coefficient_mean'], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(endpoint[2], compressed['coherent_posterior_coefficient_covariance'], atol=1e-8, rtol=1e-8)
    gap = endpoint[3]-baseline; gap_squared = float(gap@gap)
    rows = [{'branch': name, 'source_nll_per_label': s[0]/len(y), 'total_source_nll_change_vs_native': s[0]-value,
        'common_mean_bank_rms_shift_vs_native': float(np.sqrt(np.mean((s[3]-baseline)**2))),
        'common_mean_bank_rms_gap_to_coherent_gp': float(np.sqrt(np.mean((s[3]-endpoint[3])**2))),
        'signed_fraction_of_coherent_gp_native_mean_difference': float((s[3]-baseline)@gap/gap_squared),
        'mean_common_posterior_bank_variance': float(np.trace(d@s[2]@d.T)/len(d))}
        for name, s in states.items()]
    centered = np.eye(len(d))-np.ones((len(d), len(d)))/len(d)
    kernel = scale**2*shared_kernel(reference, reference, anchor)
    centered_variance = float(np.trace(centered@kernel@centered))
    scopes = {}
    for name, columns in [('full_dictionary', list(range(d.shape[1]))),
        ('native_common_groups', [0]+[j+1 for j, g in enumerate(basis.group_index) if native['common_group_variances'][g] > 0])]:
        z = d[:, columns]; operator = np.linalg.lstsq(z, np.eye(len(d)), rcond=None)[0]
        outside = np.eye(len(d))-z@operator
        scopes[name] = float(np.trace(outside@kernel@outside.T)/centered_variance)
    geometry = {'source_label_count': len(y), 'native_prior_center': float(m0[0]), 'gp_prior_center': float(m1[0]),
        'native_common_intercept_variance': float(a0[0, 0]), 'gp_common_intercept_variance': float(a1[0, 0]),
        'native_common_centered_bank_variance': float(np.trace(centered@d@a0@d.T@centered)/len(d)),
        'gp_common_centered_bank_variance': centered_variance/len(d),
        'gp_prior_outside_scope_fraction_of_centered_bank_variance': scopes,
        'common_shape_groups': {name: {'native_total_coefficient_variance': float(np.diag(a0)[1:][basis.group_index == g].sum()),
            'gp_total_coefficient_variance': float(np.diag(a1)[1:][basis.group_index == g].sum())}
            for g, name in enumerate(basis.group_names)},
        'native_task_mean_diagonal_variance_without_observation_noise': float(np.trace(v0)/len(y)-np.mean(np.concatenate([np.asarray(t['noise'])[:, 0] for t in tasks]))),
        'gp_task_mean_diagonal_variance_without_observation_noise': float(np.trace(v1)/len(y)-np.mean(np.concatenate([np.asarray(t['noise'])[:, 0] for t in tasks]))),
        'two_by_two_total_nll_interaction': states['gp_common_gp_task'][0]-states['gp_common_native_task'][0]-states['native_common_gp_task'][0]+value,
        'native_and_coherent_gp_endpoint_moments_verified': True}
    return rows, geometry


def run():
    started = time.process_time(); spec = read(EXECUTION); compression = read(spec['compression_execution'])
    centering = read(compression['centering_execution']); coverage = read(centering['coverage_execution'])
    profile = read(centering['profile_execution']); full = read(compression['native_full_execution'])
    compressed = read(spec['compression_laws'])['rows']; rows, geometry = [], []
    for layout in spec['layouts']:
        for deleted in [None]+spec['training_tasks']:
            suffix = Path(f'layout{layout}') if deleted is None else Path(f'layout{layout}')/f'task{deleted}'
            folder = Path(coverage['output'])/suffix
            means = read(Path(full['output'] if deleted is None else profile['output'])/suffix/'selected_source_fit.json')
            gp = read(folder/('source_fit.json' if deleted is None else 'gp_source_fit.json'))['heads'][0]['joint_gp']
            law = next(x for x in compressed if x['layout'] == layout and x['deleted_task'] == deleted and x['scope'] == 'full_dictionary')
            records, comparison = diagnose(read(folder/'inputs.json'), means['heads'][0], gp, law, spec)
            rows += [{'layout': layout, 'deleted_task': deleted, **r} for r in records]
            geometry.append({'layout': layout, 'deleted_task': deleted, **comparison})
        print('coherent source covariance attribution complete', layout, flush=True)
    fields = ('total_source_nll_change_vs_native', 'common_mean_bank_rms_gap_to_coherent_gp',
        'signed_fraction_of_coherent_gp_native_mean_difference')
    summary = {layout: {branch: {key: float(np.mean([r[key] for r in rows if r['layout'] == layout and r['branch'] == branch]))
        for key in fields} for branch in spec['branches']} for layout in spec['layouts']}
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    result = {'execution': spec, 'summary': summary, 'rows': rows, 'geometry': geometry,
        'all14_endpoint_reconstructions_verified': len(geometry) == 14, 'no_model_fitted_selected_or_adopted': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started,
        'query_paid_outcomes_read': False, 'source_holdouts_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result); print({'summary': summary, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
