"""Known-source batch likelihood with a shared source-conditioned risk law."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve
from scipy.special import digamma, ndtr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_latent_noise_feasibility import ARMS, conditional_pair
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.source_kernel_population import rbf_cross, shared_kernel

PLAN = 'performance/manifests/legacy_joint_population_f_known_source_latent_noise_plan_v1.json'


def known_structural(beta, h, r, white, dt, y, noise, dq):
    vt = dt @ r @ dt.T + np.diag(noise + white)
    b = cho_solve(cho_factor(vt, lower=True), dt @ r @ dq.T).T
    adjusted = dq - b @ dt
    mean = dq @ beta + b @ (y - dt @ beta)
    covariance = dq @ r @ dq.T - b @ dt @ r @ dq.T + adjusted @ h @ adjusted.T
    covariance += white * np.eye(len(dq))
    return mean, .5 * (covariance + covariance.T)


def direct_mean_check(source_spec, registry, layout, roster, expected_means, expected_variances):
    """Check the first source's reconstructed innovations against old-data solves."""
    model, law, public = source_model(source_spec, layout, 0)
    coverage, _ = source_paths(read(source_spec['source_context_execution']))
    inputs = read(Path(coverage['output']) / f'layout{layout}' / 'task1' / 'inputs.json')
    tasks = [t for t in inputs['source_tasks'] if t['name'] in law['source_task_names']]
    task = tasks[0]; sites = roster['policy_indices_by_source'][0][12:]
    dq = np.column_stack([np.ones(12), model.basis.transform(public[sites])])
    dt = model.designs[0]
    mean, covariance = known_structural(np.asarray(law['source_posterior_coefficient_mean']),
        np.asarray(law['source_posterior_common_covariance']), np.asarray(law['task_coefficient_covariance']),
        model.white, dt, np.asarray(task['response'])[:, 0], np.asarray(task['noise'])[:, 0], dq)
    gp = read(next(r['source_fit'] for r in registry if
        (r['layout'], r['inner_query_task']) == (layout, 'related_inventory_00')))['fits']['gp']
    anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
    center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
    amplitude, intercept, white = gp['residual_parameters']
    xs = np.vstack([t['features'] for t in tasks]); q = public[sites]
    koo = scale2 * shared_kernel(xs, xs, anchor) + block_diag(*[
        scale2 * (amplitude * rbf_cross(np.asarray(t['features']), np.asarray(t['features']), anchor['lengthscale']) + intercept)
        + np.diag(np.asarray(t['noise'])[:, 0] + scale2 * white) for t in tasks])
    cross = scale2 * shared_kernel(q, xs, anchor)
    cross[:, :12] += scale2 * (amplitude * rbf_cross(q, np.asarray(tasks[0]['features']), anchor['lengthscale']) + intercept)
    factor = cho_factor(koo, lower=True)
    gp_mean = center + cross @ cho_solve(factor, np.concatenate([np.asarray(t['response'])[:, 0] for t in tasks]) - center)
    gp_cov = scale2 * (shared_kernel(q, q, anchor) + amplitude * rbf_cross(q, q, anchor['lengthscale'])
        + intercept + white * np.eye(12)) - cross @ cho_solve(factor, cross.T)
    actual_means = np.column_stack([mean, gp_mean]); actual_variances = np.column_stack([np.diag(covariance), np.diag(gp_cov)])
    np.testing.assert_allclose(actual_means, expected_means[:12], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(actual_variances, expected_variances[:12], atol=1e-8, rtol=1e-8)
    return {'layout': layout, 'source_task': roster['source_ids'][0], 'arms': ARMS,
        'mean_max_absolute_error': float(np.max(np.abs(actual_means - expected_means[:12]))),
        'latent_variance_max_absolute_error': float(np.max(np.abs(actual_variances - expected_variances[:12]))), 'passed': True}


def run():
    started = time.process_time(); spec = read(PLAN); incremental = read(spec['incremental_source_results'])
    assert incremental['diagnostic_interpreted'] and not incremental['validation_task0_or_task1_outcomes_indexed']
    rosters = read(spec['source_rosters'])['rows']; inc_spec = incremental['execution']
    coverage, _ = source_paths(read(spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank, 50)
        for i in spec['source_tasks']}
    raw = read(coverage['source_data']); data = {'bank': raw['bank'], 'tasks': {i: raw['tasks'][i] for i in spec['source_tasks']}}
    del raw
    risk_execution = read(spec['risk_source_execution']); fields, checks = [], []
    for layout in spec['layouts']:
        roster = next(r for r in rosters if r['layout'] == layout); assert roster['source_ids'] == spec['source_tasks']
        case = next(c for c in risk_execution['cases'] if (c['layout'], c['query_task']) == (layout, 0))
        folder = Path(risk_execution['output']) / case['folder']; inputs = read(folder / 'risk_inputs.json')
        fits = read(folder / 'risk_source_fit.json'); law = fits['fits']['joint_structural']
        assert fits['head'] == 3 and inputs['training_task_ids'] == spec['source_tasks'] and law['optimizer_converged']
        sites = inputs['source_site_indices_by_task']
        coordinate, risk_tasks, center, scale = encode_diverse_source(data, spec['source_tasks'], sites, descriptors,
            public['plan']['task_distribution']['L'])
        reference = coordinate.transform(descriptors[spec['source_tasks'][0]])
        np.testing.assert_allclose(reference, inputs['public_reference_psi'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(center, inputs['source_feature_center'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(scale, inputs['source_feature_scale'], atol=1e-8, rtol=1e-8)
        basis = OrthogonalLowOrderBasis(reference); beta = np.asarray(law['coefficient_mean'])
        h = np.asarray(law['population_mean_covariance']); r = np.diag(law['residual_coefficient_diagonal']); white = law['discrepancy_variance']
        ds, eta_means, eta_variances, added_sites, observed_means, noises = [], [], [], [], [], []
        queries = []
        for source, original_sites, t, saved, full_sites in zip(spec['source_tasks'], sites, risk_tasks,
                inputs['source_tasks'], roster['policy_indices_by_source']):
            for key in ('psi', 'features', 'response', 'noise'):
                np.testing.assert_allclose(t[key], saved[key], atol=1e-8, rtol=1e-8)
            extra = full_sites[12:]; assert len(extra) == 12 and set(extra).isdisjoint(original_sites)
            dt = np.column_stack([np.ones(8), basis.transform(t['psi'])])
            dq = np.column_stack([np.ones(12), basis.transform(coordinate.transform(descriptors[source][extra]))])
            mean, covariance = known_structural(beta, h, r, white, dt, t['response'][:, 3], t['noise'][:, 3], dq)
            assert np.linalg.eigvalsh(covariance).min() > 0
            values = np.asarray([[v[0] for v in data['tasks'][source]['Y_replicates'][j]] for j in extra])
            assert values.shape == (12, 3)
            ds.append(dt); queries.append(dq); eta_means.extend(mean); eta_variances.extend(np.diag(covariance))
            observed_means.extend(values.mean(axis=1)); noises.extend(values.var(axis=1, ddof=1) / 3)
            added_sites.extend([{'source_task': source, 'policy': j} for j in extra])
        # A dense source solve detects common/task conditioning errors in the risk formula.
        d = np.vstack(ds); a = np.diag(law['common_prior_coefficient_diagonal'])
        m0 = np.asarray(fits['fits']['regularizer']['coefficient_mean']); dq = queries[0]
        koo = d @ a @ d.T + block_diag(*[dt @ r @ dt.T + np.diag(t['noise'][:, 3] + white) for dt, t in zip(ds, risk_tasks)])
        cross = dq @ a @ d.T; cross[:, :8] += dq @ r @ ds[0].T
        factor = cho_factor(koo, lower=True); error = np.concatenate([t['response'][:, 3] for t in risk_tasks]) - d @ m0
        dense_mean = dq @ m0 + cross @ cho_solve(factor, error)
        dense_cov = dq @ (a + r) @ dq.T + white * np.eye(12) - cross @ cho_solve(factor, cross.T)
        np.testing.assert_allclose(dense_mean, eta_means[:12], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(np.diag(dense_cov), eta_variances[:12], atol=1e-8, rtol=1e-8)
        record = next(row for row in incremental['rows'] if row['layout'] == layout)
        observed_means, noises = np.asarray(observed_means), np.asarray(noises)
        f_means = np.column_stack([observed_means - record[arm]['added_source_mean_residuals'] for arm in ('sparse_joint', 'same_source_GP')])
        f_variances = np.column_stack([np.asarray(record[arm]['conditional_variance_diagonal']) - noises for arm in ('sparse_joint', 'same_source_GP')])
        assert np.all(f_variances > 0)
        if layout == spec['layouts'][0]:
            checks.append(direct_mean_check(read(inc_spec['source_probe_plan']), read(inc_spec['source_gp_registry'])['rows'],
                layout, roster, f_means, f_variances))
        checks.append({'layout': layout, 'risk_coordinate_and_original_statistics_reproduced': True,
            'dense_known_source_risk_mean_max_absolute_error': float(np.max(np.abs(dense_mean - eta_means[:12]))),
            'dense_known_source_risk_variance_max_absolute_error': float(np.max(np.abs(np.diag(dense_cov) - eta_variances[:12]))),
            'added_policies_disjoint_from_original_risk_labels': True, 'passed': True})
        fields.append({'layout': layout, 'sites': added_sites, 'f_means': f_means.tolist(), 'f_variances': f_variances.tolist(),
            'eta_mean': eta_means, 'eta_variance': eta_variances})
    out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': fields,
        'original_data_only_conditionals_frozen_before_added_variance_scores': True, 'checks': checks})
    rows = []
    for field in fields:
        for j, site in enumerate(field['sites']):
            assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
            values = np.asarray([v[0] for v in data['tasks'][site['source_task']]['Y_replicates'][site['policy']]])
            label = float(values.mean()); s2 = float(values.var(ddof=1)); eta_label = np.log(s2) - digamma(1.)
            np.testing.assert_allclose(np.exp(eta_label + digamma(1.)) / 3, s2 / 3, atol=1e-12, rtol=1e-12)
            results = conditional_pair(field['f_means'][j], field['f_variances'][j], field['eta_mean'][j], field['eta_variance'][j], label, eta_label)
            for index, result in enumerate(results):
                fv = field['f_variances'][j][index]; total = fv + s2 / 3; error = label - field['f_means'][j][index]
                density = -.5 * (np.log(2 * np.pi * total) + error ** 2 / total)
                tangent = .5 * fv * (error ** 2 / total ** 2 - 1 / total); cdf = float(ndtr(error / np.sqrt(total)))
                result.update(layout=field['layout'], **site, source_plug_in_noise=s2 / 3, plug_in_batch_mean_variance=total,
                    plug_in_mean_density=float(density), conditional_minus_plug_in_mean_density=float(result['conditional_mean_density'] - density),
                    conditional_to_plug_in_variance_ratio=result['conditional_batch_mean_variance'] / total,
                    integrated_to_sample_noise_ratio=result['expected_episode_noise_over_three'] / (s2 / 3),
                    plug_in_log_f_variance_score_gradient=float(tangent), plug_in_covered_90=bool(.05 <= cdf <= .95))
                rows.append(result)
    keys = ('conditional_mean_density', 'plug_in_mean_density', 'conditional_minus_plug_in_mean_density',
        'conditional_to_plug_in_variance_ratio', 'log_conditional_f_variance_score_gradient',
        'plug_in_log_f_variance_score_gradient', 'covered_90', 'plug_in_covered_90')
    summary = {}
    for layout in spec['layouts']:
        summary[layout] = {}
        for arm in ARMS:
            selected = [r for r in rows if (r['layout'], r['arm']) == (layout, arm)]
            summary[layout][arm] = {key: float(np.mean([r[key] for r in selected])) for key in keys}
            summary[layout][arm].update({
                'variance_ratio_quantiles_0_50_100': np.quantile([r['conditional_to_plug_in_variance_ratio'] for r in selected], [0, .5, 1]).tolist(),
                'absolute_log_density_change_median': float(np.median([abs(r['conditional_minus_plug_in_mean_density']) for r in selected])),
                'tangent_direction_changes': sum(r['log_conditional_f_variance_score_gradient'] * r['plug_in_log_f_variance_score_gradient'] < 0 for r in selected)})
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'summary': summary, 'checks': checks,
        'diagnostic_interpreted': True, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_original_source_coordinate_restorations': 2,
            'actual_joint_batch_components_integrated': len(rows), 'actual_shared_variance_integrations': len(rows) // 2},
        'validation_task0_or_task1_outcomes_indexed': False, 'source6_7_evaluated': False,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Single-site known-source conditional batch densities using original f and eta Gaussian laws. Added variance likelihood counted once;source/task risk conditioning retained. Not an exact full source posterior or a joint48-label source likelihood. Original risk training uses its log-variance approximation;no source refit or held-task forecast.'})
    print('known-source latent noise', summary, 'CPU', time.process_time() - started, flush=True)


if __name__ == '__main__':
    run()
