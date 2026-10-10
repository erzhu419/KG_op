"""Joint raw-batch source update under frozen correlated Gaussian head laws."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_diverse_source_design import encode_diverse_source
from performance.check_legacy_f_source_spectral_uncertainty import source_model
from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.diagnose_legacy_f_source_latent_noise_feasibility import ARMS
from performance.diagnose_legacy_f_sparse_task_template import source_paths
from performance.diagnose_legacy_risk_transfer import no_outcomes
from performance.run_legacy_related_inventory_source import problem_for, save
from representation.legacy_observable_prior import legacy_exposure_descriptors
from representation.orthogonal_function_prior import OrthogonalLowOrderBasis
from representation.physical_risk_prior import log_sample_variance_density
from representation.source_kernel_population import rbf_cross, shared_kernel

PLAN = 'performance/manifests/legacy_joint_population_f_correlated_source_raw_update_plan_v1.json'


def conditional(mean, covariance, labels, noise):
    factor = cho_factor(covariance + np.diag(noise), lower=True)
    gain = cho_solve(factor, covariance).T
    return mean + gain @ (labels - mean), covariance - gain @ covariance, gain


def diagnostics(draws):
    """Split Rhat and Geyer positive/monotone-pair multichain autocorrelation ESS."""
    chains, count, dimensions = draws.shape
    split = draws.reshape(chains * 2, count // 2, dimensions); n = split.shape[1]
    centered = split - split.mean(axis=1, keepdims=True)
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = n * split.mean(axis=1).var(axis=0, ddof=1)
    variance = (n - 1) / n * within + between / n
    transformed = np.fft.rfft(centered, n=2 * n, axis=1)
    acov = np.fft.irfft(transformed * transformed.conj(), n=2 * n, axis=1)[:, :n].real / n
    rho = 1 - (within[None, :] - acov.mean(axis=0)) / variance[None, :]
    rho[0] = 1.
    pairs = rho[::2] + rho[1::2]
    pairs = np.minimum.accumulate(pairs, axis=0)
    positive = np.logical_and.accumulate(pairs > 0, axis=0)
    tau = np.maximum(-1 + 2 * np.sum(np.where(positive, pairs, 0), axis=0), 1.)
    return {'split_rhat': np.sqrt(variance / within).tolist(),
        'autocorrelation_ess': (chains * count / tau).tolist()}


def eta_log_likelihood(eta, f, labels, s2):
    return float(-.5 * np.sum(3 * eta + (2 * s2 + 3 * (labels - f) ** 2) * np.exp(-eta)))


def elliptical_update(eta, mean, factor, f, labels, s2, rng):
    residual = eta - mean; direction = factor @ rng.normal(size=len(eta))
    threshold = eta_log_likelihood(eta, f, labels, s2) + np.log(rng.uniform())
    angle = rng.uniform(0, 2 * np.pi); lower, upper = angle - 2 * np.pi, angle; evaluations = 0
    while True:
        proposal = mean + residual * np.cos(angle) + direction * np.sin(angle); evaluations += 1
        if eta_log_likelihood(proposal, f, labels, s2) >= threshold:
            return proposal, evaluations
        if angle < 0: lower = angle
        else: upper = angle
        angle = rng.uniform(lower, upper)


def freeze(spec, known_spec, known_fields):
    inc = read(read(known_spec['incremental_source_results'])['execution']['source_probe_plan'])
    registry = read(inc['source_registry'])['rows']; rosters = read(known_spec['source_rosters'])['rows']
    coverage, _ = source_paths(read(known_spec['source_context_execution']))
    public = read(Path(coverage['source_data']).with_name('frozen_inputs.json')); bank = [tuple(p) for p in public['bank']]
    raw = read(coverage['source_data']); data = {'bank': raw['bank'], 'tasks': {i: raw['tasks'][i] for i in spec['source_tasks']}}
    del raw
    descriptors = {i: legacy_exposure_descriptors(no_outcomes(problem_for(public['plan'], public['tasks'][i])), bank, 50)
        for i in spec['source_tasks']}
    risk_execution = read(known_spec['risk_source_execution']); fields = []
    old = np.concatenate([np.arange(24 * t, 24 * t + 12) for t in range(4)])
    new = np.concatenate([np.arange(24 * t + 12, 24 * t + 24) for t in range(4)])
    for layout in spec['layouts']:
        roster = next(r for r in rosters if r['layout'] == layout)
        reference = next(r for r in known_fields if r['layout'] == layout)
        model, law, psi = source_model(inc, layout, 0)
        ds = [np.column_stack([np.ones(24), model.basis.transform(psi[sites])]) for sites in roster['policy_indices_by_source']]
        d = np.vstack(ds); a = np.asarray(law['common_prior_coefficient_covariance']); r = np.asarray(law['task_coefficient_covariance'])
        structure = d @ a @ d.T + block_diag(*[dt @ r @ dt.T + model.white * np.eye(24) for dt in ds])
        gp = read(next(r['source_fit'] for r in registry if (r['layout'], r['inner_query_task']) ==
            (layout, 'related_inventory_00')))['fits']['gp']
        anchor = gp['anchor']; scale2 = anchor['normalization']['scale'] ** 2
        amplitude, intercept, white = gp['residual_parameters']; xs = [psi[sites] for sites in roster['policy_indices_by_source']]
        all_x = np.vstack(xs)
        generic = scale2 * shared_kernel(all_x, all_x, anchor) + block_diag(*[
            scale2 * (amplitude * rbf_cross(x, x, anchor['lengthscale']) + intercept + white * np.eye(24)) for x in xs])
        values = np.concatenate([np.asarray([[v[0] for v in data['tasks'][i]['Y_replicates'][j]] for j in sites])
            for i, sites in zip(spec['source_tasks'], roster['policy_indices_by_source'])])
        labels, s2 = values.mean(axis=1), values.var(axis=1, ddof=1)
        center = anchor['normalization']['mean'] + anchor['normalization']['scale'] * anchor['constant']
        f_means, f_covariances = [], []
        for covariance in (structure, generic):
            factor = cho_factor(covariance[np.ix_(old, old)] + np.diag(s2[old] / 3), lower=True)
            cross = covariance[np.ix_(new, old)]
            f_means.append(center + cross @ cho_solve(factor, labels[old] - center))
            cov = covariance[np.ix_(new, new)] - cross @ cho_solve(factor, cross.T)
            f_covariances.append(.5 * (cov + cov.T))
        case = next(c for c in risk_execution['cases'] if (c['layout'], c['query_task']) == (layout, 0))
        folder = Path(risk_execution['output']) / case['folder']; inputs = read(folder / 'risk_inputs.json')
        risk = read(folder / 'risk_source_fit.json')['fits']['joint_structural']
        coordinate, tasks, _, _ = encode_diverse_source(data, spec['source_tasks'], inputs['source_site_indices_by_task'], descriptors,
            public['plan']['task_distribution']['L'])
        public_risk = coordinate.transform(descriptors[spec['source_tasks'][0]])
        np.testing.assert_allclose(public_risk, inputs['public_reference_psi'], atol=1e-8, rtol=1e-8)
        basis = OrthogonalLowOrderBasis(public_risk); h = np.asarray(risk['population_mean_covariance'])
        r = np.diag(risk['residual_coefficient_diagonal']); white = risk['discrepancy_variance']
        adjusted, local = [], []
        for source, task, sites in zip(spec['source_tasks'], tasks, roster['policy_indices_by_source']):
            dt = np.column_stack([np.ones(8), basis.transform(task['psi'])])
            dq = np.column_stack([np.ones(12), basis.transform(coordinate.transform(descriptors[source][sites[12:]]))])
            factor = cho_factor(dt @ r @ dt.T + np.diag(task['noise'][:, 3] + white), lower=True)
            b = cho_solve(factor, dt @ r @ dq.T).T
            adjusted.append(dq - b @ dt); local.append(dq @ r @ dq.T - b @ dt @ r @ dq.T + white * np.eye(12))
        adjusted = np.vstack(adjusted); eta_cov = adjusted @ h @ adjusted.T + block_diag(*local)
        eta_cov = .5 * (eta_cov + eta_cov.T)
        np.testing.assert_allclose(np.column_stack(f_means), reference['f_means'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(np.column_stack([np.diag(c) for c in f_covariances]), reference['f_variances'], atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(np.diag(eta_cov), reference['eta_variance'], atol=1e-8, rtol=1e-8)
        for covariance in [*f_covariances, eta_cov]: np.linalg.cholesky(covariance)
        fields.append({'layout': layout, 'sites': reference['sites'], 'arms': ARMS,
            'f_means': [m.tolist() for m in f_means], 'f_covariances': [c.tolist() for c in f_covariances],
            'eta_mean': reference['eta_mean'], 'eta_covariance': eta_cov.tolist(),
            'added_means': labels[new].tolist(), 'added_sample_variances': s2[new].tolist()})
    return fields


def sample(field, index, spec, started):
    mean = np.asarray(field['f_means'][index]); covariance = np.asarray(field['f_covariances'][index]); lf = np.linalg.cholesky(covariance)
    em = np.asarray(field['eta_mean']); le = np.linalg.cholesky(field['eta_covariance'])
    labels, s2 = np.asarray(field['added_means']), np.asarray(field['added_sample_variances'])
    plugin_mean, plugin_cov, gain = conditional(mean, covariance, labels, s2 / 3)
    identity = (np.eye(48) - gain) @ covariance @ (np.eye(48) - gain).T + (gain * (s2 / 3)) @ gain.T
    np.testing.assert_allclose(identity, plugin_cov, atol=1e-10, rtol=1e-8)
    # Check log-likelihood differences,which remove the observation-measure constants.
    probe = mean; shifted = em + .1
    def reference(eta):
        variance = np.exp(eta) / 3
        return float(np.sum(-.5 * (np.log(2 * np.pi * variance) + (labels - probe) ** 2 / variance)
            + log_sample_variance_density(eta, s2, 3)))
    difference_error = abs((eta_log_likelihood(shifted, probe, labels, s2) - eta_log_likelihood(em, probe, labels, s2))
        - (reference(shifted) - reference(em)))
    assert difference_error <= 1e-8
    sampler = spec['sampler']; count = sampler['retained_sweeps_per_chain']; chains = sampler['chains']
    f_draws, eta_draws, evaluations = [], [], []
    for chain in range(chains):
        rng = np.random.default_rng(sampler['seed'] + 100 * int(field['layout']) + 10 * index + chain)
        eta = em + le @ rng.normal(size=48); fs, es = [], []; evaluations_chain = 0
        for iteration in range(sampler['warmup_sweeps_per_chain'] + count):
            assert time.process_time() - started < spec['maximum_single_thread_cpu_seconds']
            noise = np.exp(eta) / 3; prior_draw = mean + lf @ rng.normal(size=48)
            factor = cho_factor(covariance + np.diag(noise), lower=True)
            f = prior_draw + covariance @ cho_solve(factor, labels - prior_draw - np.sqrt(noise) * rng.normal(size=48))
            eta, n = elliptical_update(eta, em, le, f, labels, s2, rng); evaluations_chain += n
            if iteration >= sampler['warmup_sweeps_per_chain']: fs.append(f); es.append(eta)
        f_draws.append(fs); eta_draws.append(es); evaluations.append(evaluations_chain)
    f_draws, eta_draws = np.asarray(f_draws), np.asarray(eta_draws)
    diagnostic_f, diagnostic_eta = diagnostics(f_draws), diagnostics(eta_draws)
    first = f_draws[:, :count // 2].reshape(-1, 48).var(axis=0, ddof=1)
    second = f_draws[:, count // 2:].reshape(-1, 48).var(axis=0, ddof=1)
    half_difference = float(np.max(np.abs(first - second) / ((first + second) / 2)))
    max_rhat = max(max(diagnostic_f['split_rhat']), max(diagnostic_eta['split_rhat']))
    min_ess = min(min(diagnostic_f['autocorrelation_ess']), min(diagnostic_eta['autocorrelation_ess']))
    gate = spec['monte_carlo_gate']; passed = (max_rhat <= gate['maximum_split_rhat_all_f_and_eta_coordinates']
        and min_ess >= gate['minimum_autocorrelation_ess_all_f_and_eta_coordinates']
        and half_difference <= gate['maximum_first_second_half_variance_relative_difference'])
    flat = f_draws.reshape(-1, 48); posterior_mean = flat.mean(axis=0); posterior_cov = np.cov(flat, rowvar=False)
    ratio = np.diag(posterior_cov) / np.diag(plugin_cov)
    return {'layout': field['layout'], 'arm': ARMS[index], 'monte_carlo_gate_passed': passed,
        'maximum_split_rhat': max_rhat, 'minimum_autocorrelation_ess': min_ess,
        'maximum_first_second_half_f_variance_relative_difference': half_difference,
        'diagnostics_f': diagnostic_f, 'diagnostics_eta': diagnostic_eta,
        'posterior_mean': posterior_mean.tolist(), 'posterior_covariance': posterior_cov.tolist(),
        'plug_in_posterior_mean': plugin_mean.tolist(), 'plug_in_posterior_covariance': plugin_cov.tolist(),
        'posterior_variance_ratio_mean': float(ratio.mean()), 'posterior_variance_ratio_quantiles_0_50_100': np.quantile(ratio, [0, .5, 1]).tolist(),
        'posterior_mean_shift_rms_in_plugin_posterior_sd': float(np.sqrt(np.mean((posterior_mean - plugin_mean) ** 2 / np.diag(plugin_cov)))),
        'elliptical_likelihood_evaluations_by_chain': evaluations,
        'matheron_conditional_covariance_identity_max_error': float(np.max(np.abs(identity - plugin_cov))),
        'eta_conditional_likelihood_difference_identity_error': difference_error}


def run():
    started = time.process_time(); spec = read(PLAN); known = read(spec['known_source_results'])
    assert known['diagnostic_interpreted']; known_spec = read(spec['known_source_plan'])
    known_fields = read(Path(known_spec['output']) / 'frozen_inputs.json')['rows']
    fields = freeze(spec, known_spec, known_fields); out = ROOT / spec['output']; out.mkdir(parents=True, exist_ok=True)
    save(out / 'frozen_inputs.json', {'execution': spec, 'rows': fields, 'marginal_reproduction_and_positive_covariance_checks_passed': True,
        'original_data_only_priors_no_added_variance_conditioning': True})
    rows = []
    for field in fields:
        for index in range(2):
            row = sample(field, index, spec, started); rows.append(row)
            save(out / 'results.json', {'execution': spec, 'rows': rows, 'diagnostic_interpreted': False,
                'all_four_conditionals_complete': len(rows) == 4, 'cpu_seconds': time.process_time() - started})
            print('correlated raw source', field['layout'], ARMS[index], 'gate', row['monte_carlo_gate_passed'],
                'Rhat', row['maximum_split_rhat'], 'ESS', row['minimum_autocorrelation_ess'],
                'half variance', row['maximum_first_second_half_f_variance_relative_difference'],
                'posterior variance ratio', row['posterior_variance_ratio_mean'], flush=True)
    passed = all(r['monte_carlo_gate_passed'] for r in rows)
    save(out / 'results.json', {'execution': spec, 'rows': rows, 'all_four_conditionals_complete': True,
        'all_monte_carlo_gates_passed': passed, 'diagnostic_interpreted': passed, 'cpu_seconds': time.process_time() - started,
        'cost': {**spec['cost'], 'actual_original_source_coordinate_restorations': 2,
            'actual_chains': 16, 'actual_sweeps': 16 * (spec['sampler']['warmup_sweeps_per_chain'] + spec['sampler']['retained_sweeps_per_chain'])},
        'validation_task0_or_task1_outcomes_indexed': False, 'source6_7_evaluated': False,
        'no_model_adopted': True, 'independent_confirmation': False,
        'interpretation': 'Joint48-batch conditional posterior prototype retaining full source/task covariances and the variance likelihood once. Original source head/coordinate Gaussian approximations remain. Numerical posterior summaries require all registered Monte Carlo gates;no model selection or held-task forecast.'})


if __name__ == '__main__':
    run()
