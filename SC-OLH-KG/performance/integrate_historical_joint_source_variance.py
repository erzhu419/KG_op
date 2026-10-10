"""Rao-Blackwellize the unresolved joint risk direction using saved states."""

import signal
import time

import numpy as np

from performance.resolve_historical_source_variance import (
    ROOT, IDENTITY, VarianceTiltedPosterior, original_posterior_summary,
    retained_weighted_mixture, common_moments, read, save, cpu_timeout,
    variance_reference,
)
from performance.diagnose_legacy_f_inner_conditional_calibration import integrate

PLAN = 'performance/manifests/historical_source_variance_resolution_v3.json'


def integrate_risk_lines(model, states, reference_mean, reference_covariance, site, coordinate_bound=32.):
    """Expectations under p*h along a q-independent Gaussian coordinate.

    States are unweighted draws of p*h. For each residual r, return
    E_{p*h}(T/h | r). Their ratio to E_{p*h}(1/h | r) restores p.
    """
    sd = np.sqrt(reference_covariance[site, site])
    direction = reference_covariance[:, site] / sd
    coefficient = (states[:, site] - reference_mean[site]) / sd
    origins = states - coefficient[:, None] * direction
    baseline = np.asarray([model.evaluate(p)[0] for p in states]); cache = {}
    n = len(model.mean)

    def at(t, index=slice(None)):
        origin = np.atleast_2d(origins[index]); point = origin + np.asarray(t)[..., None] * direction
        evaluation = [model.evaluate(p) for p in point]
        logs = np.asarray([e[0] for e in evaluation]) - baseline[index]
        result = logs - .5 * (np.log(2 * np.pi) + np.asarray(t) ** 2)
        if isinstance(index, slice):
            moments = [model.moments(e) for e in evaluation]
            ds = np.asarray([m[0] for m in moments]); cs = np.asarray([m[1] for m in moments])
            inverse = np.exp(-np.asarray([model.log_tilt(e) for e in evaluation]))
            second = cs + ds[:, :, None] * ds[:, None, :]
            eta_second = point[:, :, None] * point[:, None, :]
            cache['statistics'] = np.column_stack([np.ones(len(point)), inverse,
                inverse[:, None] * ds, inverse[:, None] * second.reshape(-1, n*n),
                inverse[:, None] * point, inverse[:, None] * eta_second.reshape(-1, n*n)])
        return result if isinstance(index, slice) else float(result[0])

    def statistics(t): return cache['statistics']

    values, logs, error = integrate(at, statistics, coordinate_bound=coordinate_bound)
    # The joint direction has an actual long positive tail beyond the old
    # [-12,12] interval. Check its boundary against the normalizer.
    endpoint = max(float(np.max(at(t) - logs)) for t in (-coordinate_bound, coordinate_bound))
    assert endpoint < -23.025850929940457  # endpoint density < 1e-10
    weights = values[:, 1]; j = 2
    ds = values[:, j:j+n] / weights[:, None]; j += n
    second = values[:, j:j+n*n].reshape(-1, n, n) / weights[:, None, None]; j += n*n
    cs = second - ds[:, :, None] * ds[:, None, :]
    eta = values[:, j:j+n] / weights[:, None]; j += n
    eta_second = values[:, j:j+n*n].reshape(-1, n, n) / weights[:, None, None]
    eta_covariance = eta_second - eta[:, :, None] * eta[:, None, :]
    return eta, ds, cs, weights, eta_covariance, error, endpoint


def run(plan=PLAN):
    spec = read(plan); out = ROOT / spec['output']; result_path = out / 'source_results.json'
    if result_path.exists(): return read(str(result_path.relative_to(ROOT)))
    (out / 'source_cases').mkdir(parents=True, exist_ok=True)
    previous = read(spec['previous_source_registry']); rows = [dict(r) for r in previous['rows']]
    laws = read(spec['source_laws'])['rows']; checks = []; started = time.process_time()
    handler = signal.signal(signal.SIGPROF, cpu_timeout)
    signal.setitimer(signal.ITIMER_PROF, spec['maximum_single_thread_cpu_seconds']); stopped = False
    try:
        for i, case in enumerate(previous['rows']):
            if case['source_posterior_usable']: continue
            saved = read(case['component_file']); law = next(r for r in laws if r['head'] == case['head']
                and all(r[k] == case[k] for k in IDENTITY))
            risk = next(r for r in laws if r['head'] == 'eta_f' and all(r[k] == case[k] for k in IDENTITY))
            old_plan = read(saved['execution_plan']); parent = read(old_plan['covariance_plan'])
            source = next(r for r in read(parent['source_fields'])['rows'] if r['layout'] == case['layout']
                          and r['active_arm'] == parent['proposal_arm'])
            model = VarianceTiltedPosterior(law['source_mean'], law['source_covariance'], source['batch_means'],
                source['sample_variances'], saved['tilt']['sites'], saved['tilt']['center'], saved['tilt']['scale'])
            mean, covariance = variance_reference(risk['source_mean'], risk['source_covariance'], model.s2)
            states = np.asarray(saved['positive_eta_mixture']['eta_states_by_chain']); shape = states.shape
            conditional = []; errors = []; endpoints = []; flat = states.reshape(-1, 48)
            for begin in range(0, len(flat), spec['integration_batch_size']):
                batch = integrate_risk_lines(model, flat[begin:begin+spec['integration_batch_size']],
                    mean, covariance, spec['integrated_source_site'], spec['gaussian_coordinate_bound'])
                conditional.append(batch[:5]); errors.append(batch[5]); endpoints.append(batch[6])
                print('joint conditional source integral', begin+len(batch[0]), '/', len(flat), flush=True)
            eta, ds, cs, weights, eta_cov = [np.concatenate([part[j] for part in conditional]) for j in range(5)]
            eta = eta.reshape(shape); ds = ds.reshape(shape); cs = cs.reshape(*shape[:2], 48, 48)
            weights = weights.reshape(shape[:2]); eta_cov = eta_cov.reshape(*shape[:2], 48, 48)
            posterior = original_posterior_summary(model, eta, ds, cs, weights, spec)
            ec = np.asarray(posterior['eta_posterior_covariance']) + np.einsum(
                'ct,ctij->ij', weights / weights.sum(), eta_cov)
            posterior['eta_posterior_covariance'] = ec.tolist()
            raw = [[model.moments(model.evaluate(p)) for p in chain] for chain in states]
            raw_ds = np.asarray([[m[0] for m in chain] for chain in raw])
            raw_cs = np.asarray([[m[1] for m in chain] for chain in raw])
            inverse = model.inverse_weights(raw_ds, raw_cs)
            mixture = retained_weighted_mixture(model, states, raw_ds, raw_cs, inverse, posterior, spec)
            from performance.recondition_historical_source_laws import source_case
            field = source_case(law, risk, source)
            cm, cc = common_moments(field, 0, np.asarray(posterior['posterior_mean']),
                                   np.asarray(posterior['posterior_covariance']))
            path = out / 'source_cases' / ('f_layout' + '_'.join(case[k] for k in IDENTITY) + '.json')
            row = {**case, **{k: posterior[k] for k in ('monte_carlo_gate_passed', 'maximum_split_rhat',
                'minimum_autocorrelation_ess', 'maximum_first_second_half_f_variance_relative_difference')},
                'joint_direction_conditionally_integrated': True, 'mixture_gate_passed': mixture['mixture_gate_passed'],
                'source_posterior_usable': bool(posterior['monte_carlo_gate_passed'] and mixture['mixture_gate_passed']),
                'component_file': str(path.relative_to(ROOT)), 'previous_component_file': case['component_file']}
            save(path, {**saved, 'execution_plan': plan, 'gates': row, 'positive_eta_mixture': mixture,
                'common_coefficient_posterior_mean': cm.tolist(), 'common_coefficient_posterior_covariance': cc.tolist(),
                'source_f_posterior_mean': posterior['posterior_mean'],
                'source_f_posterior_variance': np.diag(posterior['posterior_covariance']).tolist(),
                'full_source_eta_posterior_mean': posterior['eta_posterior_mean'],
                'full_source_eta_posterior_covariance': posterior['eta_posterior_covariance'],
                **{k: posterior[k] for k in ('diagnostics', 'diagnostic_coordinate_order', 'first_half_posterior_mean',
                    'second_half_posterior_mean', 'first_half_posterior_variance', 'second_half_posterior_variance',
                    'normalized_inverse_weight_ESS')},
                'conditional_integration': {'joint_source_site': spec['integrated_source_site'],
                    'unweighted_tilted_states_used': len(flat), 'maximum_relative_quadrature_error_bound': max(errors),
                    'gaussian_coordinate_bound': spec['gaussian_coordinate_bound'],
                    'maximum_endpoint_log_density_relative_to_normalizer': max(endpoints),
                    'source_posterior_weights_use_conditional_inverse_tilt_expectation': True,
                    'retained_eta_component_weights_still_use_pointwise_inverse_tilt': True}})
            rows[i] = row; checks.append({**{k: case[k] for k in IDENTITY}, 'head': case['head'],
                'maximum_relative_quadrature_error_bound': max(errors), 'maximum_endpoint_log_density': max(endpoints)})
            print('joint-integrated source usable', row['source_posterior_usable'], 'ESS', row['minimum_autocorrelation_ess'],
                'halfvar', row['maximum_first_second_half_f_variance_relative_difference'], 'retention',
                mixture['maximum_source_f_diagonal_variance_relative_difference_vs_all_draws'], flush=True)
    except TimeoutError: stopped = True
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0.); signal.signal(signal.SIGPROF, handler)
    save(out / 'source_verification.json', {'execution_plan': plan, 'checks': checks})
    result = {'execution_plan': plan, 'rows': rows, 'usable_cases': sum(r['source_posterior_usable'] for r in rows),
        'reused_usable_cases': sum(r['source_posterior_usable'] for r in previous['rows']),
        'integrated_cases_completed': len(checks), 'all_source_posteriors_usable': all(r['source_posterior_usable'] for r in rows),
        'cpu_seconds': time.process_time()-started, 'new_source_draws': 0, 'stopped_at_cpu_budget': stopped,
        'original_population_prior_and_raw_likelihood_unchanged': True, 'cost': spec['cost'],
        'no_model_adopted': True, 'independent_confirmation': False}
    save(result_path, result)
    return result


if __name__ == '__main__': run()
