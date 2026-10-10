"""Positive composite quadrature for the frozen conditional source posterior."""

import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import roots_legendre

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.check_legacy_public_mean_risk_uncertainty import read
from performance.run_legacy_related_inventory_source import save

EXECUTION = 'performance/manifests/legacy_joint_population_f_task_level_reference_quadrature_execution_v4.json'


def posterior_density_u(u, law):
    scale = law['source_signal']; u = np.asarray(u); v = scale*(1-u**2)/u**2
    eigenvalues, rhs = np.asarray(law['task_variance_eigenvalues']), np.asarray(law['task_variance_rhs'])
    x = v[..., None]*eigenvalues
    difference = .5*np.sum(np.log1p(x)-v[..., None]/(1+x)*rhs**2, axis=-1)
    native = law['native_task_intercept_variance']; xn = native*eigenvalues
    native_difference = .5*np.sum(np.log1p(xn)-native/(1+xn)*rhs**2)
    information = .5*np.sum((eigenvalues/(1+x))**2, axis=-1)
    return np.exp(-difference+native_difference+.5*np.log(information)+np.log(2*scale)-3*np.log(u)
        -law['reference_log_normalizer_relative_to_native_nll'])


def positive_base_rule(law, reference, spec):
    moments = reference['reference_posterior']; mean = moments['mean']; second = mean**2+moments['standard_deviation']**2
    eigenvalues = np.asarray(law['task_variance_eigenvalues']); signal = law['source_signal']
    def integrand(u):
        if u == 0.:
            rhs = np.asarray(law['task_variance_rhs']); size = len(eigenvalues)
            native = law['native_task_intercept_variance']; xn = native*eigenvalues
            native_difference = .5*np.sum(np.log1p(xn)-native/(1+xn)*rhs**2)
            log_tail = (native_difference+.5*np.sum(rhs**2/eigenvalues-np.log(eigenvalues))
                +.5*np.log(size/2)-law['reference_log_normalizer_relative_to_native_nll'])
            endpoint_second = np.exp(np.log(2)+(2-size/2)*np.log(signal)+log_tail)/second if size == 5 else 0.
            return np.r_[0., 0., endpoint_second, np.zeros(size+size**2)]
        v = signal*(1-u**2)/u**2; x = v*eigenvalues; shrinkage = x/(1+x)
        return posterior_density_u(u, law)*np.r_[1., v/mean, v**2/second, shrinkage, np.outer(shrinkage, shrinkage).ravel()]
    points = [np.sqrt(signal/(signal+v)) for v in (law['native_task_intercept_variance'], reference['stage105_slice_upper_variance']) if v > 0]
    tolerance = spec['quadrature']
    integral, error, info = quad_vec(integrand, 0., 1., points=points, norm='max', full_output=True, workers=1,
        epsabs=tolerance['absolute_tolerance_for_scaled_integrals'], epsrel=tolerance['relative_tolerance'],
        limit=tolerance['maximum_subintervals'])
    if not info.success:
        raise RuntimeError(f'Positive source quadrature partition failed: {info.message}')
    assert error <= max(tolerance['absolute_tolerance_for_scaled_integrals'], tolerance['relative_tolerance']*np.max(np.abs(integral)))
    intervals = sorted(info.intervals.tolist())
    nodes, weights = rule_on_intervals(law, intervals, 16)
    return nodes, weights, float(error), intervals


def rule_on_intervals(law, intervals, size):
    signal = law['source_signal']; x, w = roots_legendre(size); nodes, weights = [], []
    for lo, hi in intervals:
        u = .5*(lo+hi)+.5*(hi-lo)*x
        nodes.extend((signal*(1-u**2)/u**2).tolist()); weights.extend((.5*(hi-lo)*w*posterior_density_u(u, law)).tolist())
    return np.asarray(nodes), np.asarray(weights)


def approximation_errors(variances, weights, law, reference):
    mass = weights.sum(); normalized = weights/mass
    eigenvalues, rhs = np.asarray(law['task_variance_eigenvalues']), np.asarray(law['task_variance_rhs'])
    update = np.asarray(law['common_coefficient_update_columns']); beta0 = np.asarray(law['coefficient_mean_at_zero_task_intercept_variance'])
    h0 = np.asarray(law['common_covariance_at_zero_task_intercept_variance'])
    conditional_weights = variances[:, None]/(1+variances[:, None]*eigenvalues)
    beta = beta0-(conditional_weights*rhs)@update.T; mean = normalized@beta
    displacement = beta-mean
    covariance = h0+(update*(normalized@conditional_weights))@update.T+(displacement.T*normalized)@displacement
    expected_mean = np.asarray(law['integrated_common_coefficient_mean']); expected_covariance = np.asarray(law['integrated_common_coefficient_covariance'])
    moment = reference['reference_posterior']; expected_second = moment['mean']**2+moment['standard_deviation']**2
    return {'raw_mass_absolute_error': float(abs(mass-1)), 'mean_variance_relative_error': float(abs(normalized@variances/moment['mean']-1)),
        'coefficient_mean_max_absolute_error': float(np.max(np.abs(mean-expected_mean))),
        'common_covariance_relative_frobenius_error': float(np.linalg.norm(covariance-expected_covariance)/np.linalg.norm(expected_covariance)),
        'second_moment_absolute_error': float(abs(normalized@(variances**2)-expected_second))}


def passes(errors, law, reference, spec):
    t = spec['tolerances']; m = reference['reference_posterior']; second = m['mean']**2+m['standard_deviation']**2
    return (errors['raw_mass_absolute_error'] <= t['raw_mass_absolute']
        and errors['mean_variance_relative_error'] <= t['mean_variance_relative']
        and errors['coefficient_mean_max_absolute_error'] <= t['coefficient_mean_max_absolute']
        and errors['common_covariance_relative_frobenius_error'] <= t['common_covariance_relative_frobenius']
        and errors['second_moment_absolute_error'] <= m['normalized_vector_integral_error']*law['source_signal']**2+t['second_moment_extra_relative']*second)


def run():
    started = time.process_time(); spec = read(EXECUTION); reference = read(spec['reference_results'])
    assert reference['all14_source_integrations_verified']
    source = read(spec['reference_laws']); laws, rows = [], []
    for law in source['rows']:
        layout, deleted = law['layout'], law['deleted_task']
        r = next(x for x in reference['rows'] if x['layout'] == layout and x['deleted_task'] == deleted)
        variances, weights, error, intervals = positive_base_rule(law, r, spec)
        base_errors = approximation_errors(variances, weights, law, r); base_passed = passes(base_errors, law, r, spec)
        trials = []; selected = None
        if base_passed:
            for size in spec['nodes_per_subinterval_resolutions']:
                nodes, probability = rule_on_intervals(law, intervals, size)
                errors = approximation_errors(nodes, probability, law, r); passed = passes(errors, law, r, spec)
                trials.append({'node_count': len(nodes), 'nodes_per_subinterval': size, 'errors': errors, 'passed': passed})
                if passed:
                    selected = {'layout': layout, 'deleted_task': deleted, 'variance_nodes': nodes.tolist(),
                        'posterior_weights': (probability/probability.sum()).tolist(), 'source_reference_law': law,
                        'base_variance_nodes': variances.tolist(), 'base_posterior_weights': weights.tolist(),
                        'adaptive_u_intervals': intervals, 'nodes_per_subinterval': size}
                    laws.append(selected); break
        row = {'layout': layout, 'deleted_task': deleted, 'base_node_count': len(variances), 'adaptive_interval_count': len(intervals),
            'adaptive_vector_error': error, 'base_errors': base_errors, 'base_rule_passed': base_passed, 'composite_trials': trials,
            'rule_complete': selected is not None, 'selected_node_count': len(selected['variance_nodes']) if selected else None}
        rows.append(row); print('source variance quadrature', layout, deleted, len(variances), row['selected_node_count'], flush=True)
    out = ROOT/spec['output']; out.mkdir(parents=True, exist_ok=True)
    complete = len(laws) == 14
    save(out/'source_quadrature.json', {'execution': spec, 'all14_rules_frozen_without_query_data': complete, 'rows': laws})
    result = {'execution': spec, 'rows': rows, 'all14_source_rules_complete': complete, 'no_model_adopted': True,
        'cost': spec['cost'], 'cpu_seconds': time.process_time()-started, 'query_paid_outcomes_read': False,
        'source6_7_evaluated': False, 'target_promoted': False, 'independent_confirmation': False}
    save(out/'results.json', result)
    print({'all14_source_rules_complete': complete, 'cpu_seconds': result['cpu_seconds']}, flush=True)


if __name__ == '__main__':
    run()
