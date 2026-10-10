"""Conditional source spectral integration in the original orthogonal family."""

import numpy as np
from scipy.integrate import quad_vec


def integrate_task_spectrum(likelihood, native, quadrature, scalar_reference):
    """Moment-match beta, common covariance H and task diagonal R over gamma.

    Nuisance source parameters remain fixed. Center beta at the native source
    posterior to avoid subtracting large uncentered second moments.
    """
    parameters = np.asarray(native['optimizer_parameters']).copy()
    value, _, state = likelihood.evaluate(parameters)
    for key, expected in (
        ('mean', 'coefficient_mean'), ('mean_covariance', 'population_mean_covariance'),
        ('residual_diagonal', 'residual_coefficient_diagonal'),
        ('discrepancy', 'discrepancy_variance'), ('common_diagonal', 'common_prior_coefficient_diagonal')):
        np.testing.assert_allclose(state[key], native[expected], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(value, native['source_negative_log_likelihood_per_label'], atol=1e-8, rtol=1e-8)
    center = state['mean']; size = len(center)
    parameters[-2] = quadrature['reference_gamma']
    reference = likelihood.evaluate(parameters)[0]
    evaluations = 0

    def moments(gamma):
        nonlocal evaluations
        parameters[-2] = gamma
        nll, _, current = likelihood.evaluate(parameters)
        weight = np.exp(-likelihood.label_count*(nll-reference))
        displacement = current['mean']-center
        evaluations += 1
        return weight*np.r_[1., gamma, gamma**2, displacement,
            current['mean_covariance'].ravel(), np.outer(displacement, displacement).ravel(),
            current['residual_diagonal']]

    endpoints = [quadrature['bounds'][0], *quadrature['split_points'], quadrature['bounds'][1]]
    pieces = [quad_vec(moments, lo, hi, epsabs=quadrature['epsabs'], epsrel=quadrature['epsrel'])
        for lo, hi in zip(endpoints[:-1], endpoints[1:])]
    total = sum(p[0] for p in pieces); mass = total[0]; normalized = total/mass
    displacement = normalized[3:3+size]
    offset = 3+size
    within = normalized[offset:offset+size**2].reshape(size, size)
    offset += size**2
    between = normalized[offset:offset+size**2].reshape(size, size)-np.outer(displacement, displacement)
    residual = normalized[offset+size**2:]
    common = within+between
    for covariance in (within, between, common, common+np.diag(residual)):
        np.testing.assert_allclose(covariance, covariance.T, atol=1e-10, rtol=1e-10)
        assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    tail = .01*likelihood.signal/(size-1)
    assert residual[0] >= 0. and np.all(residual[1:] >= tail-1e-12)
    gamma_mean = normalized[1]
    gamma_sd = np.sqrt(normalized[2]-gamma_mean**2)
    normalizer = mass/(endpoints[-1]-endpoints[0])
    for actual, key in ((normalizer, 'normalizer_against_uniform_reference'),
        (gamma_mean, 'conditional_likelihood_mean_gamma'), (gamma_sd, 'conditional_likelihood_sd_gamma')):
        np.testing.assert_allclose(actual, scalar_reference[key], atol=1e-8, rtol=1e-8)
    law = {'coefficient_mean': (center+displacement).tolist(),
        'population_mean_covariance': common.tolist(),
        'residual_coefficient_diagonal': residual.tolist(),
        'discrepancy_variance': native['discrepancy_variance'], 'source_signal_scale': likelihood.signal}
    diagnostics = {'source_label_count': likelihood.label_count,
        'native_fitted_gamma': native['spectral_exponent'],
        'conditional_likelihood_mean_gamma': float(gamma_mean),
        'conditional_likelihood_sd_gamma': float(gamma_sd),
        'normalizer_against_uniform_reference': float(normalizer),
        'quadrature_error_bound_over_normalizer': float(sum(p[1] for p in pieces)/mass),
        'likelihood_evaluations': evaluations,
        'within_gamma_common_covariance': within.tolist(),
        'between_gamma_common_covariance': between.tolist(),
        'common_covariance_min_eigenvalue': float(np.linalg.eigvalsh(common).min()),
        'new_task_covariance_min_eigenvalue': float(np.linalg.eigvalsh(common+np.diag(residual)).min()),
        'source_hyperparameters_reoptimized': False,
        'native_law_and_scalar_spectral_integral_reproduced': True}
    return law, diagnostics
