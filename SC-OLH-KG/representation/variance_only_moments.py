"""Local log-risk moments conditional only on an observed sample variance."""

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar

from representation.physical_risk_prior import log_sample_variance_density


def variance_only_moments(mean, variance, sample_variance, replicates):
    """Gaussian eta prior times exact chi-square likelihood; no mean label."""
    def log_density(eta):
        return -.5*(eta-mean)**2/variance+log_sample_variance_density(eta, sample_variance, replicates)

    lo, hi = mean-12*np.sqrt(variance), mean+12*np.sqrt(variance)
    mode = minimize_scalar(lambda eta: -log_density(eta), bounds=(lo, hi), method='bounded').x
    shift = log_density(mode)

    def integrand(eta):
        return np.exp(log_density(eta)-shift)*np.array([1., eta-mean, (eta-mean)**2, np.exp(eta)])

    values, error = quad_vec(integrand, lo, hi, points=[mode], epsabs=1e-9, epsrel=1e-9)
    moments = values/values[0]
    return {'eta_mean': float(mean+moments[1]), 'eta_variance': float(moments[2]-moments[1]**2),
        'expected_episode_variance': float(moments[3]), 'integration_relative_error': float(error/values[0])}
