"""Restrict task excess functions while learning the original common structure."""

import numpy as np
from scipy.optimize import minimize

from representation.source_function_prior import SourceFunctionPrior
from representation.source_joint_population import JointCovariancePopulationLikelihood


class InterceptTaskPopulationLikelihood(JointCovariancePopulationLikelihood):
    def __init__(self, prior, batches, observation_variances):
        super().__init__(prior, batches, observation_variances)
        count = len(self.basis.group_names)
        self.free_indices = np.r_[np.arange(count+2), 2*count+3]
        self.full_parameter_count = 2*count+4

    def initial_parameters(self, previous):
        return np.asarray(previous['optimizer_parameters'])[self.free_indices]

    def expand(self, parameters):
        full = np.zeros(self.full_parameter_count)
        full[self.free_indices] = parameters
        return full

    def evaluate(self, parameters):
        value, gradient, state = super().evaluate(self.expand(parameters))
        return value, gradient[self.free_indices], state


def fit_intercept_task_population(likelihood, previous, *, optimizer):
    count = len(likelihood.basis.group_names); initial = likelihood.initial_parameters(previous)
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'], bounds=[(0., None)]*count+[(0., 4.), (0., None), (0., None)],
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    upper = np.full(len(result.x), np.inf); upper[count] = 4.
    projected = np.where(((result.x <= 0) & (gradient > 0)) |
        ((result.x >= upper) & (gradient < 0)), 0., gradient)
    info = {**previous, 'population_model': 'sparse spectral common function; task intercept plus original weak tail',
        'source_group_variances': state['group_variances'].tolist(), 'source_zero_group_count': count,
        'spectral_exponent': 0., 'task_spectral_exponent_identified': False,
        'common_group_variances': state['common_group_variances'].tolist(),
        'common_zero_group_count': int(np.count_nonzero(state['common_group_variances'] == 0)),
        'common_spectral_exponent': state['common_spectral_exponent'],
        'common_prior_coefficient_diagonal': state['common_diagonal'].tolist(),
        'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(), 'discrepancy_variance': state['discrepancy'],
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': optimizer['max_iterations'], 'optimizer_message': str(result.message),
        'optimizer_parameters': likelihood.expand(result.x).tolist(),
        'optimizer_initial_parameters': likelihood.expand(initial).tolist(),
        'optimizer_free_parameter_indices': likelihood.free_indices.tolist(),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': likelihood.evaluate(initial)[0]}
    np.testing.assert_allclose(state['residual_diagonal'][1:], likelihood.tail, atol=1e-14)
    covariance = state['mean_covariance']+np.diag(state['residual_diagonal'])
    return SourceFunctionPrior(likelihood.basis, state['mean'], covariance, state['discrepancy'], info)
