"""One source-fitted correlation strength in the native common coefficient prior."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


class CommonPriorCorrelationLikelihood:
    def __init__(self, likelihood, native, direction):
        self.source = likelihood
        self.native = native
        self.direction = np.asarray(direction)
        self.diagonal = np.asarray(native['common_prior_coefficient_diagonal'])
        count = len(likelihood.basis.group_names)
        self.task_parameters = np.asarray(native['optimizer_parameters'])[count+1:]

    def evaluate(self, p):
        covariance = np.diag(self.diagonal)+float(p)*self.direction
        factor = cho_factor(covariance, lower=True)
        inverse = cho_solve(factor, np.eye(len(self.diagonal)))
        value, _, state = self.source.evaluate_with_common_precision(
            self.task_parameters, inverse, 2*np.log(np.diag(factor[0])).sum())
        displacement = state['mean']-self.source.prior.mean
        gradient = .5*(inverse-inverse@(state['mean_covariance']+np.outer(displacement, displacement))@inverse)
        derivative = float(np.sum(gradient*self.direction)/self.source.label_count)
        state['common_prior_covariance'] = covariance
        return value, derivative, state


def fit_common_correlation(likelihood, optimizer):
    initial_value = likelihood.native['source_negative_log_likelihood_per_label']
    fitted = minimize(lambda p: likelihood.evaluate(p[0])[:2], np.array([0.]), jac=True,
        method=optimizer['method'], bounds=[(0., 1.)],
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    p = float(fitted.x[0]); value, derivative, state = likelihood.evaluate(p)
    projected = 0. if (p <= 0 and derivative > 0) or (p >= 1 and derivative < 0) else derivative
    return {**likelihood.native,
        'population_model': 'native orthogonal common/task family with fixed-marginal source GP common correlation',
        'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(),
        'discrepancy_variance': state['discrepancy'],
        'common_prior_coefficient_covariance': state['common_prior_covariance'].tolist(),
        'common_correlation_direction': likelihood.direction.tolist(), 'common_correlation_strength': p,
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': initial_value,
        'native_optimizer_parameters': likelihood.native['optimizer_parameters'],
        'optimizer_parameters': [p], 'optimizer_initial_parameters': [0.],
        'optimizer_converged': bool(fitted.success), 'optimizer_message': str(fitted.message),
        'optimizer_iterations': int(fitted.nit), 'optimizer_evaluations': int(fitted.nfev),
        'optimizer_projected_gradient_max': float(abs(projected)),
        'optimizer_unprojected_derivative': derivative, 'optimizer': optimizer,
        'common_marginals_support_spectrum_intercept_tail_and_task_law_unchanged': True}
