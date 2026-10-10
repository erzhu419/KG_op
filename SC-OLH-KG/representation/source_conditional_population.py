"""Conditional empirical-Bayes source fitting in the original Gaussian family.

The source empirical center/regularizer is fixed for every conditional term.
This is a composite fitting criterion, not unbiased internal cross-validation.
"""

from copy import copy

import numpy as np
from scipy.optimize import minimize


class SourceTaskConditionalLikelihood:
    def __init__(self, full):
        self.full = full
        self.subsets = []
        for i in range(len(full.blocks)):
            subset = copy(full)
            subset.blocks = [b for j, b in enumerate(full.blocks) if j != i]
            subset.label_count = sum(len(b[1]) for b in subset.blocks)
            self.subsets.append(subset)

    def evaluate(self, parameters):
        value, gradient, state = self.full.evaluate(parameters)
        total = len(self.subsets)*self.full.label_count*value
        derivative = len(self.subsets)*self.full.label_count*gradient
        for subset in self.subsets:
            v, g, _ = subset.evaluate(parameters)
            total -= subset.label_count*v
            derivative -= subset.label_count*g
        return float(total/self.full.label_count), derivative/self.full.label_count, state


def fit_source_conditional_population(likelihood, native, optimizer):
    full = likelihood.full; prior = full.prior; count = len(full.basis.group_names)
    initial = np.asarray(native['optimizer_parameters'])
    initial_conditional = likelihood.evaluate(initial)[0]
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'],
        bounds=[(0., None)]*count+[(0., 4.)]+[(0., None)]*(count+1)+[(0., 4.), (0., None)],
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    conditional, gradient, _ = likelihood.evaluate(result.x)
    marginal, _, state = full.evaluate(result.x)
    upper = np.full(len(result.x), np.inf); upper[count] = upper[-2] = 4.
    projected = np.where(((result.x <= 0) & (gradient > 0)) |
        ((result.x >= upper) & (gradient < 0)), 0., gradient)
    return {**prior.basis.diagnostics(), 'mode': 'source_orthogonal_group_spectral',
        'population_model': 'original common/task Gaussian family with source conditional composite fitting',
        'source_fitting_criterion': 'conditional empirical-Bayes source-task composite density',
        'empirical_regularizer_fixed_from_outer_training_sources': True,
        'internal_cross_validation_unbiased': False,
        'source_label_count': full.label_count, 'source_task_count': len(full.blocks),
        'source_signal_scale': full.signal, 'tail_variance_budget': full.basis.feature_dim*full.tail,
        'source_group_variances': state['group_variances'].tolist(),
        'source_zero_group_count': int(np.count_nonzero(state['group_variances'] == 0)),
        'spectral_exponent': state['spectral_exponent'], 'spectral_exponent_bounds': [0., 4.],
        'common_group_variances': state['common_group_variances'].tolist(),
        'common_zero_group_count': int(np.count_nonzero(state['common_group_variances'] == 0)),
        'common_spectral_exponent': state['common_spectral_exponent'],
        'common_prior_coefficient_diagonal': state['common_diagonal'].tolist(),
        'common_intercept_variance_fixed': float(prior.covariance[0, 0]),
        'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(),
        'discrepancy_variance': state['discrepancy'], 'discrepancy_lower_bound': 0.,
        'new_task_covariance_includes_population_mean_uncertainty': True,
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': optimizer['max_iterations'], 'optimizer_message': str(result.message),
        'optimizer_parameters': result.x.tolist(), 'optimizer_initial_parameters': initial.tolist(),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': marginal,
        'initial_source_negative_log_likelihood_per_label': full.evaluate(initial)[0],
        'source_conditional_composite_negative_log_likelihood_per_label': conditional,
        'initial_source_conditional_composite_negative_log_likelihood_per_label': initial_conditional,
        'fixed_common_source_negative_log_likelihood_per_label': native['fixed_common_source_negative_log_likelihood_per_label'],
        'likelihood': 'Gaussian source batch labels; source conditional composite fitting; full-source Gaussian posterior'}
