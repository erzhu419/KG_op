"""Source marginal likelihood for a shared mean and residual task functions."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from representation.orthogonal_function_prior import spectral_weights
from representation.source_function_prior import SourceFunctionPrior


class GaussianPopulationLikelihood:
    """Integrate common and task coefficients using task/feature solves."""

    def __init__(self, prior, batches, observation_variances):
        self.prior = prior
        self.basis = prior.basis
        self.signal = prior.diagnostics['source_signal_scale']
        self.tail = .01*self.signal/self.basis.feature_dim
        factor = cho_factor(prior.covariance, lower=True)
        self.precision0 = cho_solve(factor, np.eye(len(prior.mean)))
        self.logdet0 = 2*np.log(np.diag(factor[0])).sum()
        self.blocks = []
        for batch, noise in zip(batches, observation_variances):
            design = np.column_stack([np.ones(len(batch.psi)), self.basis.transform(batch.psi)])
            self.blocks.append((design, np.asarray(batch.signals)-design @ prior.mean, np.asarray(noise)))
        self.label_count = sum(len(y) for _, y, _ in self.blocks)

    def initial_parameters(self):
        info = self.prior.diagnostics
        return np.r_[self.prior.covariance[0, 0]/self.signal,
            np.asarray(info['source_group_variances'])/self.signal,
            info['spectral_exponent'], self.prior.discrepancy_variance/self.signal]

    def evaluate(self, parameters):
        return self.evaluate_with_common_precision(parameters, self.precision0, self.logdet0)

    def evaluate_with_common_precision(self, parameters, precision0, logdet0):
        groups = self.signal*np.asarray(parameters[1:-2])
        weights, derivative = spectral_weights(self.basis, parameters[-2])
        residual_diagonal = np.r_[self.signal*parameters[0],
            groups[self.basis.group_index]*weights+self.tail]
        delta = self.signal*parameters[-1]
        precision, rhs = precision0.copy(), np.zeros(len(self.prior.mean))
        logdet, solved = logdet0, []
        for design, error, noise in self.blocks:
            variance = (design*residual_diagonal) @ design.T+np.diag(noise+delta)
            factor = cho_factor(variance, lower=True)
            joined = cho_solve(factor, np.column_stack([design, error, np.eye(len(error))]))
            dimension = design.shape[1]
            S, W = joined[:, :dimension], joined[:, dimension+1:]
            precision += design.T @ S
            rhs += design.T @ joined[:, dimension]
            logdet += 2*np.log(np.diag(factor[0])).sum()
            solved.append((design, error, S, W))
        factor = cho_factor(precision, lower=True)
        displacement = cho_solve(factor, rhs)
        mean_covariance = cho_solve(factor, np.eye(len(rhs)))
        logdet += 2*np.log(np.diag(factor[0])).sum()
        # This sum equals sum(e' V^-1 e)-r' P^-1 r without subtracting
        # two large terms when the common function explains most source data.
        quadratic = float(displacement @ precision0 @ displacement)
        feature_gradient, delta_gradient = np.zeros(len(rhs)), 0.
        for design, error, S, W in solved:
            residual = error-design @ displacement
            alpha = W @ residual
            quadratic += residual @ alpha
            M = design.T @ S
            feature_gradient += .5*(np.diag(M)-np.sum((M @ mean_covariance)*M, axis=1)
                                     -(design.T @ alpha)**2)
            delta_gradient += .5*(np.trace(W)-np.sum((S @ mean_covariance)*S)-alpha @ alpha)
        gradient = np.r_[self.signal*feature_gradient[0],
            self.signal*np.bincount(self.basis.group_index, weights=feature_gradient[1:]*weights,
                                     minlength=len(self.basis.group_names)),
            feature_gradient[1:] @ (groups[self.basis.group_index]*derivative),
            self.signal*delta_gradient]/self.label_count
        value = .5*(self.label_count*np.log(2*np.pi)+logdet+quadratic)/self.label_count
        state = {'mean': self.prior.mean+displacement, 'mean_covariance': mean_covariance,
                 'residual_diagonal': residual_diagonal, 'discrepancy': float(delta),
                 'group_variances': groups, 'spectral_exponent': float(parameters[-2])}
        return float(value), gradient, state


class JointCovariancePopulationLikelihood(GaussianPopulationLikelihood):
    """Learn separate common-function and task-residual group/spectral laws."""

    def initial_parameters(self, previous):
        info = self.prior.diagnostics
        return np.r_[np.asarray(info['source_group_variances'])/self.signal, info['spectral_exponent'],
                     np.asarray(previous['optimizer_parameters'])]

    def evaluate(self, parameters):
        count = len(self.basis.group_names)
        groups = self.signal*np.asarray(parameters[:count])
        weights, derivative = spectral_weights(self.basis, parameters[count])
        common = np.r_[self.prior.covariance[0, 0], groups[self.basis.group_index]*weights+self.tail]
        value, residual_gradient, state = self.evaluate_with_common_precision(
            parameters[count+1:], np.diag(1/common), np.log(common).sum())
        displacement = state['mean']-self.prior.mean
        coefficient_gradient = .5*(common-np.diag(state['mean_covariance'])-displacement**2)/common**2
        common_gradient = np.r_[self.signal*np.bincount(self.basis.group_index,
            weights=coefficient_gradient[1:]*weights, minlength=count),
            coefficient_gradient[1:] @ (groups[self.basis.group_index]*derivative)]/self.label_count
        state.update({'common_diagonal': common, 'common_group_variances': groups,
                      'common_spectral_exponent': float(parameters[count])})
        return value, np.r_[common_gradient, residual_gradient], state


class FlatCovariancePopulationLikelihood(GaussianPopulationLikelihood):
    """Learn flat common and task spectra in the same orthogonal dictionary."""

    def initial_parameters(self, previous):
        return np.array([np.sum(previous['common_group_variances'])/self.signal,
            previous['residual_coefficient_diagonal'][0]/self.signal,
            np.sum(previous['source_group_variances'])/self.signal,
            previous['discrepancy_variance']/self.signal])

    def evaluate(self, parameters):
        common_scale, intercept, residual_scale, discrepancy = parameters
        dimension = self.basis.feature_dim
        fractions = np.bincount(self.basis.group_index)/dimension
        common = np.r_[self.prior.covariance[0, 0],
                       np.full(dimension, self.signal*common_scale/dimension+self.tail)]
        residual_parameters = np.r_[intercept, residual_scale*fractions, 0., discrepancy]
        value, residual_gradient, state = self.evaluate_with_common_precision(
            residual_parameters, np.diag(1/common), np.log(common).sum())
        displacement = state['mean']-self.prior.mean
        coefficient_gradient = .5*(common-np.diag(state['mean_covariance'])-displacement**2)/common**2
        common_gradient = self.signal*coefficient_gradient[1:].sum()/dimension/self.label_count
        gradient = np.array([common_gradient, residual_gradient[0],
                             residual_gradient[1:-2] @ fractions, residual_gradient[-1]])
        state['common_diagonal'] = common
        return value, gradient, state


def fit_flat_common_source_population(prior, batches, observation_variances, previous, *, optimizer):
    likelihood = FlatCovariancePopulationLikelihood(prior, batches, observation_variances)
    initial = likelihood.initial_parameters(previous)
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'], bounds=[(0., None)]*4,
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    projected = np.where((result.x <= 0) & (gradient > 0), 0., gradient)
    info = {**prior.basis.diagnostics(), 'mode': 'same_source_orthogonal_full',
        'population_model': 'jointly learned flat common-function and task-residual spectra',
        'source_label_count': likelihood.label_count, 'source_task_count': len(likelihood.blocks),
        'source_signal_scale': likelihood.signal, 'tail_variance_budget': .01*likelihood.signal,
        'common_function_total_excess_variance': float(likelihood.signal*result.x[0]),
        'residual_function_total_excess_variance': float(likelihood.signal*result.x[2]),
        'common_prior_coefficient_diagonal': state['common_diagonal'].tolist(),
        'common_intercept_variance_fixed': float(prior.covariance[0, 0]),
        'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(),
        'discrepancy_variance': state['discrepancy'], 'discrepancy_lower_bound': 0.,
        'new_task_covariance_includes_population_mean_uncertainty': True,
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': optimizer['max_iterations'], 'optimizer_message': str(result.message),
        'optimizer_parameters': result.x.tolist(), 'optimizer_initial_parameters': initial.tolist(),
        'optimizer_parameter_order': ['common_function_scale', 'task_intercept_scale', 'task_function_scale', 'discrepancy_scale'],
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': likelihood.evaluate(initial)[0],
        'likelihood': 'Gaussian source batch labels; log-chi-square approximation for eta'}
    covariance = state['mean_covariance']+np.diag(state['residual_diagonal'])
    return SourceFunctionPrior(prior.basis, state['mean'], covariance, state['discrepancy'], info)


def fit_joint_common_source_population(prior, batches, observation_variances, previous, *, optimizer):
    likelihood = JointCovariancePopulationLikelihood(prior, batches, observation_variances)
    initial = likelihood.initial_parameters(previous)
    return _solve_joint_common_population(likelihood, initial,
        previous['source_negative_log_likelihood_per_label'], optimizer)


def continue_joint_common_source_population(prior, batches, observation_variances, previous, *, optimizer):
    """Resume the reached joint source objective from its complete parameter vector."""
    likelihood = JointCovariancePopulationLikelihood(prior, batches, observation_variances)
    completed = _solve_joint_common_population(likelihood, np.asarray(previous['optimizer_parameters']),
        previous['fixed_common_source_negative_log_likelihood_per_label'], optimizer)
    completed.diagnostics['continued_from_optimizer_iterations'] = previous['optimizer_iterations']
    completed.diagnostics['continued_from_source_negative_log_likelihood_per_label'] = previous['source_negative_log_likelihood_per_label']
    return completed


def _solve_joint_common_population(likelihood, initial, fixed_common_value, optimizer):
    prior = likelihood.prior
    count = len(prior.basis.group_names)
    initial_value = likelihood.evaluate(initial)[0]
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'],
        bounds=[(0., None)]*count+[(0., 4.)]+[(0., None)]*(count+1)+[(0., 4.), (0., None)],
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    upper = np.full(len(result.x), np.inf); upper[count] = upper[-2] = 4.
    projected = np.where(((result.x <= 0) & (gradient > 0)) |
                         ((result.x >= upper) & (gradient < 0)), 0., gradient)
    info = {**prior.basis.diagnostics(), 'mode': 'source_orthogonal_group_spectral',
        'population_model': 'jointly learned common-function and task-residual covariances',
        'source_label_count': likelihood.label_count, 'source_task_count': len(likelihood.blocks),
        'source_signal_scale': likelihood.signal, 'tail_variance_budget': .01*likelihood.signal,
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
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': initial_value,
        'fixed_common_source_negative_log_likelihood_per_label': fixed_common_value,
        'likelihood': 'Gaussian source batch labels; log-chi-square approximation for eta'}
    covariance = state['mean_covariance']+np.diag(state['residual_diagonal'])
    return SourceFunctionPrior(prior.basis, state['mean'], covariance, state['discrepancy'], info)


def _solve_population(likelihood, initial, max_iterations):
    prior = likelihood.prior
    count = len(prior.basis.group_names)
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method='L-BFGS-B', bounds=[(0., None)]*(count+1)+[(0., 4.), (0., None)],
        options={'maxiter': max_iterations, 'ftol': 1e-9, 'gtol': 1e-5})
    value, gradient, state = likelihood.evaluate(result.x)
    upper = np.full(len(result.x), np.inf); upper[-2] = 4.
    projected = np.where(((result.x <= 0) & (gradient > 0)) |
                         ((result.x >= upper) & (gradient < 0)), 0., gradient)
    covariance = np.diag(state['residual_diagonal'])+state['mean_covariance']
    info = {**prior.basis.diagnostics(), 'mode': 'source_orthogonal_group_spectral',
        'population_model': 'shared Gaussian mean and residual task function',
        'source_label_count': likelihood.label_count, 'source_task_count': len(likelihood.blocks),
        'source_signal_scale': likelihood.signal, 'tail_variance_budget': .01*likelihood.signal,
        'source_group_variances': state['group_variances'].tolist(),
        'source_zero_group_count': int(np.count_nonzero(state['group_variances'] == 0)),
        'spectral_exponent': state['spectral_exponent'], 'spectral_exponent_bounds': [0., 4.],
        'coefficient_mean': state['mean'].tolist(),
        'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(),
        'discrepancy_variance': state['discrepancy'], 'discrepancy_lower_bound': 0.,
        'new_task_covariance_includes_population_mean_uncertainty': True,
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': max_iterations, 'optimizer_message': str(result.message),
        'optimizer_parameters': result.x.tolist(),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': value,
        'likelihood': 'Gaussian source batch labels; log-chi-square approximation for eta'}
    return SourceFunctionPrior(prior.basis, state['mean'], covariance, state['discrepancy'], info)


def fit_joint_source_population(prior, batches, observation_variances, *, max_iterations=1024):
    likelihood = GaussianPopulationLikelihood(prior, batches, observation_variances)
    return _solve_population(likelihood, likelihood.initial_parameters(), max_iterations)


def continue_joint_source_population(regularizer, batches, observation_variances, previous, *, max_iterations=4096):
    """Complete the same objective from a recorded iteration-limited source fit."""
    likelihood = GaussianPopulationLikelihood(regularizer, batches, observation_variances)
    d = previous.diagnostics
    initial = np.r_[d['residual_coefficient_diagonal'][0]/likelihood.signal,
                   np.asarray(d['source_group_variances'])/likelihood.signal,
                   d['spectral_exponent'], previous.discrepancy_variance/likelihood.signal]
    initial_value, _, _ = likelihood.evaluate(initial)
    completed = _solve_population(likelihood, initial, max_iterations)
    completed.diagnostics.update({'continued_from_optimizer_iterations': d['optimizer_iterations'],
        'continued_from_converged': d['optimizer_converged'],
        'continued_from_source_negative_log_likelihood_per_label': initial_value,
        'previous_source_negative_log_likelihood_per_label': d['source_negative_log_likelihood_per_label']})
    return completed


def same_source_full_residual_prior(prior):
    """Pool only task residual function variance, retaining shared-mean error."""
    residual = np.asarray(prior.diagnostics['residual_coefficient_diagonal']).copy()
    residual[1:] = residual[1:].mean()
    covariance = np.asarray(prior.diagnostics['population_mean_covariance'])+np.diag(residual)
    return SourceFunctionPrior(prior.basis, prior.mean.copy(), covariance, prior.discrepancy_variance,
        {**prior.diagnostics, 'mode': 'same_source_orthogonal_full',
         'residual_coefficient_diagonal': residual.tolist(), 'shared_mean_uncertainty_and_discrepancy': True})
