"""Coherent common scale, spectrum and correlation estimation on fixed support."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from representation.orthogonal_function_prior import spectral_weights


class CommonGeometryLikelihood:
    def __init__(self, source, native, geometry):
        self.source, self.native = source, native
        self.groups = np.flatnonzero(np.asarray(native['common_group_variances']) > 0)
        self.active = np.flatnonzero(np.isin(source.basis.group_index, self.groups))
        np.testing.assert_array_equal(self.active, geometry['active_common_shape_indices'])
        self.correlation = np.asarray(geometry['projected_gp_active_shape_correlation'])
        self.group_masks = [source.basis.group_index[self.active] == g for g in self.groups]
        count = len(source.basis.group_names)
        self.task_parameters = np.asarray(native['optimizer_parameters'])[count+1:]
        self.initial = np.r_[np.sqrt(np.asarray(native['common_group_variances'])[self.groups]/source.signal),
            native['common_spectral_exponent'], 0.]
        self.bounds = [(0., None)]*len(self.groups)+[(0., 4.), (0., 1.)]

    def covariance(self, parameters):
        weights, derivative = spectral_weights(self.source.basis, parameters[-2])
        groups = np.zeros(len(self.source.basis.group_names))
        groups[self.groups] = self.source.signal*parameters[:-2]**2
        root_weight = np.sqrt(self.source.signal*weights[self.active])
        amplitude = sum(s*mask for s, mask in zip(parameters[:-2], self.group_masks))
        root = amplitude*root_weight
        correlation = (1-parameters[-1])*np.eye(len(root))+parameters[-1]*self.correlation
        covariance = np.diag(np.r_[self.native['common_prior_coefficient_diagonal'][0],
            np.full(self.source.basis.feature_dim, self.source.tail)])
        indices = np.ix_(self.active+1, self.active+1)
        covariance[indices] += root[:, None]*correlation*root[None, :]
        root_derivatives = [root_weight*mask for mask in self.group_masks]
        root_derivatives.append(.5*root*derivative[self.active]/weights[self.active])
        directions = [dr[:, None]*correlation*root[None, :]+root[:, None]*correlation*dr[None, :]
            for dr in root_derivatives]
        directions.append(root[:, None]*(self.correlation-np.eye(len(root)))*root[None, :])
        return covariance, groups, directions

    def evaluate(self, parameters):
        covariance, groups, directions = self.covariance(np.asarray(parameters))
        factor = cho_factor(covariance, lower=True)
        inverse = cho_solve(factor, np.eye(len(covariance)))
        value, _, state = self.source.evaluate_with_common_precision(
            self.task_parameters, inverse, 2*np.log(np.diag(factor[0])).sum())
        displacement = state['mean']-self.source.prior.mean
        gradient = .5*(inverse-inverse@(state['mean_covariance']+np.outer(displacement, displacement))@inverse)
        active_gradient = gradient[np.ix_(self.active+1, self.active+1)]
        derivatives = np.array([np.sum(active_gradient*d)/self.source.label_count for d in directions])
        state.update(common_prior_covariance=covariance, common_group_variances=groups,
            common_spectral_exponent=float(parameters[-2]), common_correlation_strength=float(parameters[-1]))
        return value, derivatives, state


def fit_common_geometry(likelihood, optimizer):
    fitted = minimize(lambda p: likelihood.evaluate(p)[:2], likelihood.initial, jac=True,
        method=optimizer['method'], bounds=likelihood.bounds,
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(fitted.x)
    projected = gradient.copy()
    for j, (lo, hi) in enumerate(likelihood.bounds):
        if (fitted.x[j] <= lo and gradient[j] > 0) or (hi is not None and fitted.x[j] >= hi and gradient[j] < 0):
            projected[j] = 0.
    return {**likelihood.native,
        'population_model': 'source-estimated common scales/spectrum/correlation, fixed native sparse support and task law',
        'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'common_prior_coefficient_covariance': state['common_prior_covariance'].tolist(),
        'common_prior_coefficient_diagonal': np.diag(state['common_prior_covariance']).tolist(),
        'common_group_variances': state['common_group_variances'].tolist(),
        'common_zero_group_count': int(np.count_nonzero(state['common_group_variances'] == 0)),
        'common_spectral_exponent': state['common_spectral_exponent'],
        'common_correlation_strength': state['common_correlation_strength'],
        'common_active_shape_indices': likelihood.active.tolist(),
        'common_prior_active_correlation_template': likelihood.correlation.tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(), 'discrepancy_variance': state['discrepancy'],
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': likelihood.native['source_negative_log_likelihood_per_label'],
        'native_optimizer_parameters': likelihood.native['optimizer_parameters'],
        'optimizer_parameter_names': [f'common_group_sd:{likelihood.source.basis.group_names[g]}' for g in likelihood.groups]
            +['common_spectral_exponent', 'common_correlation_strength'],
        'optimizer_parameters': fitted.x.tolist(), 'optimizer_initial_parameters': likelihood.initial.tolist(),
        'optimizer_converged': bool(fitted.success), 'optimizer_message': str(fitted.message),
        'optimizer_iterations': int(fitted.nit), 'optimizer_evaluations': int(fitted.nfev),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))), 'optimizer': optimizer,
        'native_support_intercept_tail_task_law_preserved': True, 'common_marginals_and_spectrum_unchanged': False}
