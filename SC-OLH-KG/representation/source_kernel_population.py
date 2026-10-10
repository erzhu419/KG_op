"""Generic shared-function GP and independent residual task GPs."""

import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize


def rbf_cross(left, right, lengthscale):
    difference = (left[:, None]-right[None, :])/np.asarray(lengthscale)
    return np.exp(-.5*np.sum(difference**2, axis=2))


def shared_kernel(left, right, anchor):
    return (anchor['signal_variance']*rbf_cross(left, right, anchor['lengthscale'])
            + anchor['constant_variance'])


class KernelPopulationLikelihood:
    """Source tasks share sites; whiten the common GP without variance jitter."""

    def __init__(self, features, responses, noises, anchor):
        self.features = np.asarray(features)
        self.anchor = anchor
        self.unit_kernel = rbf_cross(self.features, self.features, anchor['lengthscale'])
        self.constant_kernel = np.ones_like(self.unit_kernel)
        self.identity = np.eye(len(self.features))
        covariance = shared_kernel(self.features, self.features, anchor)
        self.root = (np.full((len(self.features), 1), np.sqrt(anchor['constant_variance']))
                     if anchor['signal_variance'] == 0 else np.linalg.cholesky(covariance))
        self.coefficient_identity = np.eye(self.root.shape[1])
        self.blocks = [(np.asarray(y)-anchor['constant'], np.asarray(noise))
                       for y, noise in zip(responses, noises)]
        self.label_count = sum(len(y) for y, _ in self.blocks)

    def evaluate(self, parameters):
        amplitude, intercept, discrepancy = parameters
        residual_covariance = amplitude*self.unit_kernel+intercept*self.constant_kernel
        precision, rhs = self.coefficient_identity.copy(), np.zeros(self.root.shape[1])
        logdet, solved = 0., []
        for error, noise in self.blocks:
            variance = residual_covariance+np.diag(noise+discrepancy)
            factor = cho_factor(variance, lower=True)
            joined = cho_solve(factor, np.column_stack([self.root, error, self.identity]))
            dimension = self.root.shape[1]
            S, W = joined[:, :dimension], joined[:, dimension+1:]
            precision += self.root.T @ S
            rhs += self.root.T @ joined[:, dimension]
            logdet += 2*np.log(np.diag(factor[0])).sum()
            solved.append((error, S, W))
        factor = cho_factor(precision, lower=True)
        displacement = cho_solve(factor, rhs)
        uncertainty = cho_solve(factor, self.coefficient_identity)
        logdet += 2*np.log(np.diag(factor[0])).sum()
        quadratic = float(displacement @ displacement)
        gradient = np.zeros(3)
        for error, S, W in solved:
            remaining = error-self.root @ displacement
            alpha = W @ remaining
            quadratic += remaining @ alpha
            contrast = W-S @ uncertainty @ S.T-np.outer(alpha, alpha)
            gradient += .5*np.array([np.sum(contrast*self.unit_kernel),
                                    contrast.sum(), np.trace(contrast)])
        value = .5*(self.label_count*np.log(2*np.pi)+logdet+quadratic)/self.label_count
        state = {'whitened_mean': displacement, 'whitened_covariance': uncertainty,
                 'mean': self.anchor['constant']+self.root @ displacement,
                 'mean_covariance': self.root @ uncertainty @ self.root.T}
        return float(value), gradient/self.label_count, state


def fit_kernel_population(tasks, head, anchor, *, optimizer):
    x = np.asarray(tasks[0]['features'])
    if not all(np.array_equal(x, np.asarray(task['features'])) for task in tasks):
        raise ValueError('this frozen source population must use one common site grid')
    center, scale = anchor['normalization']['mean'], anchor['normalization']['scale']
    responses = [(np.asarray(t['response'])[:, head]-center)/scale for t in tasks]
    noises = [np.asarray(t['noise'])[:, head]/scale**2 for t in tasks]
    likelihood = KernelPopulationLikelihood(x, responses, noises, anchor)
    initial = np.array([anchor['signal_variance'], anchor['constant_variance'],
                        anchor['discrepancy_variance']])
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'], bounds=[(0., None)]*3,
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'],
                 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    projected = np.where((result.x <= 0) & (gradient > 0), 0., gradient)
    return {'anchor': anchor, 'residual_parameters': result.x.tolist(),
        'common_function_whitened_mean': state['whitened_mean'].tolist(),
        'common_function_whitened_covariance': state['whitened_covariance'].tolist(),
        'common_function_source_mean': state['mean'].tolist(),
        'source_label_count': likelihood.label_count, 'source_task_count': len(tasks),
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': optimizer['max_iterations'],
        'optimizer_message': str(result.message),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': value,
        'optimizer_initial_residual_parameters': initial.tolist()}


def predict_kernel_population(source_features, target_features, fit):
    """Include interpolation uncertainty for a new independent task."""
    xs, xt = np.asarray(source_features), np.asarray(target_features)
    anchor = fit['anchor']
    if anchor['signal_variance'] == 0:
        transfer = np.full((len(xt), 1), np.sqrt(anchor['constant_variance']))
    else:
        root = np.linalg.cholesky(shared_kernel(xs, xs, anchor))
        transfer = solve_triangular(root, shared_kernel(xs, xt, anchor), lower=True).T
    common_mean = anchor['constant']+transfer @ np.asarray(fit['common_function_whitened_mean'])
    uncertainty = np.asarray(fit['common_function_whitened_covariance'])
    common_covariance = (shared_kernel(xt, xt, anchor)-transfer @ transfer.T
                         +transfer @ uncertainty @ transfer.T)
    amplitude, intercept, discrepancy = fit['residual_parameters']
    covariance = (common_covariance+amplitude*rbf_cross(xt, xt, anchor['lengthscale'])
                  +intercept+discrepancy*np.eye(len(xt)))
    scale = anchor['normalization']['scale']
    mean = anchor['normalization']['mean']+scale*common_mean
    return mean, scale**2*(covariance+covariance.T)/2


class JointCommonKernelLikelihood:
    """Learn common and residual RBF amplitudes at the existing lengthscales.

    The reached population has six tasks on16 sites. A96-label dense solve
    supports exact zero common amplitudes without factoring the common kernel.
    The source-derived common intercept variance stays fixed.
    """

    def __init__(self, features, responses, noises, anchor):
        self.features, self.anchor = np.asarray(features), anchor
        self.unit_kernel = rbf_cross(self.features, self.features, anchor['lengthscale'])
        count = len(responses)
        self.common_function_kernel = np.tile(self.unit_kernel, (count, count))
        self.common_intercept_kernel = np.ones_like(self.common_function_kernel)
        self.residual_function_kernel = block_diag(*[self.unit_kernel]*count)
        self.residual_intercept_kernel = block_diag(*[np.ones_like(self.unit_kernel)]*count)
        self.identity = np.eye(count*len(self.features))
        self.error = np.concatenate(responses)-anchor['constant']
        self.noise = np.concatenate(noises)
        self.task_count, self.label_count = count, len(self.error)

    def evaluate(self, parameters):
        common_amplitude, amplitude, intercept, discrepancy = parameters
        covariance = (common_amplitude*self.common_function_kernel
            +self.anchor['constant_variance']*self.common_intercept_kernel
            +amplitude*self.residual_function_kernel+intercept*self.residual_intercept_kernel
            +np.diag(self.noise+discrepancy))
        factor = cho_factor(covariance, lower=True)
        solved = cho_solve(factor, np.column_stack([self.error, self.identity]))
        alpha, inverse = solved[:, 0], solved[:, 1:]
        contrast = .5*(inverse-np.outer(alpha, alpha))
        gradient = np.array([np.sum(contrast*self.common_function_kernel),
                             np.sum(contrast*self.residual_function_kernel),
                             np.sum(contrast*self.residual_intercept_kernel), np.trace(contrast)])
        value = .5*(self.label_count*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+self.error @ alpha)
        count, n = self.task_count, len(self.features)
        state = {'alpha_sum': alpha.reshape(count, n).sum(axis=0),
                 'precision_sum': inverse.reshape(count, n, count, n).sum(axis=(0, 2))}
        return float(value/self.label_count), gradient/self.label_count, state


def fit_joint_common_kernel_population(tasks, head, previous, *, optimizer):
    anchor = previous['anchor']; x = np.asarray(tasks[0]['features'])
    if not all(np.array_equal(x, np.asarray(t['features'])) for t in tasks):
        raise ValueError('the registered population requires common source sites')
    center, scale = anchor['normalization']['mean'], anchor['normalization']['scale']
    responses = [(np.asarray(t['response'])[:, head]-center)/scale for t in tasks]
    noises = [np.asarray(t['noise'])[:, head]/scale**2 for t in tasks]
    likelihood = JointCommonKernelLikelihood(x, responses, noises, anchor)
    initial = np.r_[anchor['signal_variance'], previous['residual_parameters']]
    initial_value = likelihood.evaluate(initial)[0]
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'], bounds=[(0., None)]*4,
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    projected = np.where((result.x <= 0) & (gradient > 0), 0., gradient)
    return {'anchor': {**anchor, 'signal_variance': float(result.x[0])},
        'residual_parameters': result.x[1:].tolist(),
        'common_alpha_sum': state['alpha_sum'].tolist(), 'common_precision_sum': state['precision_sum'].tolist(),
        'common_intercept_variance_fixed': anchor['constant_variance'],
        'source_label_count': likelihood.label_count, 'source_task_count': len(tasks),
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': optimizer['max_iterations'], 'optimizer_message': str(result.message),
        'optimizer_parameters': result.x.tolist(), 'optimizer_initial_parameters': initial.tolist(),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': initial_value,
        'fixed_common_source_negative_log_likelihood_per_label': previous['source_negative_log_likelihood_per_label']}


def predict_joint_common_kernel_population(source_features, target_features, fit):
    """Condition a new task, retaining shared interpolation uncertainty."""
    xs, xt = np.asarray(source_features), np.asarray(target_features)
    anchor = fit['anchor']; cross = shared_kernel(xt, xs, anchor)
    mean = anchor['constant']+cross @ np.asarray(fit['common_alpha_sum'])
    covariance = shared_kernel(xt, xt, anchor)-cross @ np.asarray(fit['common_precision_sum']) @ cross.T
    amplitude, intercept, discrepancy = fit['residual_parameters']
    covariance += (amplitude*rbf_cross(xt, xt, anchor['lengthscale'])+intercept
                   +discrepancy*np.eye(len(xt)))
    scale = anchor['normalization']['scale']
    return anchor['normalization']['mean']+scale*mean, scale**2*(covariance+covariance.T)/2


class UnequalSiteKernelLikelihood:
    """Exact common GP plus task GPs on each task's actual observation sites."""

    def __init__(self, features, responses, noises, anchor):
        self.features = np.vstack(features)
        self.anchor = anchor
        self.common_kernel = rbf_cross(self.features, self.features, anchor['lengthscale'])
        self.residual_kernel = block_diag(*[
            rbf_cross(x, x, anchor['lengthscale']) for x in features])
        self.residual_intercept = block_diag(*[np.ones((len(x), len(x))) for x in features])
        self.error = np.concatenate(responses)-anchor['constant']
        self.noise = np.concatenate(noises)
        self.identity = np.eye(len(self.error))
        self.task_count, self.label_count = len(features), len(self.error)

    def evaluate(self, parameters):
        common, residual, intercept, discrepancy = parameters
        covariance = (common*self.common_kernel+self.anchor['constant_variance']
            +residual*self.residual_kernel+intercept*self.residual_intercept
            +np.diag(self.noise+discrepancy))
        factor = cho_factor(covariance, lower=True)
        solved = cho_solve(factor, np.column_stack([self.error, self.identity]))
        alpha, inverse = solved[:, 0], solved[:, 1:]
        contrast = .5*(inverse-np.outer(alpha, alpha))
        gradient = np.array([np.sum(contrast*self.common_kernel),
            np.sum(contrast*self.residual_kernel), np.sum(contrast*self.residual_intercept),
            np.trace(contrast)])/self.label_count
        value = .5*(self.label_count*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()
                    +self.error @ alpha)/self.label_count
        return float(value), gradient, {'alpha': alpha, 'precision': inverse}


def _unequal_site_likelihood(tasks, head, anchor):
    center, scale = anchor['normalization']['mean'], anchor['normalization']['scale']
    return UnequalSiteKernelLikelihood([np.asarray(t['features']) for t in tasks],
        [(np.asarray(t['response'])[:, head]-center)/scale for t in tasks],
        [np.asarray(t['noise'])[:, head]/scale**2 for t in tasks], anchor)


def _solve_unequal_sites(likelihood, initial, bounds, optimizer):
    result = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True,
        method=optimizer['method'], bounds=bounds,
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'],
                 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(result.x)
    projected = np.where((result.x <= 0) & (gradient > 0), 0., gradient)
    for i, (lower, upper) in enumerate(bounds):
        if lower == upper:
            projected[i] = 0.
    return {'anchor': {**likelihood.anchor, 'signal_variance': float(result.x[0])},
        'residual_parameters': result.x[1:].tolist(),
        'source_features': likelihood.features.tolist(),
        'common_alpha': state['alpha'].tolist(), 'common_precision': state['precision'].tolist(),
        'source_label_count': likelihood.label_count, 'source_task_count': likelihood.task_count,
        'optimizer_converged': bool(result.success), 'optimizer_iterations': int(result.nit),
        'optimizer_max_iterations': optimizer['max_iterations'], 'optimizer_message': str(result.message),
        'optimizer_parameters': result.x.tolist(), 'optimizer_initial_parameters': initial.tolist(),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'source_negative_log_likelihood_per_label': value,
        'initial_source_negative_log_likelihood_per_label': likelihood.evaluate(initial)[0]}


def fit_unequal_site_kernel_population(tasks, head, anchor, *, optimizer):
    likelihood = _unequal_site_likelihood(tasks, head, anchor)
    common = anchor['signal_variance']
    initial = np.array([common, common, anchor['constant_variance'], anchor['discrepancy_variance']])
    return _solve_unequal_sites(likelihood, initial, [(common, common)]+[(0., None)]*3, optimizer)


def fit_joint_common_unequal_site_kernel_population(tasks, head, previous, *, optimizer):
    likelihood = _unequal_site_likelihood(tasks, head, previous['anchor'])
    return _solve_unequal_sites(likelihood, np.asarray(previous['optimizer_parameters']),
                               [(0., None)]*4, optimizer)


def predict_unequal_site_kernel_population(target_features, fit):
    """A new independent task, with shared interpolation error retained."""
    xs, xt = np.asarray(fit['source_features']), np.asarray(target_features)
    anchor = fit['anchor']; cross = shared_kernel(xt, xs, anchor)
    mean = anchor['constant']+cross @ np.asarray(fit['common_alpha'])
    covariance = shared_kernel(xt, xt, anchor)-cross @ np.asarray(fit['common_precision']) @ cross.T
    amplitude, intercept, discrepancy = fit['residual_parameters']
    covariance += (amplitude*rbf_cross(xt, xt, anchor['lengthscale'])+intercept
                   +discrepancy*np.eye(len(xt)))
    scale = anchor['normalization']['scale']
    return anchor['normalization']['mean']+scale*mean, scale**2*(covariance+covariance.T)/2
