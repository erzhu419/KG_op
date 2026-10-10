"""Separate public-bank task level from shape in the original dictionary."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.orthogonal_function_prior import spectral_weights
from representation.source_joint_population import JointCovariancePopulationLikelihood


def task_coefficient_covariance(diagonal, center):
    transform = np.eye(len(diagonal)); transform[0, 1:] = -np.asarray(center)
    return (transform*np.asarray(diagonal))@transform.T


class CenteredTaskPopulationLikelihood(JointCovariancePopulationLikelihood):
    """Common design D is unchanged; independent task effects use centered Z."""

    def __init__(self, prior, batches, observation_variances, bank_centers):
        super().__init__(prior, batches, observation_variances)
        self.task_designs = []
        for (design, _, _), center in zip(self.blocks, bank_centers):
            task_design = design.copy(); task_design[:, 1:] -= center
            self.task_designs.append(task_design)

    def evaluate_with_common_precision(self, parameters, precision0, logdet0):
        groups = self.signal*np.asarray(parameters[1:-2])
        weights, derivative = spectral_weights(self.basis, parameters[-2])
        diagonal = np.r_[self.signal*parameters[0], groups[self.basis.group_index]*weights+self.tail]
        delta = self.signal*parameters[-1]
        precision, rhs = precision0.copy(), np.zeros(len(self.prior.mean))
        logdet, solved = logdet0, []
        for (design, error, noise), task_design in zip(self.blocks, self.task_designs):
            variance = (task_design*diagonal)@task_design.T+np.diag(noise+delta)
            factor = cho_factor(variance, lower=True)
            joined = cho_solve(factor, np.column_stack([design, error, np.eye(len(error))]))
            size = design.shape[1]; S, W = joined[:, :size], joined[:, size+1:]
            precision += design.T@S; rhs += design.T@joined[:, size]
            logdet += 2*np.log(np.diag(factor[0])).sum()
            solved.append((design, task_design, error, S, W))
        factor = cho_factor(precision, lower=True)
        displacement = cho_solve(factor, rhs)
        common_covariance = cho_solve(factor, np.eye(len(rhs)))
        logdet += 2*np.log(np.diag(factor[0])).sum()
        quadratic = float(displacement@precision0@displacement)
        feature_gradient, delta_gradient = np.zeros(len(rhs)), 0.
        for design, task_design, error, S, W in solved:
            alpha = W@(error-design@displacement)
            quadratic += (error-design@displacement)@alpha
            cross = S.T@task_design
            feature_gradient += .5*(np.sum(task_design*(W@task_design), axis=0)
                -np.sum(cross*(common_covariance@cross), axis=0)-(task_design.T@alpha)**2)
            delta_gradient += .5*(np.trace(W)-np.sum((S@common_covariance)*S)-alpha@alpha)
        gradient = np.r_[self.signal*feature_gradient[0],
            self.signal*np.bincount(self.basis.group_index, weights=feature_gradient[1:]*weights,
                minlength=len(self.basis.group_names)),
            feature_gradient[1:]@(groups[self.basis.group_index]*derivative), self.signal*delta_gradient]/self.label_count
        value = .5*(self.label_count*np.log(2*np.pi)+logdet+quadratic)/self.label_count
        return float(value), gradient, {'mean': self.prior.mean+displacement,
            'mean_covariance': common_covariance, 'residual_diagonal': diagonal,
            'discrepancy': float(delta), 'group_variances': groups,
            'spectral_exponent': float(parameters[-2])}


def fit_centered_task_population(likelihood, native, optimizer):
    count = len(likelihood.basis.group_names)
    initial = np.asarray(native['optimizer_parameters'])
    initial_value = likelihood.evaluate(initial)[0]
    fitted = minimize(lambda p: likelihood.evaluate(p)[:2], initial, jac=True, method=optimizer['method'],
        bounds=[(0., None)]*count+[(0., 4.)]+[(0., None)]*(count+1)+[(0., 4.), (0., None)],
        options={'maxiter': optimizer['max_iterations'], 'ftol': optimizer['ftol'], 'gtol': optimizer['gtol']})
    value, gradient, state = likelihood.evaluate(fitted.x)
    upper = np.full(len(fitted.x), np.inf); upper[count] = upper[-2] = 4.
    projected = np.where(((fitted.x <= 0) & (gradient > 0)) | ((fitted.x >= upper) & (gradient < 0)), 0., gradient)
    return {'population_model': 'original common function and public-bank-centered independent task level/shape',
        'coefficient_mean': state['mean'].tolist(), 'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(),
        'discrepancy_variance': state['discrepancy'], 'source_signal_scale': likelihood.signal,
        'source_group_variances': state['group_variances'].tolist(), 'spectral_exponent': state['spectral_exponent'],
        'common_prior_coefficient_diagonal': state['common_diagonal'].tolist(),
        'common_group_variances': state['common_group_variances'].tolist(),
        'common_spectral_exponent': state['common_spectral_exponent'],
        'source_label_count': likelihood.label_count, 'source_task_count': len(likelihood.blocks),
        'source_negative_log_likelihood_per_label': value,
        'initial_centered_source_negative_log_likelihood_per_label': initial_value,
        'native_uncentered_source_negative_log_likelihood_per_label': native['source_negative_log_likelihood_per_label'],
        'optimizer_parameters': fitted.x.tolist(), 'optimizer_initial_parameters': initial.tolist(),
        'optimizer_converged': bool(fitted.success), 'optimizer_message': str(fitted.message),
        'optimizer_iterations': int(fitted.nit), 'optimizer_evaluations': int(fitted.nfev),
        'optimizer_projected_gradient_max': float(np.max(np.abs(projected))),
        'optimizer': optimizer, 'tail_variance_budget': likelihood.basis.feature_dim*likelihood.tail,
        'query_task_covariance_requires_public_bank_center': True}


def centered_g_field(template, design, law, center):
    mean, covariance = template.predict(full_cov=True)
    task_covariance = task_coefficient_covariance(law['residual_coefficient_diagonal'], center)
    coefficient_covariance = np.asarray(law['population_mean_covariance'])+task_covariance
    mean[:, 0] = design@np.asarray(law['coefficient_mean'])
    covariance[:, 0, :, 0] = design@coefficient_covariance@design.T
    covariance[:, 0, :, 0] += law['discrepancy_variance']*np.eye(len(design))
    return JointFunctionRiskPosterior.from_moments(mean, covariance)
