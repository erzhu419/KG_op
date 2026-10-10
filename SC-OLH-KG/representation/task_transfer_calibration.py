"""Source predictive calibration retaining native common and task spectra."""

import numpy as np

from representation.joint_function_risk import JointFunctionRiskPosterior


def scaled_task_diagonal(native, feature_dim, multipliers):
    intercept, function = multipliers
    tail = .01*native['source_signal_scale']/feature_dim
    original = np.asarray(native['residual_coefficient_diagonal'])
    return np.r_[intercept*original[0], tail+function*(original[1:]-tail)]


def calibrated_g_field(template, design, native, multipliers):
    """Keep the native common posterior and eta law; scale only task excess."""
    residual = scaled_task_diagonal(native, design.shape[1]-1, multipliers)
    mean, covariance = template.predict(full_cov=True)
    mean[:, 0] = design@np.asarray(native['coefficient_mean'])
    covariance[:, 0, :, 0] = design@(
        np.asarray(native['population_mean_covariance'])+np.diag(residual))@design.T
    covariance[:, 0, :, 0] += native['discrepancy_variance']*np.eye(len(design))
    return JointFunctionRiskPosterior.from_moments(mean, covariance)


def recondition_common(likelihood, native, multipliers):
    """Condition source common coefficients under the revised task law."""
    count = len(likelihood.basis.group_names)
    parameters = np.asarray(native['optimizer_parameters']).copy()
    parameters[count+1] *= multipliers[0]
    parameters[count+2:2*count+2] *= multipliers[1]
    value, _, state = likelihood.evaluate(parameters)
    return {'coefficient_mean': state['mean'].tolist(),
        'population_mean_covariance': state['mean_covariance'].tolist(),
        'residual_coefficient_diagonal': state['residual_diagonal'].tolist(),
        'discrepancy_variance': state['discrepancy'], 'source_signal_scale': likelihood.signal,
        'common_prior_coefficient_diagonal': state['common_diagonal'].tolist(),
        'common_group_variances': state['common_group_variances'].tolist(),
        'common_spectral_exponent': state['common_spectral_exponent'],
        'source_group_variances': state['group_variances'].tolist(),
        'spectral_exponent': state['spectral_exponent'],
        'source_negative_log_likelihood_per_label': value, 'conditioned_parameters': parameters.tolist(),
        'multipliers': list(multipliers), 'source_hyperparameters_reoptimized': False}
