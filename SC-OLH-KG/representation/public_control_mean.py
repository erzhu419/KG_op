"""Public control-schema coordinates and head-specific source function moments."""

import numpy as np


def public_control_mean_coordinates(problem, points):
    return np.asarray([np.r_[problem.observable_state_exposure(p).channel_means,
                             np.std(problem.normalize(p))] for p in points])


def source_prior_moments(prior, psi):
    design = np.column_stack([np.ones(len(psi)), prior.basis.transform(psi)])
    covariance = design @ prior.covariance @ design.T+prior.discrepancy_variance*np.eye(len(psi))
    return design @ prior.mean, (covariance+covariance.T)/2
