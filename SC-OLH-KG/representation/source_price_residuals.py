"""Source-task price residual moments around a shared public reference.

One public-box variance contributes one reference task. Each source task
contributes its Gaussian conditional residual second moment, including price
estimation uncertainty. No absolute-price second moment or fixed .01 floor is
carried into this price-coordinate covariance. Target residual means stay zero.
This is a regularized moment estimate, not converged hierarchical Bayes.
"""

import numpy as np


def source_price_residual_covariance(design, responses, noises, offsets, bounds):
    design, offsets = np.asarray(design), np.asarray(offsets)
    center = np.full(design.shape[1], sum(bounds) / 2)
    parent_covariance = np.eye(len(center)) * (bounds[1] - bounds[0]) ** 2 / 12
    precision = np.linalg.inv(parent_covariance)
    reference = offsets + design @ center
    moment = parent_covariance.copy()
    conditional_means, conditional_covariances = [], []
    for labels, noise in zip(responses, noises):
        weighted = design / np.asarray(noise)[:, None]
        covariance = np.linalg.inv(precision + design.T @ weighted)
        mean = covariance @ weighted.T @ (np.asarray(labels) - reference)
        moment += covariance + np.outer(mean, mean)
        conditional_means.append(mean)
        conditional_covariances.append(covariance)
    covariance = moment / (len(responses) + 1)
    return covariance, {"mode": "public_center_source_residual_second_moment",
        "source_task_count": len(responses), "public_price_reference": center.tolist(),
        "public_reference_task_count": 1, "public_reference_covariance": parent_covariance.tolist(),
        "source_conditional_residual_means": np.asarray(conditional_means).tolist(),
        "source_conditional_price_covariances": np.asarray(conditional_covariances).tolist(),
        "source_mean_likelihood": "Gaussian working likelihood using recorded source mean-noise variances",
        "target_residual_mean": [0.] * len(center), "converged_hierarchical_bayes": False}


def pooled_residual_covariance(covariance):
    """Remove anisotropy and correlation at the same physical price trace."""
    return np.eye(len(covariance)) * np.trace(covariance) / len(covariance)
