"""Pointwise latent-risk diagnostics retaining Gaussian mixture moments."""
import numpy as np
from scipy.special import logsumexp, ndtr


def latent_risk_metrics(means, variances, weights, truth):
    weights, truth = np.asarray(weights), np.asarray(truth)
    mean = means @ weights
    within = variances @ weights
    between = ((means - mean[:, None]) ** 2) @ weights
    density = -.5 * ((truth[:, None] - means) ** 2 / variances
                      + np.log(2 * np.pi * variances))
    pit = ndtr((truth[:, None] - means) / np.sqrt(variances)) @ weights
    environmental_mean = np.exp(means + .5 * variances) @ weights
    return {
        "eta_rmse": float(np.sqrt(np.mean((mean - truth) ** 2))),
        "latent_log_score": float(np.mean(logsumexp(density + np.log(weights), axis=1))),
        "coverage90": float(np.mean((pit >= .05) & (pit <= .95))),
        "lower_miss": float(np.mean(pit < .05)),
        "upper_miss": float(np.mean(pit > .95)),
        "posterior_variance": float(np.mean(within + between)),
        "within_variance": float(np.mean(within)),
        "between_variance": float(np.mean(between)),
        "environmental_variance_prediction": float(environmental_mean.mean()),
        "environmental_variance_truth": float(np.exp(truth).mean()),
        "environmental_variance_bias": float(np.mean(environmental_mean - np.exp(truth))),
    }
