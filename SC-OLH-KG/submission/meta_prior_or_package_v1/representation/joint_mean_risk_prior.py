"""Gaussian coefficient means conditional on finite physical-risk particles.

The same paid batch supplies its sample mean and sample variance. They are
independent conditional on the true Gaussian mean and variance; posterior
coefficient means and risk particles generally become dependent. This model
has no independent per-policy discrepancy term.
"""

import numpy as np
from scipy.special import logsumexp

from core.verification_kg import particle_pass_probability
from representation.physical_risk_prior import variance_log_likelihood


class JointMeanRiskPosterior:
    def __init__(self, coefficient_mean, coefficient_covariance, risk_prior):
        self.risk_prior = risk_prior
        particles = len(risk_prior.weights)
        self.coefficient_mean = np.repeat(np.asarray(coefficient_mean)[None, :], particles, axis=0)
        self.coefficient_covariance = np.repeat(np.asarray(coefficient_covariance)[None, :, :], particles, axis=0)
        self.log_weights = np.log(risk_prior.weights).copy()
        self.paid_batches = 0
        self.paid_episodes = 0

    def update_batch(self, features, sample_means, risk_batch):
        """Update once, returning the pre-observation joint log density.

        Variance labels use the existing bias-corrected log-variance density,
        including its Jacobian. Mean likelihoods marginalize coefficient
        uncertainty and use each particle's episode variance divided by n.
        """
        features = np.atleast_2d(np.asarray(features))
        sample_means = np.atleast_1d(sample_means)
        if len(features) != len(sample_means) or len(features) != len(risk_batch.X):
            raise ValueError("Mean features, labels and risk batches must align")
        variances = np.asarray(risk_batch.X) @ self.risk_prior.coefficients.T
        self.log_weights += variance_log_likelihood(risk_batch, self.risk_prior.coefficients)
        for a, y, episode_variance in zip(features, sample_means, variances):
            u = self.coefficient_covariance @ a
            predicted_mean = self.coefficient_mean @ a
            label_variance = u @ a + episode_variance / risk_batch.replicates
            error = y - predicted_mean
            self.log_weights += -.5 * (np.log(2 * np.pi * label_variance) + error ** 2 / label_variance)
            self.coefficient_mean += u * (error / label_variance)[:, None]
            self.coefficient_covariance -= u[:, :, None] * u[:, None, :] / label_variance[:, None, None]
        evidence = float(logsumexp(self.log_weights))
        self.log_weights -= evidence
        self.paid_batches += len(features)
        self.paid_episodes += len(features) * risk_batch.replicates
        return evidence

    def conditional_moments(self, features):
        features = np.atleast_2d(np.asarray(features))
        mean = features @ self.coefficient_mean.T
        variance = np.einsum("kp,mpq,kq->km", features, self.coefficient_covariance, features, optimize=True)
        return mean, np.maximum(variance, 0.)

    def predict(self, features, risk_design, full_cov=False):
        """Return joint (g, eta) latent moments; full covariance is (K,2,K,2)."""
        features = np.atleast_2d(np.asarray(features))
        conditional, conditional_variance = self.conditional_moments(features)
        eta = np.log(np.asarray(risk_design) @ self.risk_prior.coefficients.T)
        weights = np.exp(self.log_weights)
        means = np.column_stack([conditional @ weights, eta @ weights])
        g_centered, eta_centered = conditional - means[:, 0, None], eta - means[:, 1, None]
        if not full_cov:
            variance = np.column_stack([(conditional_variance + g_centered ** 2) @ weights,
                                        eta_centered ** 2 @ weights])
            return means, variance
        average_covariance = np.einsum("m,mpq->pq", weights, self.coefficient_covariance)
        g_cov = features @ average_covariance @ features.T + (g_centered * weights) @ g_centered.T
        eta_cov = (eta_centered * weights) @ eta_centered.T
        cross_cov = (g_centered * weights) @ eta_centered.T
        covariance = np.empty((len(features), 2, len(features), 2))
        covariance[:, 0, :, 0], covariance[:, 1, :, 1] = g_cov, eta_cov
        covariance[:, 0, :, 1], covariance[:, 1, :, 0] = cross_cov, cross_cov.T
        return means, covariance

    def verifier_probability(self, features, risk_design, protocol):
        mean, variance = self.conditional_moments(features)
        risks = np.asarray(risk_design) @ self.risk_prior.coefficients.T
        return particle_pass_probability(mean, variance, risks, np.exp(self.log_weights), protocol)

    def diagnostics(self):
        weights = np.exp(self.log_weights)
        return {"effective_particles": float(1. / (weights @ weights)),
                "paid_batches": self.paid_batches, "paid_episodes": self.paid_episodes,
                "coefficient_mean": (weights @ self.coefficient_mean).tolist()}
