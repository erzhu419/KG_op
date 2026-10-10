"""Empirical meta-prior retaining between-archive functional covariances."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from representation.source_function_prior import SourceFunctionPrior


def align_prior(prior, basis):
    """Exact affine conversion, including intercept/function covariance."""
    if prior.basis.names != basis.names:
        raise ValueError("archive components must use the same public dictionary")
    transform = np.diag(np.r_[1., basis.scale / prior.basis.scale])
    transform[0, 1:] = (basis.mean - prior.basis.mean) / prior.basis.scale
    return SourceFunctionPrior(basis, transform @ prior.mean,
        transform @ prior.covariance @ transform.T,
        prior.discrepancy_variance, prior.diagnostics)


@dataclass
class ArchiveCovariancePrior:
    components: list
    archive_keys: list

    @classmethod
    def from_contexts(cls, contexts, bases, excluded_seeds, held_task):
        eligible = [c for c in contexts if c["seed"] not in excluded_seeds and c["held_task"] == held_task]
        components, keys = [], []
        for c in eligible:
            basis = bases[c["seed"]]
            components.append(SourceFunctionPrior(basis,
                np.r_[c["eta_intercept_mean"], np.zeros(basis.feature_dim)],
                np.diag([.25] + c["eta_prior_function_variance"]),
                c["eta_discrepancy_variance"], {"mode": "archive_covariance_component"}))
            keys.append((c["seed"], c["archive_group"]))
        return cls(components, keys)

    @property
    def weights(self):
        return np.full(len(self.components), 1. / len(self.components))


def prequential_components(priors, psi, ids, labels, noise):
    """One Cholesky per component; each prefix uses only its past labels.

    Equal candidate IDs share the function discrepancy. Individual measurement
    noises are independent, including when the same candidate is queried again.
    """
    ids, labels, noise = np.asarray(ids), np.asarray(labels), np.asarray(noise)
    evidence, density, means, variances = [], [], [], []
    for prior in priors:
        design = np.column_stack((np.ones(len(ids)), prior.basis.transform(psi[ids])))
        covariance = design @ prior.covariance @ design.T
        covariance += prior.discrepancy_variance * (ids[:, None] == ids[None, :])
        covariance += np.diag(noise)
        lower = np.linalg.cholesky(covariance)
        prior_mean = design @ prior.mean
        residual = solve_triangular(lower, labels - prior_mean, lower=True)
        log_density = -.5 * (residual**2 + np.log(2 * np.pi)) - np.log(np.diag(lower))
        evidence.append(np.r_[0., np.cumsum(log_density)[:-1]])
        density.append(log_density)
        means.append(prior_mean + np.array([lower[t, :t] @ residual[:t] for t in range(len(ids))]))
        variances.append(np.maximum(np.diag(lower)**2 - noise, 0.))
    return {"log_evidence": np.array(evidence).T, "log_density": np.array(density).T,
            "mean": np.array(means).T, "variance": np.array(variances).T}


def mixture_forecasts(log_evidence, log_density, weights):
    with np.errstate(divide="ignore"):
        logs = np.log(weights)
    posterior = log_evidence + logs
    posterior -= logsumexp(posterior, axis=1, keepdims=True)
    return logsumexp(posterior + log_density, axis=1), np.exp(posterior)


def joint_archive_forecasts(risk, past_auxiliary_evidence, archive_weights):
    """Share archive identity across heads; risk/full odds remain eta-specific.

    Risk columns are (structural, full) pairs in archive order. Auxiliary
    evidence has one column per archive and includes strictly earlier f/g
    observations only. The current risk label occurs in log_density, for
    scoring, and never in component posterior weights.
    """
    weights = np.repeat(np.asarray(archive_weights) / 2., 2)
    eta_e, eta_p = risk["log_evidence"], risk["log_density"]
    joint_e = eta_e + np.repeat(past_auxiliary_evidence, 2, axis=1)
    scores, posteriors = {}, {}
    scores["J"], posteriors["J"] = mixture_forecasts(joint_e, eta_p, weights)
    scores["A"], posteriors["A"] = mixture_forecasts(eta_e, eta_p, weights)
    scores["G"], posteriors["G"] = mixture_forecasts(
        joint_e[:, 1::2], eta_p[:, 1::2], archive_weights)
    return scores, posteriors


def sharing_branches(risk, past_auxiliary_evidence, archive_weights):
    """Shared archive versus independent archives for f/g and eta.

    The separate branch integrates f/g and eta archive identities separately;
    it retains the same archive pool and risk structural/full weights.
    """
    scores, risk_posteriors = joint_archive_forecasts(
        risk, past_auxiliary_evidence, archive_weights)
    with np.errstate(divide="ignore"):
        archive_logs = np.log(archive_weights)
        risk_logs = np.log(np.repeat(np.asarray(archive_weights) / 2., 2))
    eta_e = risk["log_evidence"] + risk_logs
    auxiliary = past_auxiliary_evidence + archive_logs
    shared_e = logsumexp(eta_e + np.repeat(past_auxiliary_evidence, 2, axis=1), axis=1)
    separate_e = logsumexp(eta_e, axis=1) + logsumexp(auxiliary, axis=1)
    return {"log_evidence": np.column_stack((shared_e, separate_e)),
            "log_density": np.column_stack((scores["J"], scores["A"])),
            "scores": scores, "risk_posteriors": risk_posteriors}


def sharing_forecasts(branches, sharing_prior=.5):
    """Integrate the sharing state using only its past joint evidence.

    For the uniform Beta reference, sharing_prior is its mean 1/2. Each task
    has one sharing state; integrating the probability before that state is
    drawn gives these same two prior masses.
    """
    scores, sharing = mixture_forecasts(branches["log_evidence"],
        branches["log_density"], [sharing_prior, 1. - sharing_prior])
    risk = sharing[:, :1] * branches["risk_posteriors"]["J"] + \
           sharing[:, 1:] * branches["risk_posteriors"]["A"]
    return scores, sharing, risk
