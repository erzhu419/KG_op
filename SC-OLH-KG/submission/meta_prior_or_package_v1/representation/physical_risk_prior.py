"""Finite empirical-Bayes prior for observable cumulative cost risk.

Particles are physical variance coefficients, not stored good policies.
Source tasks learn their mixture distribution. Paid target batches update it
with the exact Gaussian sample-variance likelihood. KG uses log-risk moments,
so its existing Gaussian fantasy approximation remains explicit.
"""

from dataclasses import dataclass
import itertools

import numpy as np
from scipy.special import digamma, gammaln, logsumexp, ndtri
from scipy.stats import qmc


@dataclass(frozen=True)
class RiskBatchData:
    X: np.ndarray
    sample_variance: np.ndarray
    replicates: int

    @classmethod
    def from_log_labels(cls, X, eta, replicates):
        df = replicates - 1
        bias = digamma(df / 2) + np.log(2 / df)
        return cls(np.asarray(X), np.exp(np.asarray(eta) + bias), replicates)


def variance_log_likelihood(task, coefficients):
    """Joint density of bias-corrected log sample variances, per particle."""
    variance = np.asarray(task.X) @ np.asarray(coefficients).T
    if np.any(variance <= 0):
        raise ValueError("physical risk design must give positive variance")
    s2 = np.asarray(task.sample_variance)[:, None]
    k = (task.replicates - 1) / 2
    # Includes the log-variance transformation Jacobian. The bias correction
    # is a translation, with unit Jacobian. No true variance labels are used.
    log_density = k * (np.log(k) + np.log(s2) - np.log(variance)) - k * s2 / variance - gammaln(k)
    return log_density.sum(axis=0)


def physical_particle_bank(reference, variance_scale, *, seed=105042, bits=9, log_sd=2.):
    """Three local and two shared channels, matching the Inventory interface.

    Eight declared factor-group masks include zero local/shared/volume risk.
    Positive floor and a correlation parameterization keep every covariance
    PSD. Scale uses pooled source variances; reference geometry uses no labels.
    """
    reference = np.asarray(reference)
    if reference.shape[1] != 9:
        raise ValueError("Inventory risk has nine physical coefficients")
    average = reference.mean(axis=0)
    z = ndtri(qmc.Sobol(9, scramble=True, seed=seed).random_base2(bits))
    floor = variance_scale / 4 * np.exp(log_sd * z[:, 0])
    local = variance_scale / (12 * average[1:4]) * np.exp(log_sd * z[:, 1:4])
    shared_sd = np.sqrt(variance_scale / (8 * average[[4, 6]])) * np.exp(.5 * log_sd * z[:, 4:6])
    correlation = np.tanh(z[:, 6])
    volume = variance_scale / (8 * average[7:9]) * np.exp(log_sd * z[:, 7:9])
    coefficients, families, names = [], [], []
    for group, (use_local, use_shared, use_volume) in enumerate(itertools.product((0, 1), repeat=3)):
        block = np.column_stack([floor, local * use_local,
            shared_sd[:, 0] ** 2 * use_shared,
            np.prod(shared_sd, axis=1) * correlation * use_shared,
            shared_sd[:, 1] ** 2 * use_shared, volume * use_volume])
        coefficients.append(block)
        families.extend([group] * len(block))
        names.append("+".join(name for name, used in zip(("local", "shared", "volume"),
                                                        (use_local, use_shared, use_volume)) if used) or "floor_only")
    return np.vstack(coefficients), np.array(families), names


@dataclass
class PhysicalRiskPrior:
    coefficients: np.ndarray
    weights: np.ndarray
    families: np.ndarray
    family_names: list
    diagnostics: dict

    def target_model(self):
        return PhysicalRiskPosterior(self)


def fit_source_physical_risk(tasks, reference, *, seed=105042, bits=9, log_sd=2.,
                             concentration=1., maxiter=2000, tolerance=1e-7):
    """Learn a finite task-law distribution by regularized marginal likelihood.

    One virtual source task distributes its mass over the pooled-scale bank.
    The supplied floor prevents a hard source veto. Each real source has its
    own latent risk coefficients, shared across its public planning contexts.
    """
    tasks = list(tasks)
    scale = float(np.mean([np.mean(task.sample_variance) for task in tasks]))
    coefficients, families, names = physical_particle_bank(reference, scale, seed=seed, bits=bits, log_sd=log_sd)
    q = np.full(len(coefficients), 1. / len(coefficients))
    likelihood = np.array([variance_log_likelihood(task, coefficients) for task in tasks])
    weights = q.copy()
    history, converged = [], False
    for iteration in range(maxiter):
        joint = likelihood + np.log(weights)
        responsibility = np.exp(joint - logsumexp(joint, axis=1)[:, None])
        updated = (responsibility.sum(axis=0) + concentration * q) / (len(tasks) + concentration)
        objective = float(logsumexp(likelihood + np.log(updated), axis=1).sum()
                          + concentration * q @ np.log(updated))
        history.append(objective)
        converged = bool(len(history) > 1 and
                         history[-1] - history[-2] < tolerance * max(1., abs(objective)))
        weights = updated
        if converged:
            break
    info = {"source_tasks": len(tasks), "source_labels": sum(len(task.X) for task in tasks),
            "source_variance_scale": scale, "source_optimizer_converged": converged,
            "source_optimizer_iterations": iteration + 1, "objective_history": history,
            "prior_mass_floor": concentration / (len(tasks) + concentration),
            "likelihood": "exact Gaussian log-sample-variance density",
            "representation": "public physical exposure functions; no learned encoder"}
    pooled = PhysicalRiskPrior(coefficients, q, families, names, {**info, "mode": "same_source_pooled_scale"})
    learned = PhysicalRiskPrior(coefficients, weights, families, names, {**info, "mode": "source_learned_physical_distribution"})
    return learned, pooled


def fit_source_risk_families(tasks, reference, *, seed=105042, bits=9, log_sd=2.,
                             concentration=1., maxiter=2000, tolerance=1e-7):
    """Learn seven factor-family weights; keep within-family laws uncertain.

    The same physical bank, pooled scale and exact likelihood are retained.
    Eight task laws cannot justify unrestricted individual-atom weights.
    """
    tasks = list(tasks)
    scale = float(np.mean([np.mean(task.sample_variance) for task in tasks]))
    coefficients, families, names = physical_particle_bank(reference, scale, seed=seed, bits=bits, log_sd=log_sd)
    atomic = np.array([variance_log_likelihood(task, coefficients) for task in tasks])
    likelihood = np.column_stack([logsumexp(atomic[:, families == group], axis=1)
                                   - np.log(np.sum(families == group)) for group in range(8)])
    q = np.full(8, 1. / 8)
    weights, history, converged = q.copy(), [], False
    for iteration in range(maxiter):
        joint = likelihood + np.log(weights)
        responsibility = np.exp(joint - logsumexp(joint, axis=1)[:, None])
        weights = (responsibility.sum(axis=0) + concentration * q) / (len(tasks) + concentration)
        objective = float(logsumexp(likelihood + np.log(weights), axis=1).sum()
                          + concentration * q @ np.log(weights))
        history.append(objective)
        converged = bool(len(history) > 1 and
                         history[-1] - history[-2] < tolerance * max(1., abs(objective)))
        if converged:
            break
    info = {"mode": "source_learned_factor_families", "source_tasks": len(tasks),
            "source_labels": sum(len(task.X) for task in tasks), "source_variance_scale": scale,
            "source_distribution_free_weights": 7, "within_family_distribution": "pooled_scale_reference_unchanged",
            "source_optimizer_converged": converged, "source_optimizer_iterations": iteration + 1,
            "objective_history": history, "likelihood": "exact Gaussian log-sample-variance density"}
    particle_weights = weights[families] / (2 ** bits)
    return PhysicalRiskPrior(coefficients, particle_weights, families, names, info)


class PhysicalRiskPosterior:
    def __init__(self, prior):
        self.prior = prior
        self.log_weights = np.log(prior.weights).copy()
        self.paid_batches = 0
        self.paid_episodes = 0

    def update_batch(self, task):
        self.log_weights += variance_log_likelihood(task, self.prior.coefficients)
        self.log_weights -= logsumexp(self.log_weights)
        self.paid_batches += len(task.X)
        self.paid_episodes += len(task.X) * task.replicates

    def predict(self, X, full_cov=False):
        eta = np.log(np.asarray(X) @ self.prior.coefficients.T)
        weights = np.exp(self.log_weights)
        mean = eta @ weights
        centered = eta - mean[:, None]
        covariance = (centered * weights) @ centered.T
        return mean, covariance if full_cov else np.maximum(np.diag(covariance), 0.)

    def log_predictive_density(self, task):
        return float(logsumexp(self.log_weights + variance_log_likelihood(task, self.prior.coefficients)))

    def diagnostics(self):
        weights = np.exp(self.log_weights)
        mean = weights @ self.prior.coefficients
        variance = weights @ (self.prior.coefficients - mean) ** 2
        return {"family_probabilities": {name: float(weights[self.prior.families == group].sum())
                                        for group, name in enumerate(self.prior.family_names)},
                "effective_particles": float(1. / (weights @ weights)),
                "max_particle_weight": float(weights.max()),
                "coefficient_mean": mean.tolist(), "coefficient_sd": np.sqrt(variance).tolist(),
                "paid_batches": self.paid_batches, "paid_episodes": self.paid_episodes}
