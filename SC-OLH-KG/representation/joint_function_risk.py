"""Joint g/eta filtering for a finite source-learned function prior.

The Gaussian batch likelihood is exact; each component's posterior is projected
to its first two moments after each paid batch. Full cross-candidate and g/eta
covariance is retained. This is not an exact multibatch posterior or a physical
risk-factor model.
"""

import itertools

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import digamma, logsumexp

from representation.physical_risk_prior import log_sample_variance_density


def local_joint_moments(mean, covariance, g_label, eta_label, replicates):
    """Condition on one raw value, or a replicated mean and variance label.

    A single episode has no sample variance: eta_label must be None. Its
    Normal likelihood still informs latent eta through the observed residual.
    """
    if replicates < 1 or int(replicates) != replicates:
        raise ValueError("replicates must be a positive integer")
    if (replicates == 1) != (eta_label is None):
        raise ValueError("one episode has no eta label; batches require one")
    mean, covariance = np.asarray(mean), np.asarray(covariance)
    gm, em = mean
    gv, ev, cross = covariance[0, 0], covariance[1, 1], covariance[0, 1]
    conditional_variance = max(float(gv - cross ** 2 / ev), 0.)
    loading = cross / ev
    sample_variance = None
    if replicates > 1:
        k = (replicates - 1) / 2
        sample_variance = float(np.exp(eta_label + digamma(k) - np.log(k)))

    def at_eta(eta):
        observation_variance = np.exp(eta) / replicates
        prior_mean = gm + loading * (eta - em)
        total = conditional_variance + observation_variance
        error = g_label - prior_mean
        posterior_mean = prior_mean + conditional_variance / total * error
        posterior_variance = conditional_variance * observation_variance / total
        log_density = -.5 * (np.log(2 * np.pi * ev) + (eta - em) ** 2 / ev)
        if sample_variance is not None:
            log_density += log_sample_variance_density(eta, sample_variance, replicates)
        log_density += -.5 * (np.log(2 * np.pi * total) + error ** 2 / total)
        return log_density, posterior_mean, posterior_variance

    lo, hi = em - 12 * np.sqrt(ev), em + 12 * np.sqrt(ev)
    mode = minimize_scalar(lambda eta: -at_eta(eta)[0], bounds=(lo, hi), method="bounded").x
    shift = at_eta(mode)[0]

    def integrand(eta):
        density, posterior_mean, posterior_variance = at_eta(eta)
        dg, de = posterior_mean - gm, eta - em
        return np.exp(density - shift) * np.array([1., dg, posterior_variance + dg ** 2,
                                                  de, de ** 2, dg * de])

    integral, error = quad_vec(integrand, lo, hi, points=[mode], epsabs=1e-9, epsrel=1e-9)
    statistics = integral / integral[0]
    displacement = statistics[[1, 3]]
    posterior_covariance = np.array([[statistics[2], statistics[5]],
                                    [statistics[5], statistics[4]]]) - np.outer(displacement, displacement)
    return {"mean": mean + displacement, "covariance": posterior_covariance,
            "log_evidence": float(shift + np.log(integral[0])),
            "integration_relative_error": float(error / integral[0])}


def joint_batch_log_score(components, g_label, eta_label, replicates):
    """Score the same exact observation density for any Gaussian mixture."""
    return float(logsumexp([np.log(weight) + local_joint_moments(mean, covariance,
        g_label, eta_label, replicates)["log_evidence"] for weight, mean, covariance in components]))


class JointFunctionRiskPosterior:
    @classmethod
    def from_moments(cls, mean, covariance):
        """Initialize a kernel mean/log-variance field for the same raw filter.

        The moments describe latent functions, without fresh episode noise.
        Candidate/head order is exactly that returned by predict(full_cov=True).
        """
        mean, covariance = np.asarray(mean, dtype=float), np.asarray(covariance, dtype=float)
        n = len(mean)
        if mean.shape != (n, 2) or covariance.shape != (n, 2, n, 2):
            raise ValueError('kernel moments must use candidate/head/candidate/head order')
        result = cls.__new__(cls)
        result.candidate_count = n
        result.component_pairs = [(0, 0)]
        result.g_structural = result.eta_structural = [False]
        result.means = mean.reshape(1, 2*n).copy()
        result.covariances = covariance.reshape(1, 2*n, 2*n).copy()
        result.log_weights = np.zeros(1)
        result.paid_batches = result.paid_episodes = 0
        result.max_integration_relative_error = 0.
        return result

    def __init__(self, g_priors, eta_priors, candidate_psi):
        self.candidate_count = len(candidate_psi)
        n = self.candidate_count
        self.component_pairs = list(itertools.product(range(len(g_priors)), range(len(eta_priors))))
        self.g_structural = [p.diagnostics.get("mode") in {"source_orthogonal_group_spectral", "source_orthogonal_inventory_support"} for p in g_priors]
        self.eta_structural = [p.diagnostics.get("mode") == "source_orthogonal_group_spectral" for p in eta_priors]
        means, covariances = [], []
        for gi, ei in self.component_pairs:
            mean = np.empty(2 * n)
            covariance = np.zeros((2 * n, 2 * n))
            for head, prior in enumerate((g_priors[gi], eta_priors[ei])):
                design = np.column_stack([np.ones(n), prior.basis.transform(candidate_psi)])
                mean[head::2] = design @ prior.mean
                covariance[head::2, head::2] = (design @ prior.covariance @ design.T
                                             + prior.discrepancy_variance * np.eye(n))
            means.append(mean)
            covariances.append(covariance)
        self.means, self.covariances = np.array(means), np.array(covariances)
        self.log_weights = np.full(len(means), -np.log(len(means)))
        self.paid_batches = self.paid_episodes = 0
        self.max_integration_relative_error = 0.

    def local_components(self, candidate):
        indices = [2 * candidate, 2 * candidate + 1]
        return [(float(np.exp(weight)), mean[indices], covariance[np.ix_(indices, indices)])
                for weight, mean, covariance in zip(self.log_weights, self.means, self.covariances)]

    def update_batch(self, candidate, g_label, eta_label, replicates):
        if len(self.means) > 1:
            from representation.vectorized_joint_function_risk import update_raw_observation
            return update_raw_observation(self, candidate, g_label, eta_label, replicates)
        indices = [2 * candidate, 2 * candidate + 1]
        evidence = []
        for component in range(len(self.means)):
            mean, covariance = self.means[component], self.covariances[component]
            point_covariance = covariance[np.ix_(indices, indices)]
            result = local_joint_moments(mean[indices], point_covariance, g_label, eta_label, replicates)
            gain = np.linalg.solve(point_covariance, covariance[indices, :]).T
            # Complete the local moments using the unchanged Gaussian
            # conditional distribution of all other candidate functions.
            self.means[component] = mean + gain @ (result["mean"] - mean[indices])
            self.covariances[component] = covariance + gain @ (
                result["covariance"] - point_covariance) @ gain.T
            self.covariances[component] = .5 * (self.covariances[component] + self.covariances[component].T)
            evidence.append(result["log_evidence"])
            self.max_integration_relative_error = max(self.max_integration_relative_error,
                                                      result["integration_relative_error"])
        self.log_weights += evidence
        log_evidence = float(logsumexp(self.log_weights))
        self.log_weights -= log_evidence
        self.paid_batches += 1
        self.paid_episodes += replicates
        return log_evidence

    def predict(self, full_cov=False):
        weights = np.exp(self.log_weights)
        mean = weights @ self.means
        centered = self.means - mean
        covariance = (np.einsum("m,mij->ij", weights, self.covariances)
                      + (centered.T * weights) @ centered)
        if full_cov:
            return mean.reshape(-1, 2), covariance.reshape(self.candidate_count, 2, self.candidate_count, 2)
        return mean.reshape(-1, 2), np.diag(covariance).reshape(-1, 2)

    def diagnostics(self):
        weights = np.exp(self.log_weights)
        return {"paid_batches": self.paid_batches, "paid_episodes": self.paid_episodes,
                "component_weights": weights.tolist(),
                "g_structural_weight": float(sum(w for w, pair in zip(weights, self.component_pairs)
                                                 if self.g_structural[pair[0]])),
                "eta_structural_weight": float(sum(w for w, pair in zip(weights, self.component_pairs)
                                                   if self.eta_structural[pair[1]])),
                "max_integration_relative_error": self.max_integration_relative_error,
                "likelihood": "Normal raw observation; replicated batches also use log-chi-square sample variance",
                "posterior_approximation": "Gaussian moment projection after each paid batch"}
