"""Paid target evidence weighs a structural model against its full dictionary."""

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from representation.source_function_prior import SourceFunctionPrior


def full_dictionary_prior(prior, reference_psi):
    basis = type(prior.basis)(reference_psi)
    lookup = {name: j + 1 for j, name in enumerate(basis.names)}
    indices = [0] + [lookup[name] for name in prior.basis.names]
    mean = np.zeros(basis.feature_dim + 1)
    covariance = np.zeros((len(mean), len(mean)))
    mean[indices] = prior.mean
    covariance[np.ix_(indices, indices)] = prior.covariance
    return SourceFunctionPrior(basis, mean, covariance, prior.discrepancy_variance,
                               prior.diagnostics)


class TransferMixtureGPR:
    """Two independent Gaussian posteriors and their moment-matched acquisition.

    Every paid response updates both components once. Model weights use the
    pre-observation predictive density. The between-model covariance is kept.
    """

    def __init__(self, priors, public_candidates, reference_psi=None):
        reference = public_candidates if reference_psi is None else reference_psi
        expanded = [full_dictionary_prior(p, reference) for p in priors]
        self.models = [p.target_model(public_candidates) for p in expanded]
        self.moment = expanded[0].target_model(public_candidates)
        self.log_weights = np.full(2, -np.log(2.0))
        self.observations = 0
        self.refresh()

    def refresh(self):
        weights = np.exp(self.log_weights)
        a = sum(w * m.a for w, m in zip(weights, self.models))
        covariance = sum(w * (m.C + np.outer(m.a - a, m.a - a))
                         for w, m in zip(weights, self.models))
        self.moment.a, self.moment.C = a, covariance
        self.moment.lambda_i = float(sum(w * m.lambda_i for w, m in zip(weights, self.models)))
        self.moment.sampled_set = list(self.models[0].sampled_set)
        self.moment.sol_to_idx = dict(self.models[0].sol_to_idx)
        self.moment._invalidate_backend_cache()

    def update(self, x, y, noise):
        self.log_weights += np.array([norm.logpdf(y, m.posterior_mean(x),
                 np.sqrt(m.posterior_var(x) + noise)) for m in self.models])
        self.log_weights -= logsumexp(self.log_weights)
        for model in self.models:
            model.update(x, y, noise)
        self.observations += 1
        self.refresh()

    @property
    def a(self):
        return self.moment.a

    @property
    def C(self):
        return self.moment.C

    @property
    def lambda_i(self):
        return self.moment.lambda_i

    @property
    def sol_to_idx(self):
        return self.moment.sol_to_idx

    def augmented_feature_matrix(self, X):
        return self.moment.augmented_feature_matrix(X)

    def posterior_mean_many(self, X):
        return self.moment.posterior_mean_many(X)

    def posterior_var_many(self, X):
        return self.moment.posterior_var_many(X)

    def diagnostics(self):
        return {"structural_weight": float(np.exp(self.log_weights[0])),
                "full_dictionary_weight": float(np.exp(self.log_weights[1])),
                "paid_observations": self.observations}
