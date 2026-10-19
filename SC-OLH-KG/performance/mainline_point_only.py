"""Product of native single-policy marginals after shared initialization.

Each policy owns its complete source identity mixture and both native
(response, log-variance) mixtures. A new paired observation updates that
policy only. The shared initialization has already informed every marginal.
"""

from copy import deepcopy

import numpy as np

from representation.joint_function_risk import JointFunctionRiskPosterior
from representation.joint_response_mixture import JointResponseMixture


def _single_point_field(field, candidate):
    index = slice(2*candidate, 2*candidate+2)
    means = field.means[:, index].copy()
    covariances = field.covariances[:, index, index].copy()
    result = JointFunctionRiskPosterior.from_moments(
        means[0].reshape(1, 2), covariances[0].reshape(1, 2, 1, 2))
    result.means, result.covariances = means, covariances
    result.log_weights = field.log_weights.copy()
    result.component_pairs = list(field.component_pairs)
    result.g_structural = list(field.g_structural)
    result.eta_structural = list(field.eta_structural)
    result.paid_batches = field.paid_batches
    result.paid_episodes = field.paid_episodes
    result.max_integration_relative_error = field.max_integration_relative_error
    return result


class PointOnlyResponseMixture:
    @classmethod
    def from_model(cls, model):
        """Copy only local blocks; retain no reference to the full covariance."""
        result = cls()
        result.names = list(model.names)
        result.candidate_count = model.components[0][0].candidate_count
        result.paid_episodes = model.paid_episodes
        result.history = deepcopy(model.history)
        result.local_models = []
        for candidate in range(result.candidate_count):
            pairs = [tuple(_single_point_field(field, candidate) for field in pair)
                     for pair in model.components]
            local = JointResponseMixture(model.names, pairs)
            local.log_weights = model.log_weights.copy()
            local.paid_episodes = model.paid_episodes
            result.local_models.append(local)
        result._means = np.empty((result.candidate_count, 4))
        result._point_covariances = np.empty((result.candidate_count, 4, 4))
        result._noise = np.empty((result.candidate_count, 2))
        for candidate in range(result.candidate_count):
            result._refresh(candidate)
        return result

    @property
    def weights(self):
        return np.asarray([local.weights for local in self.local_models])

    def _refresh(self, candidate):
        local = self.local_models[candidate]
        mean, covariance = local.predict_joint()
        self._means[candidate] = mean[0]
        self._point_covariances[candidate] = covariance[0, :, 0, :]
        noise = np.zeros(2)
        for identity_weight, pair in zip(local.weights, local.components):
            for head, field in enumerate(pair):
                eta_mean, eta_variance = field.means[:, 1], field.covariances[:, 1, 1]
                noise[head] += identity_weight*(np.exp(field.log_weights)
                    @ np.exp(eta_mean+eta_variance/2))
        self._noise[candidate] = noise

    def moment_state(self):
        covariance = np.zeros((self.candidate_count, 4, self.candidate_count, 4))
        indices = np.arange(self.candidate_count)
        covariance[indices, :, indices, :] = self._point_covariances
        return {'mean': self._means.copy(), 'cov': covariance, 'noise': self._noise.copy()}

    def update_raw(self, candidate, values):
        local_record = self.local_models[candidate].update_raw(0, values)
        record = dict(local_record, candidate=int(candidate))
        self.history.append(record)
        self.paid_episodes += 1
        self._refresh(candidate)
        return record
