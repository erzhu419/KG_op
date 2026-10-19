"""Coupled query selection and independent-point terminal recommendation.

Both channels start after the same complete initialization and receive each
new paired observation once. Only one physical episode is charged. Selection
retains cross-policy propagation; the terminal channel updates its own point.
"""

from copy import deepcopy

from performance.mainline_moment_policy import moment_state
from performance.mainline_point_only import PointOnlyResponseMixture


class DualChannelResponseMixture:
    @classmethod
    def from_model(cls, model):
        result = cls()
        result.selection_model = model
        result.terminal_model = PointOnlyResponseMixture.from_model(model)
        result.names = list(model.names)
        result.candidate_count = result.terminal_model.candidate_count
        result.history = deepcopy(model.history)
        return result

    @property
    def paid_episodes(self):
        assert self.selection_model.paid_episodes == self.terminal_model.paid_episodes
        return self.selection_model.paid_episodes

    @property
    def weights(self):
        return self.terminal_model.weights

    @property
    def selection_weights(self):
        return self.selection_model.weights

    def query_state(self):
        return moment_state(self.selection_model)

    def moment_state(self):
        return self.terminal_model.moment_state()

    def update_raw(self, candidate, values):
        before = self.paid_episodes
        shared_values = tuple(map(float, values))
        selection = self.selection_model.update_raw(candidate, shared_values)
        terminal = self.terminal_model.update_raw(candidate, shared_values)
        assert self.paid_episodes == before+1
        record = {'candidate': int(candidate), 'replicates': 1,
            'paid_episodes': self.paid_episodes, 'shared_raw_values': list(shared_values),
            'selection_evidence': selection, 'terminal_evidence': terminal,
            'dual_channel_update': True}
        self.history.append(record)
        return record


class FrozenQueryDualChannelResponseMixture:
    """Keep initialized coupled selection moments; update own-point terminal.

    Selection needs only the aggregate native moments and exact episode noise,
    so the full mixture is released after extracting the terminal marginals.
    """

    @classmethod
    def from_model(cls, model):
        result = cls()
        result._selection_state = moment_state(model)
        result._selection_weights = model.weights.copy()
        result.terminal_model = PointOnlyResponseMixture.from_model(model)
        result.names = list(model.names)
        result.candidate_count = result.terminal_model.candidate_count
        result.history = deepcopy(model.history)
        return result

    @property
    def paid_episodes(self):
        return self.terminal_model.paid_episodes

    @property
    def weights(self):
        return self.terminal_model.weights

    @property
    def selection_weights(self):
        return self._selection_weights.copy()

    def query_state(self):
        return {key: value.copy() for key, value in self._selection_state.items()}

    def moment_state(self):
        return self.terminal_model.moment_state()

    def update_raw(self, candidate, values):
        before = self.paid_episodes
        shared_values = tuple(map(float, values))
        terminal = self.terminal_model.update_raw(candidate, shared_values)
        assert self.paid_episodes == before + 1
        record = {'candidate': int(candidate), 'replicates': 1,
            'paid_episodes': self.paid_episodes, 'shared_raw_values': list(shared_values),
            'selection_updated': False, 'terminal_evidence': terminal,
            'frozen_query_update': True}
        self.history.append(record)
        return record
