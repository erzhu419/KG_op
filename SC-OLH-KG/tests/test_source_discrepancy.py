from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from representation.task_posterior import (  # noqa: E402
    FiniteTaskModelEnsemble,
    FiniteTaskPosterior,
    TaskExpertState,
)


class FakeGPR:
    def __init__(self, mean):
        self.mean = float(mean)
        self.updates = 0

    def posterior_mean(self, x):
        del x
        return self.mean

    def posterior_var(self, x):
        del x
        return 0.02

    def update(self, x, y, variance):
        del x, y, variance
        self.updates += 1


class FakeVariance:
    def predict_variance(self, output_index, x, problem):
        del output_index, x, problem
        return 0.01

    def update(self, *args, **kwargs):
        del args, kwargs
        return {"status": "updated"}

    def diagnostics(self):
        return {"status": "fit"}


class FakeProblem:
    alpha = 0.05


def _ensemble(update):
    states = [
        TaskExpertState(
            name="matching",
            gpr_models=[FakeGPR(0.0), FakeGPR(0.0)],
            variance_model=FakeVariance(),
            problem=FakeProblem(),
        ),
        TaskExpertState(
            name="mismatching",
            gpr_models=[FakeGPR(3.0), FakeGPR(3.0)],
            variance_model=FakeVariance(),
            problem=FakeProblem(),
        ),
    ]
    posterior = FiniteTaskPosterior(
        [state.name for state in states],
        temperature=1.0,
        boundary_score_weight=1.0,
    )
    return FiniteTaskModelEnsemble(
        states,
        posterior,
        source_discrepancy_update=update,
    )


def test_source_discrepancy_posterior_downweights_mismatching_expert():
    ensemble = _ensemble(True)
    before = ensemble.posterior.posterior_weights()
    update = ensemble.update((1, 2), np.array([0.0, 0.0]), tau=0.5)
    after = ensemble.posterior.posterior_weights()
    assert after[0] > before[0]
    assert after[1] < before[1]
    assert update["source_discrepancy_update"] is True
    assert all(
        model.updates == 1
        for state in ensemble.states
        for model in state.gpr_models
    )


def test_frozen_discrepancy_control_still_updates_target_expert_models():
    ensemble = _ensemble(False)
    before = ensemble.posterior.posterior_weights()
    update = ensemble.update((1, 2), np.array([0.0, 0.0]), tau=0.5)
    after = ensemble.posterior.posterior_weights()
    np.testing.assert_allclose(after, before)
    assert update["posterior"]["status"] == "frozen_source_discrepancy"
    assert update["source_discrepancy_update"] is False
    assert all(
        model.updates == 1
        for state in ensemble.states
        for model in state.gpr_models
    )
