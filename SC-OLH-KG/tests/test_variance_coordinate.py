from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cumulative_risk import cumulative_feature_vector  # noqa: E402
from problems.rzdt import make_problem  # noqa: E402
from problems.single_objective import ScalarizedProblem  # noqa: E402
from representation.meta_prior import (  # noqa: E402
    LearnedMetaPrior,
    MetaPriorProblemAdapter,
)
from representation.observable_exposure import (  # noqa: E402
    canonical_observable_state_descriptor,
)
from representation.variance_coordinate import (  # noqa: E402
    SourceAlignedVarianceRiskCoordinate,
)


def _problem(name, d=12):
    return ScalarizedProblem(make_problem(name, d=d, L=100, sigma=0.04))


def test_variance_head_is_nonnegative_reproducible_and_fixed_dimensional():
    problems = (
        _problem("FactorShockStatePolicyRZDT1"),
        _problem("InventorySupplyChain"),
    )
    inputs = []
    variance = []
    domains = []
    for domain_index, problem in enumerate(problems):
        rng = np.random.default_rng(700 + domain_index)
        for _ in range(20):
            x = problem.sample_random(rng)
            inputs.append(canonical_observable_state_descriptor(
                problem.observable_state_exposure(x)))
            variance.append(float(problem.true_sigma(x)[1]) ** 2)
            domains.append(problem.problem_name)
    model = SourceAlignedVarianceRiskCoordinate(
        local_dim=3,
        shared_dim=2,
    ).fit(inputs, variance, domains)
    first = model.risk_exposure_from_descriptor(inputs[0])
    second = model.risk_exposure_from_descriptor(inputs[0])
    assert first.A.shape == (3,)
    assert first.N.shape == (2,)
    assert np.all(first.A >= 0.0)
    assert np.all(first.N >= 0.0)
    assert np.isclose(np.sum(first.N), 1.0)
    np.testing.assert_allclose(first.A, second.A)
    np.testing.assert_allclose(first.N, second.N)
    assert cumulative_feature_vector(first).shape == (1 + 3 + 3 + 2,)


def test_meta_prior_uses_one_observable_input_with_independent_heads():
    sources = [
        ("FactorShockStatePolicyRZDT1", _problem(
            "FactorShockStatePolicyRZDT1")),
        ("InventorySupplyChain", _problem("InventorySupplyChain")),
    ]
    prior = LearnedMetaPrior(
        local_dim=3,
        shared_dim=2,
        component_stage="legacy_all",
        observable_mean_coordinate=True,
        observable_mean_mode="boundary_aligned",
        observable_mean_training_target="chance_margin",
        observable_mean_input_mode="observable_state_exposure",
        observable_mean_latent_dim=3,
        observable_variance_input_mode="observable_state_exposure",
        source_observation_mode="replicated",
        source_observation_replicates=3,
        source_design_mode="universal_mixture",
        source_universal_fraction=1.0,
        teacher_records_per_domain=0,
        seed=719,
    ).fit_from_source_problems(
        sources,
        n_records_per_domain=20,
        rng=np.random.default_rng(719),
    )
    target = MetaPriorProblemAdapter(_problem("QueueResourceControl", d=100), prior)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("target truth/provider was called")

    target.base.true_objective = forbidden
    target.base.true_constraint_mean = forbidden
    target.base.true_sigma = forbidden
    target.base.risk_exposures = forbidden
    x = target.sample_random(np.random.default_rng(720))
    mean_features = target.gpr_basis_map(output_index=1).features(x)
    risk = target.risk_exposures(x)
    cumulative = cumulative_feature_vector(risk)
    assert np.all(np.isfinite(mean_features))
    assert np.all(np.isfinite(cumulative))
    beta = target.cumulative_hvd_prior_beta(
        output_index=1,
        feature_dim=len(cumulative),
    )
    assert beta is not None
    assert beta.shape == cumulative.shape
    assert prior.observable_mean_model is not prior.observable_variance_model
    contract = target.mean_risk_coordinate_contract()
    assert contract["separate_mean_variance_heads"] is True
    assert contract["shared_observable_exposure_input"] is True
    audit = target.admissibility_audit()
    assert audit["admissible_strict_lodo"] is True
    assert audit["observable_state_mean_head_used"] is True
    assert audit["observable_state_variance_head_used"] is True
    diagnostics = prior.diagnostics()["observable_variance_coordinate"]
    assert diagnostics["target_oracle_used"] is False
    assert diagnostics["mean_head_parameters_shared"] is False
