import numpy as np

from representation.archive_covariance_prior import prequential_components
from representation.risk_class_information import candidate_risk_predictions, risk_class_information
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior


def test_candidate_predictions_match_existing_repeated_query_forecasts():
    rng = np.random.default_rng(3911)
    psi = rng.uniform(-2, 2, (16, 8))
    basis = FixedFunctionBasis(rng.uniform(-2, 2, (96, 8)))
    prior = SourceFunctionPrior(basis, np.r_[.3, np.zeros(68)], np.diag(np.r_[.25, np.full(68, .02)]), .02, {})
    ids = np.array([0, 1, 2, 0, 3, 2, 4, 0, 2])
    labels, noise = rng.normal(size=9), np.full(9, .2)
    old = prequential_components([prior], psi, ids, labels, noise)
    means, variance = candidate_risk_predictions([prior], psi, ids[:8], labels[:8], noise[:8], .2)
    np.testing.assert_allclose(means[ids[8], 0], old["mean"][8, 0], atol=1e-12)
    np.testing.assert_allclose(variance[ids[8], 0], old["variance"][8, 0] + .2, atol=1e-12)


def test_identical_predictive_distributions_have_no_class_information():
    means, variance = np.full((3, 4), 2.), np.full((3, 4), .5)
    np.testing.assert_allclose(risk_class_information(means, variance, [.1, .4, .2, .3]), 0., atol=1e-12)


def test_separated_states_reach_group_entropy_without_local_atom_information():
    means, variance, weights = np.array([[0., 0., 15., -15.]]), np.full((1, 4), .01), np.array([.1, .4, .2, .3])
    state_weights = np.r_[weights[:2].sum(), weights[2:]]
    entropy = -state_weights @ np.log(state_weights)
    np.testing.assert_allclose(risk_class_information(means, variance, weights), entropy, atol=1e-12)
