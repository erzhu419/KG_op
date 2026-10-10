import numpy as np
from scipy.special import ndtri

from representation.latent_risk_metrics import latent_risk_metrics


def test_mixture_variance_keeps_between_component_uncertainty():
    result = latent_risk_metrics(np.array([[-1., 3.]]), np.array([[.5, 1.5]]), [.75, .25], [0.])
    assert result["eta_rmse"] == 0.
    assert result["within_variance"] == .75
    assert result["between_variance"] == 3.
    assert result["posterior_variance"] == 3.75


def test_environmental_moment_integrates_components_before_exponentiation():
    result = latent_risk_metrics(np.array([[-1., 3.]]), np.array([[.5, 1.5]]), [.75, .25], [0.])
    expected = .75 * np.exp(-.75) + .25 * np.exp(3.75)
    np.testing.assert_allclose(result["environmental_variance_prediction"], expected)
    np.testing.assert_allclose(result["environmental_variance_bias"], expected - 1.)
    assert not np.isclose(expected, np.exp(.5 * result["posterior_variance"]))


def test_single_component_score_and_coverage_use_latent_variance():
    truth = ndtri([.01, .5, .99]) * 2.
    result = latent_risk_metrics(np.zeros((3, 1)), np.full((3, 1), 4.), [1.], truth)
    np.testing.assert_allclose(result["latent_log_score"], np.mean(-.5 * (truth ** 2 / 4. + np.log(8 * np.pi))))
    assert result["coverage90"] == 1. / 3.
    assert result["lower_miss"] == result["upper_miss"] == 1. / 3.
