import numpy as np
from scipy.special import logsumexp

from representation.archive_covariance_prior import sharing_branches, sharing_forecasts


def example():
    rng = np.random.default_rng(130)
    risk = {"log_evidence": rng.normal(size=(12, 4)),
            "log_density": rng.normal(size=(12, 4))}
    return risk, rng.normal(size=(12, 2)), np.array([.4, .6])


def test_sharing_matches_explicit_independent_archive_enumeration():
    risk, auxiliary, weights = example()
    branches = sharing_branches(risk, auxiliary, weights)
    scores, _, posterior = sharing_forecasts(branches)
    components, futures, component_ids = [], [], []
    for i in range(2):
        for k in range(2):
            j = 2*i+k
            components.append(risk["log_evidence"][:, j]+auxiliary[:, i]+np.log(.5*weights[i]/2))
            futures.append(risk["log_density"][:, j])
            component_ids.append(j)
    for f in range(2):
        for i in range(2):
            for k in range(2):
                j = 2*i+k
                components.append(risk["log_evidence"][:, j]+auxiliary[:, f]+np.log(.5*weights[f]*weights[i]/2))
                futures.append(risk["log_density"][:, j])
                component_ids.append(j)
    e, p = np.array(components).T, np.array(futures).T
    normalized = np.exp(e-logsumexp(e, axis=1, keepdims=True))
    expected = logsumexp(e+p, axis=1)-logsumexp(e, axis=1)
    expected_risk = np.column_stack([normalized[:, np.array(component_ids)==j].sum(axis=1) for j in range(4)])
    np.testing.assert_allclose(scores, expected, atol=1e-12)
    np.testing.assert_allclose(posterior, expected_risk, atol=1e-12)


def test_neutral_auxiliary_evidence_and_sharing_endpoints():
    risk, auxiliary, weights = example()
    branches = sharing_branches(risk, auxiliary, weights)
    for q, name in ((1., "J"), (0., "A")):
        scores, _, posterior = sharing_forecasts(branches, q)
        np.testing.assert_allclose(scores, branches["scores"][name], atol=1e-12)
        np.testing.assert_allclose(posterior, branches["risk_posteriors"][name], atol=1e-12)
    neutral = sharing_branches(risk, np.zeros_like(auxiliary), weights)
    scores, posterior, _ = sharing_forecasts(neutral)
    np.testing.assert_allclose(scores, neutral["scores"]["A"], atol=1e-12)
    np.testing.assert_allclose(posterior, .5, atol=1e-12)


def test_current_scoring_label_cannot_change_sharing_posterior():
    risk, auxiliary, weights = example()
    before = sharing_forecasts(sharing_branches(risk, auxiliary, weights))
    risk["log_density"] += np.arange(4)*100.
    after = sharing_forecasts(sharing_branches(risk, auxiliary, weights))
    np.testing.assert_allclose(before[1], after[1], atol=1e-12)
    np.testing.assert_allclose(before[2], after[2], atol=1e-12)
