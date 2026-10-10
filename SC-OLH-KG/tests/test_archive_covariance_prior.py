import numpy as np

from representation.archive_covariance_prior import align_prior, prequential_components, joint_archive_forecasts
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior
from performance.target_risk_residual import predictive_group_score


def example():
    rng = np.random.default_rng(104)
    basis = FixedFunctionBasis(rng.uniform(-2, 2, (96, 8)))
    prior = SourceFunctionPrior(basis, np.r_[.3, np.zeros(68)],
        np.diag(np.r_[.25, rng.uniform(.001, .025, 68)]), .02, {})
    psi = rng.uniform(-2, 2, (12, 8))
    ids = np.array([0, 1, 2, 0, 3, 4, 2, 5, 6, 7])
    return rng, prior, psi, ids, rng.normal(size=len(ids)), np.full(len(ids), .1)


def test_affine_alignment_preserves_complete_kernel_and_mean():
    rng, prior, psi, *_ = example()
    basis = FixedFunctionBasis(rng.uniform(-1.5, 2.5, (96, 8)))
    aligned = align_prior(prior, basis)
    old = np.column_stack((np.ones(len(psi)), prior.basis.transform(psi)))
    new = np.column_stack((np.ones(len(psi)), aligned.basis.transform(psi)))
    np.testing.assert_allclose(new @ aligned.mean, old @ prior.mean, atol=1e-12)
    np.testing.assert_allclose(new @ aligned.covariance @ new.T, old @ prior.covariance @ old.T, atol=1e-12)
    assert np.max(np.abs(aligned.covariance[0, 1:])) > 0.


def test_prequential_repeated_candidates_match_grouped_gaussian():
    _, prior, psi, ids, labels, noise = example()
    output = prequential_components([prior], psi, ids, labels, noise)
    for t in range(1, len(ids)):
        reference = predictive_group_score([prior], psi, ids[:t], labels[:t], noise[:t],
            0., int(ids[t]), labels[t:t+1], noise[t:t+1])
        np.testing.assert_allclose(output["log_density"][t, 0], reference, atol=1e-12)


def test_future_labels_cannot_change_prefix_evidence_or_prediction():
    _, prior, psi, ids, labels, noise = example()
    before = prequential_components([prior], psi, ids, labels, noise)
    labels[5:] += 100.
    after = prequential_components([prior], psi, ids, labels, noise)
    for key in ("log_evidence", "mean", "variance"):
        np.testing.assert_allclose(before[key][:6], after[key][:6], atol=1e-12)


def test_joint_archive_evidence_preserves_within_archive_risk_odds():
    rng, prior, psi, ids, labels, noise = example()
    other = SourceFunctionPrior(prior.basis, prior.mean, prior.covariance*2., .02, {})
    risk = prequential_components([prior, other, prior, other], psi, ids, labels, noise)
    auxiliary = rng.normal(size=(len(ids), 2))
    _, posterior = joint_archive_forecasts(risk, auxiliary, [.5, .5])
    expected = risk['log_evidence'][:, 0]-risk['log_evidence'][:, 1]
    for first in (0, 2):
        np.testing.assert_allclose(np.log(posterior['J'][:, first]/posterior['J'][:, first+1]), expected, atol=1e-12)
