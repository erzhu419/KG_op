import numpy as np

from performance.conditional_sharing_calibration import fit_sharing_prior, posterior_q_moments


def test_identical_branch_scores_retain_uniform_sharing_uncertainty():
    rng = np.random.default_rng(131)
    evidence = rng.normal(size=(64,2))
    predictive = np.repeat(rng.normal(size=(64,1)),2,axis=1)
    fit = fit_sharing_prior(evidence,predictive,64)
    np.testing.assert_allclose(fit['q_moments'],[.5,1/3,1/4],atol=1e-12)
    np.testing.assert_allclose(fit['q_sd'],np.sqrt(1/12),atol=1e-12)


def test_continuous_sharing_moments_match_explicit_node_posterior():
    evidence = np.array([[0.,0.],[1.,-1.],[-1.,1.]])
    fit = fit_sharing_prior(evidence,np.array([[.3,0.],[0.,.4],[.2,-.1]]),64)
    q,w = np.array(fit['q_nodes']),np.array(fit['posterior_weights'])
    mean,sd = posterior_q_moments({'log_evidence':evidence},fit)
    for i,e in enumerate(evidence):
        posterior = w*(q*np.exp(e[0])+(1-q)*np.exp(e[1]))
        posterior /= posterior.sum()
        expected = posterior@q
        np.testing.assert_allclose(mean[i],expected,atol=1e-12)
        np.testing.assert_allclose(sd[i],np.sqrt(posterior@((q-expected)**2)),atol=1e-12)
