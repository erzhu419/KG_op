import sys
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_hermitenorm
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from performance.adapt_historical_source_covariance import coefficient_fits, gaussian_rows_log_density
from performance.transport_historical_source_samples import reference_transport
from representation.exchangeable_mean import ExchangeableBoundaryMeanCoordinate


def test_coefficient_fit_covariance_matches_exact_noise_pushforward():
    X = np.column_stack([np.linspace(-2.,2.,8),np.linspace(-2.,2.,8)**2])
    noise = np.linspace(.2,.5,8); y = .7+X@np.array([.2,-.1])
    task = {'response':np.column_stack([y]*4), 'noise':np.column_stack([noise]*4)}
    _,covariances = coefficient_fits(np.tile(X,(4,1)),[task]*4,0,{'center':0.,'scale':1.},.001)
    # Exact Gaussian second-moment quadrature for the declared eight noises.
    coefficient_draws = [ExchangeableBoundaryMeanCoordinate._solve(X,y+sign*np.sqrt(8*noise[j])*np.eye(8)[j],
        1/noise,.001)[0] for j in range(8) for sign in (-1.,1.)]
    draws=np.asarray(coefficient_draws); centered=draws-draws.mean(axis=0)
    np.testing.assert_allclose(covariances[0],centered.T@centered/16,atol=1e-12,rtol=1e-10)


def test_posterior_transport_recovers_exact_evidence_ratio_with_jacobian():
    means=[np.array([.2,-.3]),np.array([.4,.1])]
    covariances=[np.array([[.9,.15],[.15,.6]]),np.array([[.6,-.12],[-.12,.8]])]
    noise=np.diag([.3,.4]); y=np.array([.7,-.5]); references=[]
    for mean,C in zip(means,covariances):
        P=np.linalg.inv(np.linalg.inv(C)+np.linalg.inv(noise))
        references.append((P@(np.linalg.solve(C,mean)+np.linalg.solve(noise,y)),P))
    z,w=roots_hermitenorm(3); grid=np.stack(np.meshgrid(z,z,indexing='ij'),axis=-1).reshape(-1,2)
    draws=references[0][0]+grid@np.linalg.cholesky(references[0][1]).T
    moved,log_jacobian=reference_transport(draws,*references)
    old=gaussian_rows_log_density(draws,means[0],covariances[0])+multivariate_normal.logpdf(draws,mean=y,cov=noise)
    new=gaussian_rows_log_density(moved,means[1],covariances[1])+multivariate_normal.logpdf(moved,mean=y,cov=noise)
    expected=multivariate_normal.logpdf(y,mean=means[1],cov=covariances[1]+noise)-multivariate_normal.logpdf(y,mean=means[0],cov=covariances[0]+noise)
    logs=(np.log(w)[:,None]+np.log(w)[None,:]-np.log(2*np.pi)).reshape(-1)
    actual=logsumexp(logs+new-old+log_jacobian)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0.)
    assert abs(logsumexp(logs+new-old)-expected)>.001
