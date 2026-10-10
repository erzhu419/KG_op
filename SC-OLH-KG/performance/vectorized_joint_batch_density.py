"""The existing raw batch marginal density, vectorized over Gaussian components."""

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import minimize_scalar
from scipy.special import digamma

from representation.physical_risk_prior import log_sample_variance_density


def component_batch_log_evidence(means, covariances, label, eta_label, replicates):
    means, covariances = np.asarray(means), np.asarray(covariances)
    gm, em = means.T; gv, ev = covariances[:, 0, 0], covariances[:, 1, 1]
    sd = np.sqrt(ev); cross = covariances[:, 0, 1]
    conditional_variance = np.maximum(gv-cross**2/ev, 0.); loading = cross/sd
    k = (replicates-1)/2
    sample_variance = float(np.exp(eta_label+digamma(k)-np.log(k)))

    def at(t, index=slice(None)):
        eta = em[index]+sd[index]*t
        total = conditional_variance[index]+np.exp(eta)/replicates
        error = label-gm[index]-loading[index]*t
        return (-.5*(np.log(2*np.pi)+t**2+np.log(2*np.pi*total)+error**2/total)
            +log_sample_variance_density(eta, sample_variance, replicates))

    modes = np.array([minimize_scalar(lambda t: -at(t, i), bounds=(-12., 12.), method='bounded').x
        for i in range(len(means))])
    shifts = at(modes)
    integral, error, info = quad_vec(lambda t: np.exp(at(t)-shifts), -12., 12.,
        points=np.unique([modes.min(), np.median(modes), modes.max()]), norm='max',
        epsabs=1e-10, epsrel=1e-10, workers=1, full_output=True)
    if not info.success:
        raise RuntimeError(f'Vectorized batch density integration failed: {info.message}')
    relative_error_bound = float(error/integral.min())
    assert np.all(integral > 0) and relative_error_bound <= 1e-7
    return shifts+np.log(integral), relative_error_bound
