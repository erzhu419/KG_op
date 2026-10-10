"""Restore the frozen fixed task covariance without rerunning optimization."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from representation.orthogonal_function_prior import spectral_weights
from representation.source_function_prior import SourceFunctionPrior


def restore_common_mean_prior(basis, diagnostics, batches, noises):
    """Recover the original GLS intercept variance from source records only.

    The common-mean artifact records all fitted spectral parameters but not
    the intercept variance. The original prior defines it from source GLS
    intercepts, with a .25 floor. No hyperparameter or mean is fitted here.
    """
    weights, _ = spectral_weights(basis, diagnostics['spectral_exponent'])
    groups = np.asarray(diagnostics['source_group_variances'])
    tail = diagnostics['tail_variance_budget']/basis.feature_dim
    tau = groups[basis.group_index]*weights+tail
    delta = diagnostics['discrepancy_variance']
    intercepts = []
    for batch, noise in zip(batches, noises):
        design = basis.transform(batch.psi)
        covariance = np.diag(np.asarray(noise)+delta)+(design*tau) @ design.T
        precision_one = cho_solve(cho_factor(covariance, lower=True), np.ones(len(design)))
        intercepts.append(float(precision_one @ batch.signals/precision_one.sum()))
    intercept_variance = float(np.var(intercepts, ddof=1)) if len(intercepts) > 1 else 0.
    covariance = np.diag(np.r_[max(intercept_variance, .25), tau])
    return SourceFunctionPrior(basis, np.asarray(diagnostics['coefficient_mean']),
                               covariance, delta, dict(diagnostics))
