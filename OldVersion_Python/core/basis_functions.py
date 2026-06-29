"""Basis (feature) function computation.

Translated from: feat.m

In the old MATLAB code:
  key{i} = [eye(n), 2*eye(n)]  for all i=1,2,3
  F_x{i}(j) = prod(x.^key{i}(:,j))  for j=1..2n

  With key = [eye(n), 2*eye(n)]:
    - Columns 1..n of key: prod(x.^e_j) = x_j     (linear features)
    - Columns n+1..2n:     prod(x.^(2*e_j)) = x_j^2 (quadratic features)

  Result: phi(x) = [1, x_1, ..., x_d, x_1^2, ..., x_d^2]
  Length: p = 2*d + 1
"""
import numpy as np


def compute_features(x_norm):
    """Compute the basis function vector phi(x) for normalized solution x.

    Parameters
    ----------
    x_norm : np.ndarray of shape (d,)
        Normalized solution in [0, 1]^d.

    Returns
    -------
    np.ndarray of shape (2*d+1,)
        Feature vector [1, x_1, ..., x_d, x_1^2, ..., x_d^2].
    """
    x = np.asarray(x_norm, dtype=float)
    d = len(x)
    phi = np.empty(2 * d + 1)
    phi[0] = 1.0        # intercept
    phi[1:d+1] = x      # linear features
    phi[d+1:] = x**2    # quadratic features
    return phi


def num_features(d):
    """Return the number of basis functions for dimension d."""
    return 2 * d + 1


def feature_matrix(X_norm):
    """Compute feature matrix for multiple solutions.

    Parameters
    ----------
    X_norm : np.ndarray of shape (N, d)
        Each row is a normalized solution.

    Returns
    -------
    np.ndarray of shape (N, 2*d+1)
    """
    N, d = X_norm.shape
    Phi = np.empty((N, 2 * d + 1))
    Phi[:, 0] = 1.0
    Phi[:, 1:d+1] = X_norm
    Phi[:, d+1:] = X_norm**2
    return Phi
