"""Pre-sampling and initialization.

Translated from: pre_sample.m

Initialization procedure:
  1. Generate N0 LHD samples in [0,1]^d, round to integer grid
  2. Simulate at each sample point (original scale)
  3. Build feature matrix X_i for each objective
  4. Linear regression: b_i = (X_i^T X_i)^{-1} X_i^T y_i
  5. Initialize:
     - b{i}: regression coefficients [beta_i]
     - B{i}: var(b_hat_i) * I_p (diagonal covariance)
     - z0{i}: var(residuals_i) (prior deviation variance)
"""
import numpy as np
from scipy.stats import qmc

from .basis_functions import compute_features, num_features, feature_matrix


def presample(N0, d, problem, rng=None):
    """Generate pre-samples and initialize model parameters.

    Translated from pre_sample.m

    Parameters
    ----------
    N0 : int
        Number of pre-samples.
    d : int
        Decision dimension.
    problem : OldTestProblem
        Test problem instance.
    rng : np.random.RandomState

    Returns
    -------
    b : list of np.ndarray
        b[i] is the initial parameter vector for objective i (length p).
    B : list of np.ndarray
        B[i] is the initial covariance matrix (p x p).
    z0 : np.ndarray of shape (3,)
        Prior deviation variance for each objective.
    sampled_norms : list of np.ndarray
        Pre-sampled solutions in normalized form.
    sampled_orig : list of np.ndarray
        Pre-sampled solutions in original scale.
    Y_pre : np.ndarray of shape (N0, 3)
        Simulation outputs at pre-samples.
    """
    if rng is None:
        rng = np.random.RandomState()

    p = num_features(d)
    n_obj = 3

    # Generate N0 LHD samples in [0,1]^d
    try:
        sampler = qmc.LatinHypercube(d=d, seed=rng.randint(100000))
        S_norm = sampler.random(N0)
    except Exception:
        S_norm = rng.rand(N0, d)

    # Snap to problem's integer grid: denormalize→round→normalize
    # MATLAB: S = round(S .* diag(x_U-x_L)) ./ diag(x_U-x_L)
    x_range = problem.x_U - problem.x_L
    for i in range(d):
        S_norm[:, i] = np.round(S_norm[:, i] * x_range[i]) / x_range[i]
    S_norm = np.clip(S_norm, 0, 1)

    # Simulate at each pre-sample
    Y_pre = np.zeros((N0, n_obj))
    sampled_norms = []
    sampled_orig = []

    for k in range(N0):
        x_orig = problem.denormalize(S_norm[k])
        x_norm = problem.normalize(x_orig)  # canonical normalized form
        y = problem.simulate(x_orig, rng=rng)
        Y_pre[k] = y
        sampled_norms.append(x_norm.copy())
        sampled_orig.append(x_orig.copy())

    # Build feature matrix from canonical normalized solutions
    S_norm_canonical = np.array([s for s in sampled_norms])
    Phi = feature_matrix(S_norm_canonical)  # (N0, p)

    # Linear regression for each objective
    b = []
    B = []
    z0 = np.zeros(n_obj)

    for i in range(n_obj):
        y_i = Y_pre[:, i]
        # Ordinary least squares: b_i = (Phi^T Phi)^{-1} Phi^T y
        try:
            b_i = np.linalg.lstsq(Phi, y_i, rcond=None)[0]
        except np.linalg.LinAlgError:
            b_i = np.zeros(p)

        # Initial covariance: var(b_hat) * I
        # Translated from pre_sample.m: B{i} = var(b{i}) * eye(p)
        b_var = np.var(b_i) if len(b_i) > 1 else 0.01
        if b_var < 1e-8:
            b_var = 0.01
        B_i = b_var * np.eye(p)

        # Prior deviation variance: var(residuals)
        residuals = y_i - Phi @ b_i
        z0_i = np.var(residuals)
        if z0_i < 1e-8:
            z0_i = 0.01

        b.append(b_i)
        B.append(B_i)
        z0[i] = z0_i

    return b, B, z0, sampled_norms, sampled_orig, Y_pre
