"""Kalman (Bayesian linear regression) coefficient update.

Translated from: update_coeff.m

The parametric model for each objective i:
  mu_i(x) = f_i(x)^T @ b_i

where f_i(x) = [phi(x), e_k] is the augmented feature vector:
  - phi(x): basis functions [1, x_1, ..., x_d, x_1^2, ..., x_d^2]
  - e_k: indicator vector for the k-th unique sampled solution (deviation term)

The augmented parameter vector:
  b_i = [beta_i; zeta_i(x_1); ...; zeta_i(x_K)]

Update equations (Kalman rank-one):
  epsilon = y_i - f_t @ b_i
  gamma = lambda_i + f_t @ B_i @ f_t^T
  b_i_new = b_i + (epsilon / gamma) * B_i @ f_t^T
  B_i_new = B_i - (1/gamma) * (B_i @ f_t^T) @ (f_t @ B_i)
"""
import numpy as np


def build_augmented_feature(phi_x, solution_idx, total_sampled, is_new=False):
    """Build the augmented feature vector f_t.

    Parameters
    ----------
    phi_x : np.ndarray of shape (p,)
        Basis function values.
    solution_idx : int
        Index of this solution in sampled list.
    total_sampled : int
        Current total number of unique sampled solutions.
    is_new : bool
        Whether this solution is being sampled for the first time.

    Returns
    -------
    np.ndarray
        Augmented feature vector [phi_x, e_k] where e_k has 1 at solution_idx.
    """
    if is_new:
        # New solution: indicator has length = total_sampled (including this new one)
        indicator = np.zeros(total_sampled)
        indicator[solution_idx] = 1.0
    else:
        indicator = np.zeros(total_sampled)
        indicator[solution_idx] = 1.0
    return np.concatenate([phi_x, indicator])


def kalman_update(b, B, z0, phi_x, lem_x, solution_idx, y,
                  total_sampled_before, is_new):
    """Update posterior parameters for all objectives.

    Translated from update_coeff.m

    Parameters
    ----------
    b : list of np.ndarray
        b[i] is the augmented parameter vector for objective i.
    B : list of np.ndarray
        B[i] is the augmented covariance matrix for objective i.
    z0 : np.ndarray of shape (n_obj,)
        Prior variance for deviation terms.
    phi_x : np.ndarray of shape (p,)
        Basis function values for sampled solution.
    lem_x : np.ndarray of shape (n_obj,)
        Noise variance estimate at sampled solution.
    solution_idx : int
        Index of sampled solution.
    y : np.ndarray of shape (n_obj,)
        Observed simulation output.
    total_sampled_before : int
        Number of unique solutions BEFORE this sample.
    is_new : bool
        Whether this is a new (previously unsampled) solution.

    Returns
    -------
    b_new : list of np.ndarray
    B_new : list of np.ndarray
    """
    n_obj = len(b)
    b_new = []
    B_new = []

    for i in range(n_obj):
        b_i = b[i].copy()
        B_i = B[i].copy()
        p = len(phi_x)

        if is_new:
            # Expand b and B for new deviation term
            # Translated from update_coeff.m lines 13-19:
            #   b_t = [b_t; 0]
            #   f_t = [f_t, 1]
            #   B_t = [B_t, zeros; zeros, z0{i}]
            b_i = np.append(b_i, 0.0)
            n_old = len(B_i)
            B_new_mat = np.zeros((n_old + 1, n_old + 1))
            B_new_mat[:n_old, :n_old] = B_i
            B_new_mat[n_old, n_old] = z0[i]
            B_i = B_new_mat

        # Build augmented feature vector
        total_sampled = total_sampled_before + (1 if is_new else 0)
        f_t = build_augmented_feature(phi_x, solution_idx, total_sampled,
                                       is_new=False)

        # Ensure f_t matches b_i length
        if len(f_t) < len(b_i):
            f_t = np.concatenate([f_t, np.zeros(len(b_i) - len(f_t))])
        elif len(f_t) > len(b_i):
            f_t = f_t[:len(b_i)]

        # Kalman update
        epsilon = y[i] - f_t @ b_i
        gamma = lem_x[i] + f_t @ B_i @ f_t
        if abs(gamma) < 1e-15:
            gamma = 1e-15

        K_gain = B_i @ f_t / gamma
        b_i = b_i + epsilon * K_gain
        B_i = B_i - np.outer(K_gain, f_t @ B_i)

        # Symmetrize to avoid numerical drift
        B_i = (B_i + B_i.T) / 2.0

        b_new.append(b_i)
        B_new.append(B_i)

    return b_new, B_new


def posterior_mean(b_i, phi_x, solution_idx, total_sampled):
    """Compute posterior mean at a solution for one objective.

    Parameters
    ----------
    b_i : np.ndarray
        Augmented parameter vector for objective i.
    phi_x : np.ndarray of shape (p,)
        Basis function values.
    solution_idx : int or None
        Index in sampled list, or None if not sampled.
    total_sampled : int
        Current total number of sampled solutions.

    Returns
    -------
    float
        Posterior mean prediction.
    """
    f_t = np.zeros(len(b_i))
    p = len(phi_x)
    f_t[:p] = phi_x
    if solution_idx is not None and p + solution_idx < len(b_i):
        f_t[p + solution_idx] = 1.0
    return f_t @ b_i
