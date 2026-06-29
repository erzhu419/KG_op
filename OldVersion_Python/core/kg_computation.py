"""Knowledge Gradient factor computation and solution selection.

Translated from: KG_factor.m, KG_sol.m

KG_factor.m computes log(KG) for each of the two objectives:
  For objective i (i=1,2):
    1. Build F_t: feature matrix for all candidate solutions
    2. p = -F_t @ b_t  (negative posterior means)
    3. q = F_t @ B_t @ f_t' / sqrt(lem_x + f_t @ B_t @ f_t')
    4. Apply h-function (Frazier & Powell 2009)

KG_sol.m selects the sampling decision:
    1. Compute log_KG for each candidate on both objectives
    2. Find Pareto non-dominated in KG space
    3. Select via weighted average (0.5, 0.5)
"""
import numpy as np
from scipy.stats import norm
from .basis_functions import compute_features
from .utils import x_in_s, batch_x_in_s


def _h_function(p, q):
    """Compute the h-function value h(p, q) for KG.

    Translated from KG_factor.m lines 42-92.

    The h-function computes:
      h = log(sum_{j=0}^{J-1} (q_{j+1} - q_j) * [phi(-|a_j|) - |a_j| * Phi(-|a_j|)])

    where a_j = (p_j - p_{j+1}) / (q_{j+1} - q_j) after removing dominated entries.

    Parameters
    ----------
    p : np.ndarray of shape (K,)
        Negative posterior means.
    q : np.ndarray of shape (K,)
        KG direction coefficients.

    Returns
    -------
    float
        log(KG) value, or -inf if KG is essentially zero.
    """
    K = len(p)
    if K == 0:
        return -np.inf

    # Step 1: Sort by q ascending, break ties by p ascending
    pq = np.column_stack([p, q])
    # Sort by q first, then by p
    idx = np.lexsort((pq[:, 0], pq[:, 1]))
    pq = pq[idx]

    # Step 2: Remove entries where q_j == q_{j+1} (keep only the one with larger p)
    keep = []
    i = 0
    while i < len(pq):
        j = i
        while j < len(pq) - 1 and abs(pq[j+1, 1] - pq[j, 1]) < 1e-12:
            j += 1
        # Keep the one with largest p in this group
        keep.append(j)
        i = j + 1
    pq = pq[keep]

    if len(pq) <= 1:
        return -np.inf

    p_s = pq[:, 0]
    q_s = pq[:, 1]

    # Step 3: Compute a_j = (p_j - p_{j+1}) / (q_{j+1} - q_j)
    dim = len(p_s)
    a = np.zeros(dim)
    a[-1] = np.inf
    for j in range(dim - 1):
        dq = q_s[j+1] - q_s[j]
        if abs(dq) < 1e-15:
            a[j] = np.inf
        else:
            a[j] = (p_s[j] - p_s[j+1]) / dq

    # Step 4: Remove entries where a_j >= a_{j+1} (ensure a is non-decreasing)
    # Iterative pruning
    changed = True
    while changed:
        changed = False
        new_p, new_q, new_a = [], [], []
        i = 0
        while i < dim:
            if i < dim - 1 and a[i] >= a[i+1]:
                # Remove entry i
                changed = True
                i += 1
                continue
            new_p.append(p_s[i])
            new_q.append(q_s[i])
            new_a.append(a[i])
            i += 1

        p_s = np.array(new_p)
        q_s = np.array(new_q)
        a = np.array(new_a)
        dim = len(p_s)

        # Recompute a
        if dim > 1:
            a[-1] = np.inf
            for j in range(dim - 1):
                dq = q_s[j+1] - q_s[j]
                if abs(dq) < 1e-15:
                    a[j] = np.inf
                else:
                    a[j] = (p_s[j] - p_s[j+1]) / dq

    if dim <= 1:
        return -np.inf

    # Step 5: Compute KG value
    # sum_v = sum_{j=0}^{dim-2} (q_{j+1} - q_j) * [phi(-|a_j|) - |a_j| * Phi(-|a_j|)]
    sum_v = 0.0
    for j in range(dim - 1):
        abs_a = abs(a[j])
        term = (q_s[j+1] - q_s[j]) * (
            norm.pdf(-abs_a) - abs_a * norm.cdf(-abs_a)
        )
        sum_v += term

    if sum_v <= 0:
        return -np.inf
    return np.log(sum_v)


def compute_log_kg(candidate_solutions_norm, sampled_norms, x_norm,
                   b, B, z0, lem_x, phi_x, d):
    """Compute log(KG) for solution x on both objectives.

    Translated from KG_factor.m

    Parameters
    ----------
    candidate_solutions_norm : list of np.ndarray
        All candidate solutions in normalized form.
    sampled_norms : list of np.ndarray
        All sampled solutions in normalized form.
    x_norm : np.ndarray
        The solution for which to compute KG.
    b : list of np.ndarray
        b[i] is augmented parameter vector for objective i.
    B : list of np.ndarray
        B[i] is augmented covariance matrix for objective i.
    z0 : np.ndarray
        Prior deviation variance.
    lem_x : np.ndarray of shape (n_obj,)
        Noise variance at x.
    phi_x : np.ndarray
        Basis functions at x.
    d : int
        Decision dimension.

    Returns
    -------
    np.ndarray of shape (2,)
        log(KG) for objective 1 and 2.
    """
    K = len(candidate_solutions_norm)
    p_dim = len(phi_x)
    log_kg = np.full(2, -np.inf)

    # Find index of x in sampled solutions
    x_sampled_idx = x_in_s(sampled_norms, x_norm)
    # Find index of x in candidate set
    x_cand_idx = x_in_s(candidate_solutions_norm, x_norm)

    for obj_i in range(2):  # Only objectives 1 and 2
        b_i = b[obj_i]
        B_i = B[obj_i]
        n_total = len(b_i)

        # Build F_t: feature matrix for all candidates
        F_t = np.zeros((K, n_total))
        for k in range(K):
            phi_k = compute_features(candidate_solutions_norm[k])
            F_t[k, :p_dim] = phi_k
            # Find if candidate k has been sampled
            idx_k = x_in_s(sampled_norms, candidate_solutions_norm[k])
            if idx_k is not None and p_dim + idx_k < n_total:
                F_t[k, p_dim + idx_k] = 1.0

        # Build f_t for solution x
        f_t = np.zeros(n_total)
        f_t[:p_dim] = phi_x
        if x_sampled_idx is not None and p_dim + x_sampled_idx < n_total:
            f_t[p_dim + x_sampled_idx] = 1.0

        # Handle case where x is new (not in sampled set)
        if x_sampled_idx is None:
            # Expand dimensions to include new deviation term
            n_new = n_total + 1
            F_t_new = np.zeros((K, n_new))
            F_t_new[:, :n_total] = F_t
            if x_cand_idx is not None:
                F_t_new[x_cand_idx, n_total] = 1.0
            F_t = F_t_new

            f_t_new = np.zeros(n_new)
            f_t_new[:n_total] = f_t
            f_t_new[n_total] = 1.0
            f_t = f_t_new

            b_i_new = np.append(b_i, 0.0)
            B_i_new = np.zeros((n_new, n_new))
            B_i_new[:n_total, :n_total] = B_i
            B_i_new[n_total, n_total] = z0[obj_i]
            b_i = b_i_new
            B_i = B_i_new

        # Compute p and q vectors
        p_vec = -F_t @ b_i  # negative posterior means (for maximization)
        denom_sq = lem_x[obj_i] + f_t @ B_i @ f_t
        if denom_sq <= 0:
            continue
        q_vec = F_t @ B_i @ f_t / np.sqrt(denom_sq)

        log_kg[obj_i] = _h_function(p_vec, q_vec)

    return log_kg


def select_kg_solution(candidate_solutions_norm, sampled_norms,
                       b, B, z0, lem_all, d):
    """Select the best sampling decision based on KG values.

    Translated from KG_sol.m:
      1. Compute log_KG for each candidate
      2. Find Pareto non-dominated in KG space (2 objectives)
      3. Select via weighted average (0.5, 0.5)
    """
    K = len(candidate_solutions_norm)
    V_KG = np.full((K, 2), -np.inf)

    # Pre-compute candidate-to-sampled index mapping (once, not K times)
    cand_sampled_idx = batch_x_in_s(sampled_norms, candidate_solutions_norm)

    # Pre-compute feature matrix for all candidates
    p_dim = len(compute_features(candidate_solutions_norm[0]))
    Phi_cand = np.zeros((K, p_dim))
    for k in range(K):
        Phi_cand[k] = compute_features(candidate_solutions_norm[k])

    for k in range(K):
        x_norm = candidate_solutions_norm[k]
        phi_x = Phi_cand[k]
        lem_x = lem_all[:, k]

        log_kg = _compute_log_kg_fast(
            Phi_cand, cand_sampled_idx, sampled_norms, x_norm,
            cand_sampled_idx[k], b, B, z0, lem_x, phi_x, d
        )
        V_KG[k] = log_kg

    # Find Pareto non-dominated solutions in KG space (maximization)
    idx_sort = np.argsort(V_KG[:, 0])
    P_KG_indices = []

    for rank, k in enumerate(idx_sort):
        if rank == K - 1:
            P_KG_indices.append(k)
        else:
            remaining = idx_sort[rank+1:]
            max_obj2 = np.max(V_KG[remaining, 1])
            if V_KG[k, 1] >= max_obj2:
                P_KG_indices.append(k)

    if not P_KG_indices:
        P_KG_indices = list(range(K))

    # Select via weighted average (0.5, 0.5)
    P_KG_values = V_KG[P_KG_indices]
    weighted = P_KG_values @ np.array([0.5, 0.5])
    best_local = np.argmax(weighted)
    return P_KG_indices[best_local]


def _compute_log_kg_fast(Phi_cand, cand_sampled_idx, sampled_norms, x_norm,
                         x_sampled_idx, b, B, z0, lem_x, phi_x, d):
    """Optimized log(KG) computation.

    MATLAB-compatible: deviation terms are NEVER activated for candidates.
    The MATLAB code's x_in_s(sampled, S(:,k), n) always returns empty because
    sampled stores original-scale values but S(:,k) is normalized.
    Similarly, x is always treated as new (x_sampled_idx forced to None).
    """
    K = len(Phi_cand)
    p_dim = Phi_cand.shape[1]
    log_kg = np.full(2, -np.inf)

    x_cand_idx = None
    for k in range(K):
        if np.array_equal(Phi_cand[k], phi_x):
            x_cand_idx = k
            break

    for obj_i in range(2):
        b_i = b[obj_i]
        B_i = B[obj_i]
        n_total = len(b_i)

        # Build F_t: only parametric features, NO deviation terms
        # (MATLAB bug: x_in_s never matches → temp is always zeros)
        F_t = np.zeros((K, n_total))
        F_t[:, :p_dim] = Phi_cand
        # Deviation terms intentionally left as zeros for all candidates

        # Build f_t for solution x — always treated as new
        f_t = np.zeros(n_total)
        f_t[:p_dim] = phi_x
        # x_sampled_idx is effectively always None in MATLAB

        # Always expand for "new" solution (MATLAB behavior)
        n_new = n_total + 1
        F_t_new = np.zeros((K, n_new))
        F_t_new[:, :n_total] = F_t
        if x_cand_idx is not None:
            F_t_new[x_cand_idx, n_total] = 1.0
        F_t = F_t_new

        f_t_new = np.zeros(n_new)
        f_t_new[:n_total] = f_t
        f_t_new[n_total] = 1.0
        f_t = f_t_new

        b_i = np.append(b_i, 0.0)
        B_i_new = np.zeros((n_new, n_new))
        B_i_new[:n_total, :n_total] = B_i
        B_i_new[n_total, n_total] = z0[obj_i]
        B_i = B_i_new

        # Compute p and q vectors
        p_vec = -F_t @ b_i
        denom_sq = lem_x[obj_i] + f_t @ B_i @ f_t
        if denom_sq <= 0:
            continue
        q_vec = F_t @ B_i @ f_t / np.sqrt(denom_sq)

        log_kg[obj_i] = _h_function(p_vec, q_vec)

    return log_kg
