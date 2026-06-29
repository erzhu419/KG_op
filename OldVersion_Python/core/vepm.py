"""Variance Estimation via Parametric Model (VEPM).

Translated from: part_id.m, update_var.m, var_x.m

Old version VEPM design:
  - Features for partitioning: same as basis features minus intercept
    i.e., [x1, ..., xd, x1^2, ..., xd^2] = 2d features
  - Bin edges per feature: F_part = [0, 0.5, 1] => 2 bins each
  - Total partitions per objective: 2^(2d) = 1024 for d=5
  - Each objective has INDEPENDENT partition (though same structure)

  - Lem{i}(j): partition-level variance for objective i, partition j
  - Lem_s(i, k): solution-level variance for objective i, sampled solution k

Key parameters:
  - s0 = 1: prior weight
  - var0 = 0.01: prior variance for all objectives
  - n_thr = 20: minimum sample threshold for VEPM-guided candidate generation
"""
import numpy as np
from .basis_functions import compute_features


class VEPM:
    """Variance Estimation via Parametric Model.

    Maintains per-partition variance estimates and per-solution variance estimates.
    """

    def __init__(self, d, n_obj=3, s0=1.0, var0=0.01, n_thr=20):
        self.d = d
        self.n_obj = n_obj
        self.s0 = s0
        self.var0_scalar = var0
        self.var0 = var0 * np.ones(n_obj)
        self.n_thr = n_thr

        # Partition structure: 2d features, 2 bins each
        self.n_features = 2 * d
        self.n_bins = 2
        self.total_partitions = self.n_bins ** self.n_features  # 2^(2d)

        # Bin edges for each feature (features are in [0, 1] since x is in [0, 1])
        self.bin_edges = np.array([0.0, 0.5, 1.0])

        # Partition-level variance: Lem{i}(j) for each objective i, partition j
        # Initialize all to var0
        self.Lem = np.full((self.n_obj, self.total_partitions), var0)

        # Solution-level variance: Lem_s[i, k] for sampled solution k
        # Grows as new solutions are sampled
        self.Lem_s = np.empty((self.n_obj, 0))

    def _compute_partition_features(self, x_norm):
        """Compute the 2d partition features from normalized x.

        Features: [x_1, ..., x_d, x_1^2, ..., x_d^2]
        Same as basis functions minus intercept.
        """
        x = np.asarray(x_norm, dtype=float)
        features = np.concatenate([x, x ** 2])
        return features

    def get_partition_id(self, x_norm, obj_idx=0):
        """Compute partition ID for a normalized solution.

        Translated from part_id.m:
          For each feature j, find which bin it falls in.
          Combine: idp = sum((part_j - 1) * prod(N_{j+1:end} - 1)) + part_last

        In our simplified version with 2 bins per feature:
          bin_j = 0 if feature_j <= 0.5, else 1
          pid = sum(bin_j * 2^j)

        Note: In MATLAB, each objective can have different partitions.
        Here they share the same structure, so obj_idx is unused but kept
        for interface compatibility.
        """
        features = self._compute_partition_features(x_norm)
        features = np.clip(features, 0.0, 0.9999)
        # Determine bin for each feature: 0 if <= 0.5, 1 if > 0.5
        bin_ids = (features > 0.5).astype(int)
        # Combine into partition ID using binary encoding
        pid = 0
        for j in range(len(bin_ids)):
            pid += bin_ids[j] * (2 ** j)
        return int(pid)

    def get_variance(self, x_norm, solution_idx=None):
        """Get variance estimate for a solution.

        Translated from var_x.m:
          If x has been sampled (solution_idx is not None):
            return Lem_s[:, solution_idx]
          Else:
            find partition ID and return Lem[:, partition_id]

        Parameters
        ----------
        x_norm : np.ndarray of shape (d,)
            Normalized solution.
        solution_idx : int or None
            Index in sampled solutions list, or None if not sampled.

        Returns
        -------
        np.ndarray of shape (n_obj,)
            Variance estimate for each objective.
        """
        if solution_idx is not None and self.Lem_s.shape[1] > solution_idx:
            return self.Lem_s[:, solution_idx].copy()
        else:
            pid = self.get_partition_id(x_norm)
            return self.Lem[:, pid].copy()

    def update(self, x_norm, y_obs, theta_pred, solution_idx, num_samples,
               all_sampled_norms, all_num_samples):
        """Update variance estimates after observing y at solution x.

        Translated from update_var.m:
          1. Update solution-level variance Lem_s for the sampled solution
          2. Find all sampled solutions in the same partition
          3. Update partition-level variance Lem as weighted average

        Parameters
        ----------
        x_norm : np.ndarray of shape (d,)
            Normalized solution that was sampled.
        y_obs : np.ndarray of shape (n_obj,)
            Observed simulation output.
        theta_pred : np.ndarray of shape (n_obj,)
            Posterior mean prediction at x (f_t @ b for each objective).
        solution_idx : int
            Index of this solution in sampled list.
        num_samples : int
            Number of times this solution has been sampled (AFTER this sample).
        all_sampled_norms : list of np.ndarray
            All sampled solutions in normalized form.
        all_num_samples : np.ndarray
            Number of samples for each sampled solution.
        """
        # Ensure Lem_s has enough columns
        while self.Lem_s.shape[1] <= solution_idx:
            self.Lem_s = np.column_stack([self.Lem_s, self.var0])

        K = len(all_sampled_norms)

        for i in range(self.n_obj):
            residual_sq = (y_obs[i] - theta_pred[i]) ** 2
            old_val = self.Lem_s[i, solution_idx]
            weight = self.s0 + num_samples - 1  # num before this update
            self.Lem_s[i, solution_idx] = (
                (old_val * weight + residual_sq) / (weight + 1)
            )

            # Compute partition ID for current solution
            pid = self.get_partition_id(x_norm)

            # Find all sampled solutions in the same partition
            # Weighted average of their solution-level variances
            # Only consider solutions that already have Lem_s entries
            r = 0
            sum_var = 0.0
            n_initialized = self.Lem_s.shape[1]
            for k in range(min(K, n_initialized)):
                if k == solution_idx:
                    continue
                pid_k = self.get_partition_id(all_sampled_norms[k])
                if pid_k == pid:
                    sum_var += self.Lem_s[i, k] * all_num_samples[k]
                    r += all_num_samples[k]

            # Add current solution
            sum_var += self.Lem_s[i, solution_idx] * num_samples
            r += num_samples

            if r > 0:
                self.Lem[i, pid] = sum_var / r
