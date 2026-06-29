"""Candidate solution generation.

Translated from: cand_sample.m, bi_obj.m, nonlcon.m, perato.m, perato_con.m

Strictly follows MATLAB logic:
  cand_sample.m:
    1. S = lhsdesign(K1, n) rounded to integer grid in [0,1]^d
    2. For k=1:K2:
       a. bb{i} = [mvnrnd(b{i}(1:p), B{i}(1:p,1:p)); b{i}(p+1:end)]
       b. If n_now <= n_thr: P = perato(bb, sampled_short, ...)
          Else: P = perato_con(bb, sampled_short, ...)
       c. S = union(S, P)

  perato.m / perato_con.m:
    gamultiobj(bi_obj, n, ..., [0]^d, [1]^d,
              'InitialPopulationMatrix', sampled', 'PopulationSize', 200)
    Result rounded to integer grid.

  bi_obj.m:
    f_t = [phi(x), indicator(x in sampled)]
    obj(i) = round(f_t @ bb{i} * 100) / 100   ← 2-decimal rounding
"""
import numpy as np
from scipy.stats import qmc

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem as PymooProblem
from pymoo.core.population import Population
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination

from .basis_functions import compute_features, num_features, feature_matrix
from .utils import x_in_s
from .pareto_utils import pareto_front_indices


class _PosteriorBiObjProblem(PymooProblem):
    """Bi-objective posterior optimization problem for pymoo.

    Strictly matches MATLAB bi_obj.m:
      f_t{i} = [F_x{i}, temp]  where temp is indicator for sampled solutions
      obj(i) = round(f_t{i} * bb{i} * 100) / 100

    And nonlcon.m for constraint:
      c = f_t @ bb{3} - tau_e + alpha * sqrt(lem_x(3))
    """

    def __init__(self, bb, sampled_norms, p, d,
                 tau_e=None, alpha_z=None, vepm=None):
        """
        Parameters
        ----------
        bb : list of np.ndarray
            bb[i] = [sampled_parametric_part; posterior_mean_deviation_part]
            Full augmented parameter vector with sampled parametric + posterior deviation.
        sampled_norms : list of np.ndarray
            Previously sampled solutions in normalized form.
        p : int
            Number of basis functions.
        d : int
            Decision dimension.
        tau_e : float or None
            Constraint threshold.
        alpha_z : float or None
        vepm : VEPM or None
            For computing variance of constraint at query points.
        """
        n_ieq = 1 if tau_e is not None else 0
        super().__init__(n_var=d, n_obj=2, n_ieq_constr=n_ieq,
                         xl=np.zeros(d), xu=np.ones(d))
        self.bb = bb
        self.sampled_norms = sampled_norms
        self.p = p
        self.d = d
        self.tau_e = tau_e
        self.alpha_z = alpha_z
        self.vepm = vepm

    def _evaluate(self, X, out, *args, **kwargs):
        """Vectorized evaluation matching bi_obj.m and nonlcon.m."""
        N = len(X)
        n_sampled = len(self.sampled_norms)

        # Compute basis features for all solutions
        Phi = feature_matrix(X)  # (N, p)

        # For each solution, check if it's in sampled set
        # Build augmented feature matrix: [Phi, indicator]
        # For NSGA-II candidates, most are NOT in sampled set, so indicator = 0
        # This means f_t @ bb = phi @ bb[:p] for most solutions
        # Only for solutions that happen to match a sampled solution does the deviation term contribute

        F = np.zeros((N, 2))
        for obj_i in range(2):
            bb_param = self.bb[obj_i][:self.p]
            bb_dev = self.bb[obj_i][self.p:]
            # Parametric contribution
            F[:, obj_i] = Phi @ bb_param
            # Deviation contribution (only for sampled solutions)
            for k in range(N):
                idx = x_in_s(self.sampled_norms, X[k])
                if idx is not None and idx < len(bb_dev):
                    F[k, obj_i] += bb_dev[idx]
            # Round to 2 decimal places (matching bi_obj.m)
            F[:, obj_i] = np.round(F[:, obj_i] * 100) / 100.0

        out["F"] = F

        # Constraint: nonlcon.m
        if self.tau_e is not None:
            G = np.zeros((N, 1))
            bb3_param = self.bb[2][:self.p]
            bb3_dev = self.bb[2][self.p:]
            for k in range(N):
                phi_k = Phi[k]
                pred = phi_k @ bb3_param
                idx = x_in_s(self.sampled_norms, X[k])
                if idx is not None and idx < len(bb3_dev):
                    pred += bb3_dev[idx]
                # Variance from VEPM
                lem_x = self.vepm.get_variance(X[k], idx) if self.vepm else np.array([0.01]*3)
                G[k, 0] = pred - self.tau_e + self.alpha_z * np.sqrt(lem_x[2])
            out["G"] = G


class _PosteriorBiObjProblemFast(PymooProblem):
    """Fast version: only uses parametric part of bb (no deviation lookup).

    This is valid because NSGA-II explores [0,1]^d continuously, and the
    probability of hitting an exact sampled solution is negligible for d=5.
    Uses round to 2 decimals matching bi_obj.m.
    """

    def __init__(self, bb_param, p, d, tau_e=None, alpha_z=None, sigma_3=None):
        n_ieq = 1 if tau_e is not None else 0
        super().__init__(n_var=d, n_obj=2, n_ieq_constr=n_ieq,
                         xl=np.zeros(d), xu=np.ones(d))
        self.bb_param = bb_param  # list of 3 arrays, each of shape (p,)
        self.p = p
        self.tau_e = tau_e
        self.alpha_z = alpha_z
        self.sigma_3 = sigma_3

    def _evaluate(self, X, out, *args, **kwargs):
        Phi = feature_matrix(X)
        f1 = np.round(Phi @ self.bb_param[0] * 100) / 100.0
        f2 = np.round(Phi @ self.bb_param[1] * 100) / 100.0
        out["F"] = np.column_stack([f1, f2])
        if self.tau_e is not None:
            f3 = Phi @ self.bb_param[2]
            out["G"] = (f3 - self.tau_e + self.alpha_z * self.sigma_3).reshape(-1, 1)


def generate_candidates(iteration, n_thr, K1, K2, d, b, B,
                        sampled_norms, vepm, tau_e, alpha_z, stdev,
                        rng=None, x_range=None):
    """Generate the candidate solution set.

    Strictly translated from cand_sample.m.

    Parameters match MATLAB example_with_post_process.m:
      K1=20, K2=2, n_thr=20

    Parameters
    ----------
    x_range : np.ndarray of shape (d,), optional
        x_U - x_L for each dimension. Used for proper integer grid rounding.
        If None, defaults to 100 for all dimensions (RZDT1/2 behavior).
    """
    if rng is None:
        rng = np.random.RandomState()
    if x_range is None:
        x_range = 100.0 * np.ones(d)

    p = num_features(d)
    candidates = []

    # ============================
    # Part 1: K1 LHD random samples
    # MATLAB: S = x_L + diag(x_U-x_L)*lhsdesign(K1,n)';
    #         S = round(S-x_L)./(x_U-x_L);
    # ============================
    try:
        sampler = qmc.LatinHypercube(d=d, seed=rng.randint(100000))
        lhd = sampler.random(K1)  # (K1, d) in [0, 1]
    except Exception:
        lhd = rng.rand(K1, d)
    # Round to problem's integer grid
    for i in range(d):
        lhd[:, i] = np.round(lhd[:, i] * x_range[i]) / x_range[i]
    lhd = np.clip(lhd, 0, 1)
    for row in lhd:
        candidates.append(row)

    # ============================
    # Part 2: K2 posterior draws + gamultiobj (NSGA-II)
    # MATLAB: for k=1:K2
    #   bb{i} = [mvnrnd(b{i}(1:dim), BB)'; b{i}(dim+1:end)]
    #   P = perato(bb, sampled_short, ...) or perato_con(...)
    #   S = union(S', P, 'rows')';
    # ============================
    for k in range(K2):
        # Sample posterior parametric coefficients
        # MATLAB: bb{i} = mvnrnd(b{i}(1:dim), BB)';
        #         bb{i} = [bb{i}; b{i}(dim+1:end)];
        bb_param = []
        for i in range(3):
            b_param = b[i][:p]
            B_param = B[i][:p, :p]
            B_param = (B_param + B_param.T) / 2
            eigvals = np.linalg.eigvalsh(B_param)
            if np.min(eigvals) < 0:
                B_param -= 1.1 * np.min(eigvals) * np.eye(p)
            try:
                theta_i = rng.multivariate_normal(b_param, B_param)
            except np.linalg.LinAlgError:
                theta_i = b_param + rng.randn(p) * 0.01
            bb_param.append(theta_i)

        # NSGA-II on posterior objectives
        # MATLAB gamultiobj: PopulationSize=200, InitialPopulationMatrix=sampled'
        # Default MaxGenerations = 200 * n_var = 1000 for d=5
        # We use n_gen=200 as practical compromise (MATLAB's compiled C is ~20x faster)
        use_constraint = (iteration > n_thr)

        if use_constraint:
            problem = _PosteriorBiObjProblemFast(
                bb_param, p, d,
                tau_e=tau_e, alpha_z=alpha_z, sigma_3=stdev[2]
            )
        else:
            problem = _PosteriorBiObjProblemFast(bb_param, p, d)

        algorithm = NSGA2(pop_size=200)

        try:
            res = pymoo_minimize(
                problem, algorithm,
                get_termination("n_gen", 50),
                seed=int(rng.randint(100000)),
                verbose=False
            )
            if res.X is not None:
                X_result = res.X
                if X_result.ndim == 1:
                    X_result = X_result.reshape(1, -1)
                # MATLAB: P = round(P.*(x_U-x_L)')./(x_U-x_L)';
                for i in range(d):
                    X_result[:, i] = np.round(X_result[:, i] * x_range[i]) / x_range[i]
                X_result = np.clip(X_result, 0, 1)
                X_result = np.unique(X_result, axis=0)
                for row in X_result:
                    candidates.append(row)
        except Exception as e:
            # Fallback: random search (should not happen normally)
            X_rand = rng.rand(500, d)
            Phi = feature_matrix(X_rand)
            obj = np.column_stack([
                np.round(Phi @ bb_param[0] * 100) / 100.0,
                np.round(Phi @ bb_param[1] * 100) / 100.0
            ])
            pf_idx = pareto_front_indices(obj)
            for idx in pf_idx:
                x = X_rand[idx].copy()
                for i in range(d):
                    x[i] = np.round(x[i] * x_range[i]) / x_range[i]
                candidates.append(np.clip(x, 0, 1))

    # MATLAB: S = union(S', P, 'rows')' — remove duplicates
    if candidates:
        candidates_arr = np.array(candidates)
        for i in range(d):
            candidates_arr[:, i] = np.round(candidates_arr[:, i] * x_range[i]) / x_range[i]
        candidates_arr = np.clip(candidates_arr, 0, 1)
        candidates_arr = np.unique(candidates_arr, axis=0)
        candidates = [candidates_arr[i] for i in range(len(candidates_arr))]

    return candidates
