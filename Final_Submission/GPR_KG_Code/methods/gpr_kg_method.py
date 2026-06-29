"""
GPR-KG method wrapper.

Wraps the existing GPRKR_Algorithm from gpr_kg.py into the BaseMethod interface.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpr_kg import GPRKR_Algorithm, pareto_filter, compute_hypervolume_2d
from methods.base_method import BaseMethod
from experiments.config import HV_EVAL_INTERVAL, GPR_KG_PARAMS


def _json_safe(obj):
    """Convert numpy/tuple objects in detailed logs to JSON-safe values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return obj


class GPRKGMethod(BaseMethod):
    """GPR-KG: the proposed method (full framework)."""

    name = "GPR-KG"

    def __init__(self, **override_params):
        """Allow overriding default GPR-KG parameters for ablation studies."""
        self.params = dict(GPR_KG_PARAMS)
        self.params.update(override_params)
        # Optional instrumentation — passed through to GPRKR_Algorithm
        self.instrument = None
        self.instrument_log = []

    def run(self, problem, N, n0, seed, hv_eval_interval=HV_EVAL_INTERVAL):
        t_start = time.time()

        alg = GPRKR_Algorithm(
            problem=problem,
            N=N,
            n0=n0,
            K1=self.params['K1'],
            K2=self.params['K2'],
            lambda_i=self.params['lambda_i'],
            prior_var=self.params['prior_var'],
            w_vepm=self.params['w_vepm'],
            n_thr=self.params['n_thr'],
            seed=seed,
            partition_method=self.params.get('partition_method', 'binary_bin'),
            partition_K=self.params.get('partition_K', None),
            use_boundary_initial_design=self.params.get(
                'use_boundary_initial_design', True),
            use_archive_candidates=self.params.get(
                'use_archive_candidates', False),
            archive_neighbor_radius=self.params.get(
                'archive_neighbor_radius', 0),
            kg_selection_tiebreak=self.params.get(
                'kg_selection_tiebreak', 'crowding_distance'),
            variance_shrinkage_rho0=self.params.get(
                'variance_shrinkage_rho0', 0.0),
            variance_floor=self.params.get('variance_floor', 1e-8),
        )
        alg.instrument = self.instrument

        pareto_set = alg.run(verbose=False)
        total_time = time.time() - t_start
        self.instrument_log = alg.instrument_log

        # HV history is now computed on TRUE objectives during run
        hv_history_true = list(alg.hv_history)
        ref = problem.ref_point

        # Extract per-iteration computation times (excluding simulation)
        time_per_iter = []
        for log in alg.iteration_log:
            t_compute = (log.get('t_posterior_solve', 0) +
                         log.get('t_candidate_gen', 0) +
                         log.get('t_kg_compute', 0) +
                         log.get('t_belief_update', 0) +
                         log.get('t_vepm_update', 0) +
                         log.get('t_hv_eval', 0))
            time_per_iter.append(t_compute)

        result = self._make_result(
            pareto_solutions=pareto_set,
            problem=problem,
            hv_history=hv_history_true,
            total_time=total_time,
            time_per_iter=time_per_iter,
            n_simulations=N,
            ref_point=ref,
        )
        result["iteration_log"] = _json_safe(alg.iteration_log)
        result["observation_history"] = [
            {"x": [int(v) for v in x], "Y": [float(y) for y in Y]}
            for x, Y in alg.history
        ]
        result["instrument_log"] = _json_safe(alg.instrument_log)
        result["logging_detail"] = (
            "Per adaptive iteration: candidate_set, kg_pairs, selected point, "
            "observation, pre-update posterior mean/variance, timing, and HV "
            "snapshots when evaluated."
        )
        return result
