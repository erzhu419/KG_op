"""
InTAS problem class — GPR-KG compatible interface.

Wraps the InTAS SUMO simulation as a stochastic bi-objective
chance-constrained optimisation problem:

  min  [f1(x,ξ), f2(x,ξ)]     (efficiency, equity — both relative)
  s.t. P(f3(x,ξ) ≤ τ) ≥ 1-α  (emission chance constraint)

Decision variables: x ∈ Z^d, where d = total number of vehicle-green
phases across the 20 real Ingolstadt TLS.  Each component x_k is the
green time (rounded to nearest integer second) for the k-th phase.

The class follows the same interface as RZDT1/RZDT2/RZDT5_RR so that
gpr_kg.py / methods/ can call it unchanged.
"""

import os
import sys
import json
import glob
import random
import numpy as np
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.intas.config import (
    ROUTE_DIR, RESULTS_DIR, TAU_EMISSION, ALPHA,
    SIM_BEGIN, SIM_END, DEFAULT_N, DEFAULT_N0,
)


class InTASProblem:
    """
    InTAS network signal timing optimisation problem.

    Parameters
    ----------
    decision_space_json : str
        Path to decision_space.json produced by parse_network.save_decision_space()
    baseline_json : str
        Path to baseline.json produced by compute_baseline.compute_and_save()
    tau : float
        Emission threshold (default 0.90 — require 10% reduction)
    alpha : float
        Significance level of chance constraint (default 0.05)
    """

    name = "InTAS"

    def __init__(self,
                 decision_space_json: Optional[str] = None,
                 baseline_json: Optional[str] = None,
                 tau: float = TAU_EMISSION,
                 alpha: float = ALPHA):

        ds_path = decision_space_json or os.path.join(RESULTS_DIR, 'decision_space.json')
        bl_path = baseline_json       or os.path.join(RESULTS_DIR, 'baseline.json')

        # ── Load decision space ─────────────────────────────────────────────
        with open(ds_path) as f:
            ds = json.load(f)
        self.d        = ds['d']
        self.var_map  = [tuple(v) for v in ds['var_map']]   # (tls_id, phase_idx)
        self._bounds  = ds['bounds']                          # [[lb, ub], ...]
        self._default_x = np.array(ds['defaults'])            # real Ingolstadt plan

        # ── Load baseline ───────────────────────────────────────────────────
        with open(bl_path) as f:
            bl = json.load(f)
        self.T0 = float(bl['T0'])
        self.A0 = float(bl['A0'])
        self.E0 = float(bl['E0'])

        # ── Constraint / reference point ────────────────────────────────────
        self.tau   = float(tau)
        self.alpha = float(alpha)
        self.L     = None   # not a grid problem; keep None
        # Reference HV point: slightly worse than worst observed (1.5, 1.5)
        self.ref_point = np.array([1.5, 1.5])

        # ── Route files (demand scenarios) ─────────────────────────────────
        all_routes = sorted(glob.glob(os.path.join(ROUTE_DIR, 'InTAS_*.rou.xml')))
        if not all_routes:
            raise FileNotFoundError(
                f"No route files found in {ROUTE_DIR}. "
                "Check INTAS_ROOT in experiments/intas/config.py")
        self._route_files = all_routes
        self._rng = np.random.default_rng()

        # ── Import simulator lazily ─────────────────────────────────────────
        from experiments.intas import sumo_sim
        self._sim = sumo_sim

    # ── Bounds interface ────────────────────────────────────────────────────

    def int_bounds(self):
        """Return (lo, hi) integer arrays of shape (d,)."""
        lo = np.array([int(np.ceil(b[0]))  for b in self._bounds])
        hi = np.array([int(np.floor(b[1])) for b in self._bounds])
        return lo, hi

    # ── Sampling interface ──────────────────────────────────────────────────

    def sample_random(self) -> tuple:
        """Uniform random sample from the integer decision space."""
        lo, hi = self.int_bounds()
        x = tuple(int(self._rng.integers(lo[k], hi[k]+1)) for k in range(self.d))
        return x

    def continuous_to_int(self, x_cont: np.ndarray) -> tuple:
        """Map continuous [0,1]^d to integer decision space."""
        lo, hi = self.int_bounds()
        x_int = np.round(lo + x_cont * (hi - lo)).astype(int)
        x_int = np.clip(x_int, lo, hi)
        return tuple(int(v) for v in x_int)

    def normalize(self, x) -> np.ndarray:
        """Map integer x to [0,1]^d via (x - lo) / (hi - lo).

        Required by VEPM partitioning (gpr_kg.py expects this method on
        every problem class).  For InTAS, bounds vary per-dimension, so
        per-dim min-max normalisation is used.
        """
        x = np.asarray(x, dtype=float)
        lo, hi = self.int_bounds()
        rng = np.maximum(hi - lo, 1.0)   # avoid div-by-zero on degenerate dims
        return np.clip((x - lo) / rng, 0.0, 1.0)

    # ── Simulation interface ────────────────────────────────────────────────

    def simulate(self, x) -> np.ndarray:
        """
        Run one SUMO simulation with signal plan x.

        x can be a tuple, list, or np.ndarray of integer green times.

        Returns np.ndarray of shape (3,): [f1, f2, f3] + noise.
        """
        x_arr = np.array(x, dtype=float)
        # Pick a random demand scenario (stochastic ξ)
        route_file = random.choice(self._route_files)
        seed = int(self._rng.integers(0, 10000))
        y = self._sim.simulate(
            var_map=self.var_map,
            x=x_arr,
            route_file=route_file,
            T0=self.T0, A0=self.A0, E0=self.E0,
            seed=seed,
        )
        return y.astype(float)

    # ── True-value interface (for instrumentation / evaluation) ─────────────

    def true_objectives(self, x) -> np.ndarray:
        """
        Estimate true expected objectives at x via R=10 replications.
        Used only by the instrumentation framework (not during optimisation).
        """
        R = 10
        ys = np.array([self.simulate(x) for _ in range(R)])
        return ys.mean(axis=0)

    def true_sigma(self, x) -> np.ndarray:
        """
        Estimate noise std at x via R=20 replications.
        Used only by the instrumentation framework.
        """
        R = 20
        ys = np.array([self.simulate(x) for _ in range(R)])
        return ys.std(axis=0, ddof=1)

    # ── Feasibility interface ───────────────────────────────────────────────

    def is_truly_feasible(self, x) -> bool:
        """
        Estimate true feasibility P(f3 ≤ τ) ≥ 1-α via 30 replications.
        Used for evaluation only; not called during optimisation.
        """
        R = 30
        f3_vals = np.array([self.simulate(x)[2] for _ in range(R)])
        return float(np.mean(f3_vals <= self.tau)) >= (1 - self.alpha)

    # ── Utility ────────────────────────────────────────────────────────────

    @property
    def default_x(self) -> np.ndarray:
        """The real Ingolstadt signal plan (baseline)."""
        return self._default_x.copy()

    def __repr__(self):
        lo, hi = self.int_bounds()
        return (f"InTASProblem(d={self.d}, τ={self.tau}, α={self.alpha}, "
                f"T0={self.T0:.1f}s, A0={self.A0:.4f}, E0={self.E0:.1f}kg-CO2)")
