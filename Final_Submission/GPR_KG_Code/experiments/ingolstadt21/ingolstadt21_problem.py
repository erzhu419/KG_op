"""
Ingolstadt21Problem — GPR-KG compatible interface for the RESCO subnet.

Same API as InTASProblem (so methods/run_main.py only needs an import
swap), but parameterized for the smaller, signal-sensitive subnet.
"""

import os
import sys
import json
import random
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ingolstadt21.config import (
    ROUTE_FILE, RESULTS_DIR, TAU_EMISSION, ALPHA,
    DEFAULT_N, DEFAULT_N0,
)


class Ingolstadt21Problem:
    """RESCO ingolstadt21 signal-timing optimization problem."""

    name = "ingolstadt21"

    def __init__(self,
                 decision_space_json: Optional[str] = None,
                 baseline_json: Optional[str] = None,
                 tau: float = TAU_EMISSION,
                 alpha: float = ALPHA):

        ds_path = decision_space_json or os.path.join(RESULTS_DIR, 'decision_space.json')
        bl_path = baseline_json       or os.path.join(RESULTS_DIR, 'baseline.json')

        with open(ds_path) as f:
            ds = json.load(f)
        self.d        = ds['d']
        self.var_map  = [tuple(v) for v in ds['var_map']]
        self._bounds  = ds['bounds']
        self._default_x = np.array(ds['defaults'])

        with open(bl_path) as f:
            bl = json.load(f)
        self.T0 = float(bl['T0'])
        self.A0 = float(bl['A0'])
        self.E0 = float(bl['E0'])

        self.tau   = float(tau)
        self.alpha = float(alpha)
        self.L     = None
        self.ref_point = np.array([1.5, 1.5])

        # Single route file (vs InTAS multi-file).  Stored as a 1-element
        # list so simulate() can reuse the InTAS-style random.choice path
        # without branching.
        self._route_files = [ROUTE_FILE]
        self._rng = np.random.default_rng()

        from experiments.ingolstadt21 import sumo_sim
        self._sim = sumo_sim

    # ── Bounds interface ────────────────────────────────────────────────────

    def int_bounds(self):
        lo = np.array([int(np.ceil(b[0]))  for b in self._bounds])
        hi = np.array([int(np.floor(b[1])) for b in self._bounds])
        return lo, hi

    # ── Sampling interface ──────────────────────────────────────────────────

    def sample_random(self) -> tuple:
        lo, hi = self.int_bounds()
        x = tuple(int(self._rng.integers(lo[k], hi[k]+1)) for k in range(self.d))
        return x

    def continuous_to_int(self, x_cont: np.ndarray) -> tuple:
        lo, hi = self.int_bounds()
        x_int = np.round(lo + x_cont * (hi - lo)).astype(int)
        x_int = np.clip(x_int, lo, hi)
        return tuple(int(v) for v in x_int)

    def normalize(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        lo, hi = self.int_bounds()
        rng = np.maximum(hi - lo, 1.0)
        return np.clip((x - lo) / rng, 0.0, 1.0)

    # ── Simulation interface ────────────────────────────────────────────────

    def simulate(self, x) -> np.ndarray:
        x_arr = np.array(x, dtype=float)
        seed = int(self._rng.integers(0, 10000))
        y = self._sim.simulate(
            var_map=self.var_map,
            x=x_arr,
            route_file=self._route_files[0],
            T0=self.T0, A0=self.A0, E0=self.E0,
            seed=seed,
        )
        return y.astype(float)

    # ── True-value / feasibility interface (for instrumentation only) ───────

    def true_objectives(self, x) -> np.ndarray:
        R = 10
        ys = np.array([self.simulate(x) for _ in range(R)])
        return ys.mean(axis=0)

    def true_sigma(self, x) -> np.ndarray:
        R = 20
        ys = np.array([self.simulate(x) for _ in range(R)])
        return ys.std(axis=0, ddof=1)

    def is_truly_feasible(self, x) -> bool:
        R = 30
        f3_vals = np.array([self.simulate(x)[2] for _ in range(R)])
        return float(np.mean(f3_vals <= self.tau)) >= (1 - self.alpha)

    @property
    def default_x(self) -> np.ndarray:
        return self._default_x.copy()

    def __repr__(self):
        return (f"Ingolstadt21Problem(d={self.d}, tau={self.tau}, alpha={self.alpha}, "
                f"T0={self.T0:.1f}s, A0={self.A0:.4f}, E0={self.E0:.1f}kg-CO2)")
