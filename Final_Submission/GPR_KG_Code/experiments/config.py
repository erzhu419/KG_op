"""
Shared configuration for experiment drivers.

The reported RZDT1, RZDT2, and RZDT5_RR experiments override the legacy
problem defaults with d=5, N=150, n0=30, alpha=0.05, tau=0, and baseline
heteroscedastic noise scale sigma=0.04. See run_d5_v2.py for RZDT1/2
and run_rzdt5rr.py for RZDT5_RR.
"""

import numpy as np

# --- Simulation budget ---
DEFAULT_N = 150       # Total adaptive simulation budget (paper Section 6.1)
DEFAULT_N0 = 30       # Pre-sampling budget (~20% of N)

# --- Decision space ---
DEFAULT_D = 5         # Default dimension
DEFAULT_L = 20        # Legacy default; reported RZDT1/2 use L=100

# --- Noise ---
DEFAULT_SIGMA = 0.1   # Legacy default; reported RZDT1/2/5_RR override to 0.04
DEFAULT_ALPHA = 0.05  # Constraint confidence level (95%)

# --- HV reference point ---
REF_POINT = np.array([1.5, 1.5])          # For RZDT1/2/3/4/6 (PF in [0,1]^2)
REF_POINT_RZDT5 = np.array([2.5, 1.5])    # For RZDT5 (f1 in [1,2], f2 in [0.5,1])

# --- Replications ---
N_REPS = 10           # Independent replications per method per problem
SEED_BASE = 1000      # Starting random seed

# --- Constraint feasibility levels (Section 6.4) ---
FEASIBILITY_LOOSE = 0.9     # ~90% feasible
FEASIBILITY_MODERATE = 0.5  # ~50% feasible (default)
FEASIBILITY_TIGHT = 0.1     # ~10% feasible

# --- Noise levels (Section 6.5) ---
NOISE_LEVELS = [0.01, 0.1, 1.0]  # Homoscedastic levels
# Heteroscedastic: sigma_i(x) = 0.1 + 0.9*|f_i(x)|

# --- Scalability dimensions (Section 6.3) ---
SCALABILITY_DIMS = [5, 10, 20]

# --- GPR-KG specific parameters (reported RZDT experiments) ---
GPR_KG_PARAMS = {
    'K1': 50,           # LHD random candidates (M_rand in paper)
    'K2': 2,            # Posterior sampling + NSGA-II iterations
                        # Each run: pop=100, gen=50 → ~20 Pareto candidates (M_post in paper)
    'lambda_i': 0.1,    # Prior variance for solution-specific deviation terms
    'prior_var': 100.0, # Diffuse prior variance for parametric beta coefficients
    'w_vepm': 1.0,      # VEPM prior weight (equivalent to 1 pseudo-observation per bin)
    'n_thr': 20,        # Iterations before activating constraint in candidate gen
    'partition_features': 'auto',  # Use registry-recommended variance features
    # Structured finite-grid initial design. This uses only design-space bounds
    # (center and coordinate-axis endpoints) plus random grid points; it does
    # not use objective/constraint values and does not alter KG after
    # initialization.
    'use_boundary_initial_design': True,
    # Optional diagnostic variants retained in code but disabled by default.
    'use_archive_candidates': False,
    'archive_neighbor_radius': 0,
    'kg_selection_tiebreak': 'crowding_distance',
    'variance_shrinkage_rho0': 0.0,
    'variance_floor': 1e-8,
    # Finite-budget robust VEPM controls. Disabled by default so historical
    # GPR-KG runs remain exactly reproducible unless the driver passes
    # --robust_vepm explicitly.
    'robust_vepm': False,
    'vepm_residual_clip_factor': None,
    'vepm_new_point_weight': 1.0,
    'vepm_partition_weight_floor': 0.0,
    # Main-paper upgrade candidates.  Disabled by default so historical
    # results remain reproducible; enabled through the GPR-KG-AVB runner.
    'adaptive_vepm': False,
    'adaptive_vepm_max_features': 2,
    'adaptive_vepm_min_score': 0.0,
    'vepm_shrinkage_kappa': 0.0,
}

# --- Inner NSGA-II parameters (used inside GPR-KG candidate generation) ---
INNER_NSGA2_PARAMS = {
    'pop_size': 100,    # Population size for posterior-sampled NSGA-II
    'n_gen': 50,        # Generations; 100*50=5000 surrogate evals per run (cheap)
}

# --- NSGA-II-D and NSGA-II-K parameters ---
NSGA2_PARAMS = {
    'pop_size': 100,    # Population size for NSGA-II-D/K
    'crossover_prob': 0.9,
    'mutation_prob': None,  # Will be set to 1/d at runtime
    'crossover_eta': 20,    # SBX distribution index (higher = offspring closer to parents)
    'mutation_eta': 20,     # Polynomial mutation distribution index
}

# --- NSGA-II-D budget multiplier ---
NSGA2D_BUDGET_MULT = 10   # NSGA-II-D gets 10x budget = 1500 evaluations for d=5

# --- HV evaluation interval ---
HV_EVAL_INTERVAL = 10  # Compute HV every 10 iterations for convergence curves

# --- Timeout ---
TIMEOUT_72H = 72 * 3600  # 72-hour time limit for scalability experiments

# --- Method names (consistent with paper) ---
METHOD_NAMES = [
    'GPR-KG',
    'GPR-KG-nV',
    'cEHVI',
    'cParEGO',
    'NSGA-II-K',
    'NSGA-II-D',
    'RS',
]

# --- Test problem registry ---
# All 6 heteroscedastic benchmark problems
ALL_PROBLEMS = ['RZDT1', 'RZDT2', 'RZDT3', 'RZDT4', 'RZDT5', 'RZDT6']

# RZDT1: convex PF, sqrt noise (25x variance ratio)
# RZDT2: concave PF, sin^2 bell noise (36x variance ratio)
# RZDT3: 5-band PF, U-shaped noise (16x variance ratio)
# RZDT4: multi-modal, exponential noise (81x variance ratio)
# RZDT5: 3-point discrete PF, quadratic noise (59x variance ratio)
# RZDT6: non-uniform non-convex PF, linear gradient noise (69x variance ratio)

# HV reference point covers all 6 problems' Pareto fronts
# RZDT1/2/4/6: PF in [0,1]^2; RZDT3: f2 can reach -0.77; RZDT5: f1 in [1,2]
# Use [2.1, 1.5] for RZDT5 or the default [1.5, 1.5] for RZDT1-4/6
HETERO_REF_POINT = np.array([1.5, 1.5])  # same as standard (PF subset of [0,1]^2)

def get_problem(name, d=DEFAULT_D, L=DEFAULT_L, sigma=DEFAULT_SIGMA,
                heteroscedastic=False, alpha=DEFAULT_ALPHA):
    """Instantiate a test problem by name.

    Parameters
    ----------
    name : str
        One of 'RZDT1', 'RZDT2', 'RZDT3', 'RZDT4', 'RZDT5', 'RZDT6'.
    d, L, sigma, heteroscedastic, alpha : see TestProblem.__init__.

    Returns
    -------
    TestProblem instance (not yet calibrated — call calibrate_constraint).
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from gpr_kg import RZDT1, RZDT2, RZDT5, RZDT3, RZDT4, RZDT6
    _registry = {
        'RZDT1': RZDT1, 'RZDT2': RZDT2, 'RZDT5': RZDT5,
        'RZDT3': RZDT3, 'RZDT4': RZDT4, 'RZDT6': RZDT6,
    }
    if name not in _registry:
        raise ValueError(f"Unknown problem '{name}'. "
                         f"Available: {list(_registry.keys())}")
    return _registry[name](d=d, L=L, sigma=sigma,
                           heteroscedastic=heteroscedastic, alpha=alpha)
