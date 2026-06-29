"""
InTAS experiment configuration.

Before running:
  1. Clone InTAS: git clone https://github.com/silaslobo/InTAS
  2. Set INTAS_ROOT below to the cloned directory
  3. Run compute_baseline.py once to get T0, A0, E0
"""

import os

# ── Path configuration ──────────────────────────────────────────────────────
INTAS_ROOT   = os.environ.get("INTAS_ROOT",
               r"C:\InTAS")          # override via env var or edit here
SCENARIO_DIR = os.path.join(INTAS_ROOT, "scenario")
NET_FILE     = os.path.join(SCENARIO_DIR, "ingolstadt.net.xml")
ROUTE_DIR    = os.path.join(SCENARIO_DIR, "routes")
ADD_FILE     = os.path.join(SCENARIO_DIR, "InTAS_E1.add.xml")
BUS_ROUTES   = os.path.join(ROUTE_DIR, "BusRoutes.flow.xml")

# Results go here
RESULTS_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results", "intas")

# ── Simulation window ────────────────────────────────────────────────────────
# Morning peak 07:30-09:00 (1.5 hours).  Covers peak onset through the most
# congested hour; 09:00 cutoff keeps a 30-min tail-cleanup period short of
# the deeper recovery phase.
SIM_BEGIN = 7 * 3600 + 30 * 60    # 27000 s  (07:30)
SIM_END   = 9 * 3600              # 32400 s  (09:00)
STEP_LEN  = 1.0                    # seconds (coarser than InTAS 0.1 for speed)

# ── Route files relevant to morning peak ─────────────────────────────────────
# Each InTAS_NNN.rou.xml is a time-slice of one calibrated 24-hour day,
# NOT a separate day realisation.  For the 07:30-09:00 window we load
# three slices whose vehicles can depart within the window (accounting
# for max-depart-delay=300s):
#   InTAS_005 (07:10-07:42): 4557 vehicles in window
#   InTAS_006 (07:42-08:26): 8450 vehicles (all in window)
#   InTAS_007 (08:26-09:27): ~9200 vehicles in window (up to 09:00)
# Using explicit filenames makes the code robust to partial transfers
# (e.g. only the 3 needed files on a deploy server) without relying on
# positional indices into a sorted glob.
MORNING_PEAK_ROUTE_FILES = [
    'InTAS_005.rou.xml',
    'InTAS_006.rou.xml',
    'InTAS_007.rou.xml',
]
# Backwards compat: keep the index form around in case other code reads it.
MORNING_PEAK_ROUTE_INDICES = [4, 5, 6]

# ── Optimization budget (Phase 2, Zheng 2019 §5.2 protocol) ─────────────────
DEFAULT_N   = 300    # sequential KG evaluations (total budget = N0 + N = 400)
DEFAULT_N0  = 100    # pre-samples (LHS)
N_BASELINE  = 50     # replications for computing T0, A0, E0

# ── Constraint ───────────────────────────────────────────────────────────────
# τ=1.0 is a "do-no-harm" emission constraint: at the InTAS network the SUMO
# default plan is already near-optimal in CO2 (max 0.84% reduction across
# 800 plans), so τ<1.0 is effectively infeasible.  τ=1.0 forces the optimizer
# to find travel-time / equity gains without inflating CO2 above baseline.
TAU_EMISSION = 1.0
ALPHA        = 0.05

# ── Signal timing bounds (seconds) ───────────────────────────────────────────
GREEN_MIN = 15    # minimum green time per phase
GREEN_MAX = 90    # maximum green time per phase
YELLOW    = 3     # fixed yellow duration (not optimized)
ALL_RED   = 2     # fixed all-red clearance (not optimized)

# ── Decision-space filter (Plan D: 36-dim main-phase optimization) ──────────
# 88-dim raw parse includes auxiliary short phases (default <20s) whose bounds
# span ≤5s — they cannot meaningfully affect the objectives.  Filter to main
# vehicle-green phases (default ≥20s); these become the optimization variables.
# Bounds: [0.5 g0, 1.8 g0] clipped by [GREEN_MIN, GREEN_MAX] (Plan D).
MAIN_PHASE_MIN_GREEN = 20    # only optimize phases with default green ≥ 20s
BOUND_LO_FRAC        = 0.5   # lower bound = 0.5 × default green
BOUND_HI_FRAC        = 1.8   # upper bound = 1.8 × default green

# ── InTAS simulation parameters (from paper Table 5) ─────────────────────────
SUMO_PARAMS = {
    "ignore-junction-blocker":    "15",
    "time-to-teleport":           "300",
    "default.carfollowmodel":     "Krauss",
    "routing-algorithm":          "dijkstra",
    "device.rerouting.probability": "0.82",
    "device.rerouting.period":    "300",
    "no-step-log":                "true",
    "no-warnings":                "true",
}

# ── Emission weighting (HBEFA3 CO2-equivalent, following citation) ────────────
# We use total CO2 output from SUMO's built-in emission model (HBEFA3).
# Composite index = CO2 only (dominant greenhouse gas, >95% of CO2-equiv).
# Extend to NOx/PM via EMISSION_WEIGHTS if needed.
EMISSION_WEIGHTS = {"CO2": 1.0}   # can add "NOx": 298.0, "PMx": 1000.0 for CO2-equiv
