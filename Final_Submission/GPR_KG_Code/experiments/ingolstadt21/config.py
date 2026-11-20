"""
RESCO ingolstadt21 experiment configuration.

A focused 21-TLS urban arterial subnetwork of Ingolstadt from Ault & Sharon (2021)
for benchmarking signal control algorithms:
  - uses published fixed-time baseline plans
  - keeps rerouting OFF by default so signal effects are directly observed
  - simulates a 1-hour PM peak window (16:00-17:00) over 4,283 trips
  - controls 21 signalized intersections (44 main vehicle-green phases)
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
# Store scenario XML inputs in an ASCII-safe directory; override via
# INGOLSTADT21_DATA_DIR if needed. bootstrap_data() copies files if missing.
_PROJECT_DATA_MIRROR = os.path.join(_HERE, "data")


def _default_data_dir() -> str:
    override = os.environ.get("INGOLSTADT21_DATA_DIR")
    if override:
        return override
    if os.name == "nt":
        return r"C:\sumo_scenarios\ingolstadt21"
    # On Linux/WSL the repository path is ASCII-safe and avoids creating a
    # literal "C:\..." directory relative to the current working directory.
    return _PROJECT_DATA_MIRROR


DATA_DIR     = _default_data_dir()
NET_FILE     = os.path.join(DATA_DIR, "ingolstadt21.net.xml")
ROUTE_FILE   = os.path.join(DATA_DIR, "ingolstadt21.rou.xml")


def bootstrap_data():
    """Copy ingolstadt21 net+route files to the ASCII-safe DATA_DIR if missing.

    Called automatically by sumo_sim.simulate() and compute_baseline before
    SUMO is launched; idempotent.
    """
    import shutil
    if os.path.exists(NET_FILE) and os.path.exists(ROUTE_FILE):
        return
    os.makedirs(DATA_DIR, exist_ok=True)
    for fname in ("ingolstadt21.net.xml", "ingolstadt21.rou.xml"):
        src = os.path.join(_PROJECT_DATA_MIRROR, fname)
        dst = os.path.join(DATA_DIR, fname)
        if os.path.abspath(src) == os.path.abspath(dst):
            continue
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy2(src, dst)

# Results directory for scenario metadata (decision_space.json, baseline.json)
RESULTS_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "results", "ingolstadt21")

# ── Simulation window (RESCO original: 16:00-17:00, 1 hour) ─────────────────
SIM_BEGIN = 57600    # 16:00:00
SIM_END   = 61200    # 17:00:00
STEP_LEN  = 1.0      # SUMO 1.0s simulation step

# ── Default reference budgets ────────────────────────────────────────────────
DEFAULT_N   = 300    # sequential evaluations
DEFAULT_N0  = 100    # initial evaluations
N_BASELINE  = 50     # replications for T0/A0/E0

# ── Constraint ───────────────────────────────────────────────────────────────
# Relative CO2 ceiling tau=1.0 (do-no-harm relative to fixed-time baseline)
# under chance-constraint risk level alpha=0.05.
TAU_EMISSION = 1.0
ALPHA        = 0.05

# ── Signal timing bounds (seconds) ───────────────────────────────────────────
# RESCO ingolstadt21 default green-time distribution: min=5s, median=29s,
# max=42s, n=66.  Phases < 15s are auxiliary protected turn phases kept fixed
# at default so the decision space optimizes the 44 main vehicle-green phases.
GREEN_MIN = 10    # minimum green time per phase
GREEN_MAX = 90    # maximum green time per phase
YELLOW    = 3     # fixed yellow duration (not optimized)
ALL_RED   = 2     # fixed all-red clearance (not optimized)

# ── Decision-space filter ────────────────────────────────────────────────────
MAIN_PHASE_MIN_GREEN = 15    # only optimize phases with default green >= 15s
BOUND_LO_FRAC        = 0.5   # lb = 0.5 * default green
BOUND_HI_FRAC        = 1.8   # ub = 1.8 * default green

# ── SUMO parameters ──────────────────────────────────────────────────────────
# Keep published RESCO ingolstadt21 configuration: enable emissions output
# (for CO2) and tripinfo output (for per-vehicle delay and emission metrics).
SUMO_PARAMS = {
    "device.emissions.probability": "1.0",     # required for f3 (CO2)
    "no-step-log":                  "true",
    "no-warnings":                  "true",
}

# ── Emission weighting (HBEFA3 CO2-equivalent) ───────────────────────────────
EMISSION_WEIGHTS = {"CO2": 1.0}
