"""
RESCO ingolstadt21 experiment configuration.

A focused 21-TLS subnet of Ingolstadt extracted by Ault & Sharon (2021)
from InTAS for benchmarking signal control algorithms.  Compared to our
prior full-InTAS setup, this scenario:
  - has fixed-time baseline plans (sub-optimal => visible improvement room)
  - turns rerouting OFF by default (signal effects are not absorbed)
  - uses a 1h PM peak window (16:00-17:00) over 4283 trips
  - has 21 controlled TLS = entire study area

Local verification (3 sims, baseline / halved / doubled greens):
  f1 swing = 12.95% (vs 1.44% on full InTAS — 9x more responsive)
  f2 (Atkinson) swing = 4.99% (similar to InTAS)
  f3 (CO2)    swing = 8.96% (vs <1% on full InTAS — 10x more responsive)
  per-sim wall = 22-26 s (vs 14 min on InTAS)
"""

import os

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
# SUMO cannot parse non-ASCII characters in net/route paths under Windows GBK
# locales (the project root path contains Chinese characters), so the network
# data lives under an ASCII-only mirror.  Override with INGOLSTADT21_DATA_DIR
# if needed; bootstrap_data() copies the files there if missing.
_PROJECT_DATA_MIRROR = os.path.join(_HERE, "data")  # in-tree copy for git tracking


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

# Results go here (separate dir from InTAS so the two case studies don't collide)
RESULTS_DIR  = os.path.join(
    os.path.dirname(os.path.dirname(_HERE)),
    "results", "ingolstadt21")

# ── Simulation window (RESCO original: 16:00-17:00, 1 hour) ─────────────────
SIM_BEGIN = 57600    # 16:00:00
SIM_END   = 61200    # 17:00:00
STEP_LEN  = 1.0      # SUMO default; same as InTAS

# ── Optimization budget (matches Phase 2 InTAS protocol) ────────────────────
DEFAULT_N   = 300    # sequential KG evaluations
DEFAULT_N0  = 100    # pre-samples
N_BASELINE  = 50     # replications for T0/A0/E0

# ── Constraint ───────────────────────────────────────────────────────────────
# Initial pilot at tau=0.95 produced 0/200 obs satisfying f3<=0.95 in the KG
# search (because LHS-and-KG sampling rarely hits the "all-dim halved" corner
# that the verification probe found).  Switching to tau=1.0 (do-no-harm) makes
# the chance constraint reachable while still being meaningful: any plan that
# improves f1 or f2 must not increase CO2 above the fixed-time baseline with
# 95% confidence.
TAU_EMISSION = 1.0
ALPHA        = 0.05

# ── Signal timing bounds (seconds) ───────────────────────────────────────────
# RESCO ingolstadt21 default green-time distribution: min=5s, median=29s,
# max=42s, n=66.  The 5-15s phases are typically brief protected turn phases;
# we filter them as auxiliary (kept frozen at default in sumo_sim) so the
# decision space contains only main vehicle-green phases.
GREEN_MIN = 10    # minimum green time per phase (lower than InTAS 15 to fit
                  # ingolstadt21's tighter default greens)
GREEN_MAX = 90    # maximum green time per phase
YELLOW    = 3     # fixed yellow duration (not optimized)
ALL_RED   = 2     # fixed all-red clearance (not optimized)

# ── Decision-space filter (mirrors InTAS Plan D, tuned to ingolstadt21) ──────
MAIN_PHASE_MIN_GREEN = 15    # only optimize phases with default green >= 15s
BOUND_LO_FRAC        = 0.5   # lb = 0.5 * default green
BOUND_HI_FRAC        = 1.8   # ub = 1.8 * default green

# ── SUMO parameters ──────────────────────────────────────────────────────────
# Keep ingolstadt21 ORIGINAL config: only override emissions (for f3) and
# tripinfo output (to extract per-vehicle metrics).  Do NOT add InTAS-style
# max-depart-delay / time-to-teleport / rerouting overrides — these would
# break parity with the published RESCO benchmark.
SUMO_PARAMS = {
    "device.emissions.probability": "1.0",     # required for f3 (CO2)
    "no-step-log":                  "true",
    "no-warnings":                  "true",
}

# ── Emission weighting (HBEFA3 CO2-equivalent, same as InTAS) ────────────────
EMISSION_WEIGHTS = {"CO2": 1.0}
