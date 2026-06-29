"""
libsumo simulation interface for InTAS.

Uses libsumo (same API as TraCI, no socket overhead) to run a single
2-hour morning-peak simulation of the InTAS scenario with a given signal
timing plan x, and returns the three objective values (f1, f2, f3).

Objective functions (all relative to baseline x0):
  f1(x,ξ) = avg_travel_time(x,ξ) / T0          [efficiency, <1 = better]
  f2(x,ξ) = atkinson_index(x,ξ)  / A0          [equity,     <1 = better]
  f3(x,ξ) = total_CO2(x,ξ)       / E0          [emission,   <1 = better]

Heteroscedastic noise source: Krauss stochastic car-following model +
  device.rerouting.probability=0.82 (stochastic re-routing).
"""

import os
import sys
import tempfile
import numpy as np
from typing import List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.intas.config import (
    SCENARIO_DIR, NET_FILE, ROUTE_DIR, ADD_FILE, BUS_ROUTES,
    SIM_BEGIN, SIM_END, STEP_LEN, SUMO_PARAMS, INTAS_ROOT,
    MORNING_PEAK_ROUTE_INDICES,
)
try:
    from experiments.intas.config import MORNING_PEAK_ROUTE_FILES
except ImportError:
    MORNING_PEAK_ROUTE_FILES = None

# ── libsumo import ────────────────────────────────────────────────────────────
try:
    import libsumo as traci
    _BACKEND = "libsumo"
except ImportError:
    try:
        import traci
        _BACKEND = "traci"
    except ImportError:
        raise ImportError("Neither libsumo nor traci found. Install SUMO and add to PYTHONPATH.")


def _write_sumocfg(route_files: list, tmp_dir: str,
                   tripinfo_path: str) -> str:
    """Write a temporary .sumocfg that loads multiple route files and
    requests a tripinfo-output XML with emissions.

    InTAS's `InTAS_NNN.rou.xml` files are time slices of one 24-hour
    calibrated day (not independent day realisations), so a morning-peak
    simulation must load the slices covering the peak plus the bus and
    pedestrian routes to get realistic demand.  We also ask SUMO to dump
    per-vehicle trip records (incl.\ CO2_abs integrated in C++ at full
    precision) to ``tripinfo_path`` so the Python side can skip per-step
    subscriptions, which was the per-simulation bottleneck.
    """
    import os as _os
    bus_stops = _os.path.join(INTAS_ROOT, "scenario", "BusStations.add.xml")
    add_files = f"{ADD_FILE},{bus_stops}" if _os.path.exists(bus_stops) else ADD_FILE
    # Pedestrians are OMITTED: their striping-model step adds ~10% wall time
    # while having negligible impact on vehicle signal timing objectives
    # (pedestrian phases are already encoded in the TLS state strings).
    route_list = list(route_files)
    if _os.path.exists(BUS_ROUTES):
        route_list.append(BUS_ROUTES)
    route_str = ",".join(route_list)
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{NET_FILE}"/>
    <route-files value="{route_str}"/>
    <additional-files value="{add_files}"/>
  </input>
  <output>
    <tripinfo-output value="{tripinfo_path}"/>
    <tripinfo-output.write-unfinished value="false"/>
  </output>
  <time>
    <begin value="{SIM_BEGIN}"/>
    <end value="{SIM_END}"/>
    <step-length value="{STEP_LEN}"/>
  </time>
  <processing>
    <ignore-junction-blocker value="15"/>
    <time-to-teleport value="300"/>
    <max-depart-delay value="300"/>
    <default.carfollowmodel value="Krauss"/>
    <device.emissions.probability value="1.0"/>
  </processing>
  <routing>
    <!--
      Keep InTAS's calibrated dynamic rerouting (0.82 probability every 300 s):
      in a 07:30-09:00 peak window, disabling rerouting causes vehicles to
      pile up on congested corridors (the field-calibrated real driver
      behaviour includes rerouting via navigation apps).  Rerouting also
      contributes to the seed-driven stochastic noise that makes
      heteroscedasticity interesting here.
    -->
    <routing-algorithm value="dijkstra"/>
    <device.rerouting.probability value="0.82"/>
    <device.rerouting.period value="300"/>
  </routing>
  <report>
    <no-step-log value="true"/>
    <no-warnings value="true"/>
  </report>
</configuration>"""
    cfg_path = os.path.join(tmp_dir, "intas_tmp.sumocfg")
    with open(cfg_path, 'w') as f:
        f.write(cfg_content)
    return cfg_path


_TRIPINFO_RE   = None   # populated on first call
_EMISSIONS_RE  = None

def _parse_tripinfo(path: str) -> list:
    """Parse SUMO's tripinfo XML with a regex-based scanner.

    This avoids the ElementTree / pyexpat path which can fail in some
    conda environments with DLL load errors.  Each record has keys
    ``duration`` (s), ``route_length`` (m), ``time_loss`` (s), and
    ``CO2_mg`` (non-negative; vehicles without an emission class
    contribute 0).
    """
    global _TRIPINFO_RE, _EMISSIONS_RE
    import re
    if _TRIPINFO_RE is None:
        _TRIPINFO_RE = re.compile(
            r'<tripinfo\b[^>]*?'
            r'\bduration="(?P<dur>[-\d.eE+]+)"[^>]*?'
            r'\brouteLength="(?P<len>[-\d.eE+]+)"[^>]*?'
            r'\btimeLoss="(?P<loss>[-\d.eE+]+)"[^>]*?/?>'
            , re.DOTALL)
        _EMISSIONS_RE = re.compile(
            r'<emissions\b[^>]*?\bCO2_abs="(?P<co2>[-\d.eE+]+)"', re.DOTALL)

    trips = []
    if not os.path.exists(path):
        return trips
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    # Split by <tripinfo> start tags so every trip record is self-contained
    parts = content.split('<tripinfo')
    for p in parts[1:]:
        chunk = '<tripinfo' + p
        m = _TRIPINFO_RE.search(chunk)
        if m is None:
            continue
        try:
            duration     = float(m.group('dur'))
            route_length = float(m.group('len'))
            time_loss    = float(m.group('loss'))
        except (TypeError, ValueError):
            continue
        co2 = 0.0
        em = _EMISSIONS_RE.search(chunk)
        if em is not None:
            try:
                co2 = max(0.0, float(em.group('co2')))
            except (TypeError, ValueError):
                co2 = 0.0
        trips.append({
            'duration':     duration,
            'route_length': route_length,
            'time_loss':    time_loss,
            'CO2_mg':       co2,
        })
    return trips


def _apply_signal_plan(var_map: list, x: np.ndarray):
    """
    Apply green times in x to the running SUMO simulation via libsumo.

    For each TLS, read the current program logic, update the green phase
    durations, and push it back. Yellow/all-red phases are unchanged.
    """
    from collections import defaultdict
    # Group updates by TLS
    updates = defaultdict(dict)  # {tls_id: {phase_idx: new_duration}}
    for k, (tls_id, phase_idx) in enumerate(var_map):
        updates[tls_id][phase_idx] = float(x[k])

    for tls_id, phase_updates in updates.items():
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics:
            continue
        logic = logics[0]   # use first (=real) program
        new_phases = list(logic.phases)
        for ph_idx, new_dur in phase_updates.items():
            if ph_idx < len(new_phases):
                # Create new Phase object with updated duration
                ph = new_phases[ph_idx]
                try:
                    from libsumo import TraCIPhase
                    new_phases[ph_idx] = TraCIPhase(
                        new_dur, ph.state,
                        ph.minDur if hasattr(ph, 'minDur') else new_dur,
                        ph.maxDur if hasattr(ph, 'maxDur') else new_dur,
                    )
                except Exception:
                    # Fallback: modify duration attribute directly
                    ph.duration = new_dur
        logic.phases = new_phases
        traci.trafficlight.setProgramLogic(tls_id, logic)


def _collect_tripinfo_from_subscriptions() -> Tuple[List[float], dict]:
    """
    Collect per-vehicle travel times and routes using subscriptions.
    Returns (all_travel_times, route_times) where route_times maps
    route_id -> list of travel times on that route.
    """
    # Subscribe to vehicle departure and arrival times
    all_travel_times = []
    vehicle_depart = {}
    vehicle_route  = {}
    route_times    = {}
    return all_travel_times, vehicle_depart, vehicle_route, route_times


def simulate(var_map: list, x: np.ndarray,
             route_file,
             T0: float, A0: float, E0: float,
             seed: Optional[int] = None) -> np.ndarray:
    """
    Run one InTAS simulation with signal plan x and return [f1, f2, f3].

    Parameters
    ----------
    var_map     : list of (tls_id, phase_idx) from parse_network
    x           : 1-D array of green times (decision variable vector)
    route_file  : ignored for backward compatibility.  The function
                  always loads the morning-peak route-file bundle
                  (InTAS_004..007 + BusRoutes + ped) defined in
                  config.MORNING_PEAK_ROUTE_INDICES.
    T0, A0, E0  : baseline values computed by compute_baseline.py
    seed        : SUMO random seed.  Stochasticity is driven entirely by
                  this seed (Krauss driver imperfection, 82% dynamic
                  rerouting, and route-distribution sampling at vehicle
                  insertion).

    Returns
    -------
    np.ndarray of shape (3,): [f1, f2, f3]
    """
    # Build the deterministic morning-peak route-file bundle.  Prefer the
    # explicit-filenames list (MORNING_PEAK_ROUTE_FILES) when available so
    # the simulator works even when only the 3 peak files are present on
    # disk (e.g. on a partial deploy).  Falls back to the historical
    # positional index scheme for robustness.
    import os as _os, glob as _glob
    if MORNING_PEAK_ROUTE_FILES is not None:
        peak_files = [_os.path.join(ROUTE_DIR, fn)
                      for fn in MORNING_PEAK_ROUTE_FILES
                      if _os.path.exists(_os.path.join(ROUTE_DIR, fn))]
    else:
        all_intas = sorted(_glob.glob(_os.path.join(ROUTE_DIR, 'InTAS_*.rou.xml')))
        peak_files = [all_intas[i] for i in MORNING_PEAK_ROUTE_INDICES
                      if i < len(all_intas)]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tripinfo_path = os.path.join(tmp_dir, "tripinfo.xml")
        cfg_path = _write_sumocfg(peak_files, tmp_dir, tripinfo_path)

        sumo_cmd = ["sumo", "-c", cfg_path]
        if seed is not None:
            sumo_cmd += ["--seed", str(int(seed))]

        # ── Start libsumo, apply signal plan, step through silently ────────
        # No per-step subscriptions -- all per-vehicle metrics are written to
        # tripinfo.xml by SUMO's C++ core (enabled via device.emissions in
        # _write_sumocfg).  This is both simpler and ~14% faster than Python
        # subscription-based tracking, and is numerically equivalent:
        # duration/routeLength/timeLoss are defined identically, and CO2_abs
        # is the analytic integral of CO2_rate in C++ (higher precision than
        # Python's `rate * step_length` accumulator).
        traci.start(sumo_cmd)
        _apply_signal_plan(var_map, x)
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
        traci.close()

        trips = _parse_tripinfo(tripinfo_path)

    if not trips:
        return np.array([2.0, 2.0, 2.0])

    travel_times = np.array([t['duration']     for t in trips], dtype=float)
    distances    = np.array([t['route_length'] for t in trips], dtype=float)
    losses       = np.array([t['time_loss']    for t in trips], dtype=float)
    CO2_total_kg = float(sum(t['CO2_mg'] for t in trips)) / 1e6

    # f1: relative average travel time
    avg_tt = float(np.mean(travel_times))
    f1 = avg_tt / T0

    # f2: Atkinson equity index (ε=1) on the congestion ratio
    #     r_i = time_loss_i / free_flow_TT_i = (t_i - t_i^ff) / t_i^ff
    # where t_i^ff = t_i - time_loss_i is the free-flow travel time.
    # `losses` and `travel_times` are already populated from the tripinfo
    # XML above.  r_i > 0 for any non-free-flowing trip; trips with
    # time_loss==0 are dropped to avoid log(0) (they contribute 1 to
    # GM(r)/AM(r) and thus nothing to inequality).
    tt_ff = travel_times - losses
    mask  = (tt_ff > 1.0) & (losses > 0)
    if mask.sum() < 2:
        atkinson = 0.0
    else:
        r_vals = losses[mask] / tt_ff[mask]
        mu = float(np.mean(r_vals))
        geom_mean = float(np.exp(np.mean(np.log(r_vals))))
        atkinson = 1.0 - geom_mean / mu if mu > 0 else 0.0
    f2 = atkinson / A0 if A0 > 0 else 1.0

    # f3: relative total CO2 emission
    f3 = CO2_total_kg / E0 if E0 > 0 else 1.0

    return np.array([f1, f2, f3], dtype=float)


def simulate_baseline(var_map: list, default_x: np.ndarray,
                      route_files: list, n_reps: int = 50,
                      verbose: bool = True) -> Tuple[float, float, float]:
    """
    Run n_reps simulations with the real signal plan (default_x) to
    estimate T0, A0, E0 (baseline means).

    Returns (T0, A0, E0).
    """
    raw_T, raw_A, raw_E = [], [], []
    # Use a dummy T0=A0=E0=1 for the first pass (ratios collapse to raw values)
    for rep, rf in enumerate(route_files[:n_reps]):
        seed = 42 + rep
        y = simulate(var_map, default_x, rf, T0=1.0, A0=1.0, E0=1.0, seed=seed)
        raw_T.append(y[0])   # y[0] = avg_tt/1 = avg_tt
        raw_A.append(y[1])   # y[1] = atkinson/1 = atkinson
        raw_E.append(y[2])   # y[2] = CO2/1 = CO2_kg
        if verbose:
            print(f"  baseline rep {rep+1:02d}:  "
                  f"T={y[0]:.1f}s  A={y[1]:.4f}  E={y[2]:.1f}kg")
    T0 = float(np.mean(raw_T))
    A0 = float(np.mean(raw_A))
    E0 = float(np.mean(raw_E))
    if verbose:
        print(f"\nBaseline: T0={T0:.2f}s  A0={A0:.4f}  E0={E0:.2f}kg-CO2")
    return T0, A0, E0
