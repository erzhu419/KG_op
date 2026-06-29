"""
libsumo simulation interface for RESCO ingolstadt21.

Runs a 1-hour PM-peak simulation with the given signal-timing plan x and
returns the three objective values (f1, f2, f3).

Objective functions (relative to baseline plan x_0 with T0/A0/E0 measured
on the SAME ingolstadt21 setup):
  f1(x,xi) = avg_travel_time(x,xi) / T0    (efficiency, <1 = better)
  f2(x,xi) = atkinson_index(x,xi)  / A0    (equity,     <1 = better)
  f3(x,xi) = total_CO2(x,xi)       / E0    (emission,   <1 = better)

Compared to the InTAS sumo_sim.py:
  - Single route file (no random draw across InTAS_NNN.rou.xml)
  - No bus / pedestrian routes loaded
  - No additional files (RESCO ships net+routes only)
  - SUMO defaults for time-to-teleport / max-depart-delay / car-follow model
    (matches the published RESCO benchmark; InTAS-style overrides removed)
  - Stochasticity comes from the SUMO seed only (no rerouting randomness)
"""

import os
import sys
import tempfile
import numpy as np
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ingolstadt21.config import (
    NET_FILE, ROUTE_FILE, SIM_BEGIN, SIM_END, STEP_LEN, bootstrap_data,
)
bootstrap_data()    # idempotent: copies net/rou to ASCII-only DATA_DIR if missing

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


def _write_sumocfg(tmp_dir: str, tripinfo_path: str) -> str:
    """Write a temp sumocfg matching RESCO original config + emissions + tripinfo.

    No <processing> overrides for SUMO defaults (time-to-teleport=300,
    max-depart-delay=-1, carfollowmodel=Krauss); only what we strictly need:
      * tripinfo-output (extract per-vehicle metrics)
      * device.emissions.probability=1.0 (populate CO2_abs in tripinfo)
    """
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
  <input>
    <net-file value="{NET_FILE}"/>
    <route-files value="{ROUTE_FILE}"/>
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
    <device.emissions.probability value="1.0"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <no-warnings value="true"/>
  </report>
</configuration>"""
    cfg_path = os.path.join(tmp_dir, "ingolstadt21_tmp.sumocfg")
    with open(cfg_path, 'w') as f:
        f.write(cfg_content)
    return cfg_path


_TRIPINFO_RE  = None
_EMISSIONS_RE = None

def _parse_tripinfo(path: str) -> list:
    """Regex-based tripinfo XML scanner (avoids ElementTree DLL issues).
    Same logic as InTAS variant."""
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
    """Apply green times in x to running SUMO via libsumo.  For each TLS,
    update only the phases listed in var_map; auxiliary phases stay at
    their default (which is intended — they were filtered out of the
    decision space because their default is too short to optimize)."""
    from collections import defaultdict
    updates = defaultdict(dict)
    for k, (tls_id, phase_idx) in enumerate(var_map):
        updates[tls_id][phase_idx] = float(x[k])

    for tls_id, phase_updates in updates.items():
        logics = traci.trafficlight.getAllProgramLogics(tls_id)
        if not logics:
            continue
        logic = logics[0]
        new_phases = list(logic.phases)
        for ph_idx, new_dur in phase_updates.items():
            if ph_idx < len(new_phases):
                ph = new_phases[ph_idx]
                try:
                    from libsumo import TraCIPhase
                    new_phases[ph_idx] = TraCIPhase(
                        new_dur, ph.state,
                        ph.minDur if hasattr(ph, 'minDur') else new_dur,
                        ph.maxDur if hasattr(ph, 'maxDur') else new_dur,
                    )
                except Exception:
                    ph.duration = new_dur
        logic.phases = new_phases
        traci.trafficlight.setProgramLogic(tls_id, logic)


def simulate(var_map: list, x: np.ndarray,
             route_file=None,                  # ignored; kept for API parity
             T0: float = 1.0, A0: float = 1.0, E0: float = 1.0,
             seed: Optional[int] = None) -> np.ndarray:
    """
    Run one ingolstadt21 simulation with signal plan x and return [f1, f2, f3].

    Parameters
    ----------
    var_map : list of (tls_id, phase_idx)
    x       : 1-D array of green times
    route_file : ignored — single route file is hard-wired in config
    T0/A0/E0 : baseline values from compute_baseline.py
    seed    : SUMO random seed; drives Krauss driver imperfection
              (the only stochastic source here, since rerouting is off)
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tripinfo_path = os.path.join(tmp_dir, "tripinfo.xml")
        cfg_path = _write_sumocfg(tmp_dir, tripinfo_path)

        sumo_cmd = ["sumo", "-c", cfg_path]
        if seed is not None:
            sumo_cmd += ["--seed", str(int(seed))]

        traci.start(sumo_cmd)
        _apply_signal_plan(var_map, x)
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
        traci.close()

        trips = _parse_tripinfo(tripinfo_path)

    if not trips:
        return np.array([2.0, 2.0, 2.0])

    travel_times = np.array([t['duration']     for t in trips], dtype=float)
    losses       = np.array([t['time_loss']    for t in trips], dtype=float)
    CO2_total_kg = float(sum(t['CO2_mg'] for t in trips)) / 1e6

    # f1: relative average travel time
    avg_tt = float(np.mean(travel_times))
    f1 = avg_tt / T0

    # f2: Atkinson equity index (eps=1) on congestion ratio r = time_loss / free_flow_TT
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
