"""
Parse InTAS ingolstadt.net.xml to extract the 20 real TLS signal programs.

Outputs:
  tls_info: dict  {tls_id -> {'phases': [...], 'green_phase_indices': [...],
                               'default_greens': [...], 'cycle': float}}
  decision_info: dict  {tls_id -> {'phase_indices': [...], 'var_indices': [...]}}
  d: int  total number of decision variables (one green-time per vehicle-green phase)
  bounds: list of (lb, ub) tuples  (GREEN_MIN, GREEN_MAX) for each variable
"""

import xml.etree.ElementTree as ET
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.intas.config import NET_FILE, GREEN_MIN, GREEN_MAX, YELLOW, ALL_RED, RESULTS_DIR
try:
    from experiments.intas.config import (
        MAIN_PHASE_MIN_GREEN, BOUND_LO_FRAC, BOUND_HI_FRAC,
    )
except ImportError:
    # Backwards-compat defaults if the constants aren't defined yet.
    MAIN_PHASE_MIN_GREEN = 0     # no filter -> include all vehicle-green phases
    BOUND_LO_FRAC        = 0.5
    BOUND_HI_FRAC        = 2.0

# The 20 TLS IDs with real programs from Ingolstadt municipality.
# These are identified by having non-trivial multi-phase programs in the net.xml.
# parse_real_tls() identifies them automatically.

def _is_vehicle_green(state: str) -> bool:
    """True if the phase has at least one vehicle-green lane (G/g) and
    no vehicle-red-only lanes (r/o with no G/g)."""
    return 'G' in state or 'g' in state

def _is_yellow_or_clearance(state: str) -> bool:
    return ('y' in state or 'Y' in state) and 'G' not in state

def parse_real_tls(net_file: str = NET_FILE, n_real: int = 20):
    """
    Parse net.xml and return information about the n_real TLS with
    the most-complex signal programs (proxy for 'real' vs SUMO-auto).

    Returns
    -------
    tls_list : list of dicts, sorted by tls_id
        Each dict has keys:
          'id'                : str  TLS ID in SUMO
          'program_id'        : str  program ID (usually '0')
          'phases'            : list of {'duration': float, 'state': str}
          'green_phase_idx'   : list of int  indices of vehicle-green phases
          'default_greens'    : list of float  default green durations
          'cycle'             : float  total cycle length (s)
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    tls_programs = {}
    for tl_logic in root.findall('.//tlLogic'):
        tls_id   = tl_logic.get('id')
        prog_id  = tl_logic.get('programID', '0')
        key = tls_id
        phases   = []
        for ph in tl_logic.findall('phase'):
            phases.append({
                'duration': float(ph.get('duration', 30)),
                'state':    ph.get('state', ''),
            })
        # Keep program with most phases (real programs are more complex)
        if key not in tls_programs or len(phases) > len(tls_programs[key]['phases']):
            tls_programs[key] = {
                'id':         tls_id,
                'program_id': prog_id,
                'phases':     phases,
            }

    # Select the n_real TLS with the most phases (=most complex programs)
    ranked = sorted(tls_programs.values(),
                    key=lambda t: len(t['phases']), reverse=True)
    selected = ranked[:n_real]

    tls_list = []
    for t in sorted(selected, key=lambda t: t['id']):
        green_idx    = [i for i, ph in enumerate(t['phases'])
                        if _is_vehicle_green(ph['state'])
                        and not _is_yellow_or_clearance(ph['state'])]
        default_greens = [t['phases'][i]['duration'] for i in green_idx]
        cycle          = sum(ph['duration'] for ph in t['phases'])
        tls_list.append({
            'id':               t['id'],
            'program_id':       t['program_id'],
            'phases':           t['phases'],
            'green_phase_idx':  green_idx,
            'default_greens':   default_greens,
            'cycle':            cycle,
        })

    return tls_list


def build_decision_space(tls_list):
    """
    Build the flat decision variable vector from parsed TLS list.

    Decision variable x[k] = green time (seconds) for the k-th
    vehicle-green phase across all 20 real TLS, in order.

    Returns
    -------
    var_map : list of (tls_id, phase_idx)  mapping var index -> (TLS, phase)
    bounds  : list of (lb, ub)  per variable
    defaults: np.array  default green times (from real Ingolstadt program)
    """
    import numpy as np
    var_map  = []
    bounds   = []
    defaults = []
    for t in tls_list:
        for local_i, phase_idx in enumerate(t['green_phase_idx']):
            dg = t['default_greens'][local_i]
            # Filter out auxiliary short phases: only optimize main vehicle-green phases.
            # Phases below MAIN_PHASE_MIN_GREEN are kept frozen at their default duration
            # by sumo_sim._apply_signal_plan (which only touches phases listed in var_map).
            if dg < MAIN_PHASE_MIN_GREEN:
                continue
            var_map.append((t['id'], phase_idx))
            lb = max(GREEN_MIN, dg * BOUND_LO_FRAC)
            ub = min(GREEN_MAX, dg * BOUND_HI_FRAC)
            lb = min(lb, ub - 5)             # ensure lb < ub
            bounds.append((float(lb), float(ub)))
            defaults.append(float(dg))
    return var_map, bounds, np.array(defaults)


def save_decision_space(out_dir: str = RESULTS_DIR):
    """Parse net.xml and save decision space to JSON for later use."""
    import numpy as np
    os.makedirs(out_dir, exist_ok=True)
    tls_list = parse_real_tls()
    var_map, bounds, defaults = build_decision_space(tls_list)
    d = len(var_map)
    info = {
        'd':        d,
        'var_map':  var_map,
        'bounds':   bounds,
        'defaults': defaults.tolist(),
        'tls_list': [{k: v for k, v in t.items() if k != 'phases'}
                     for t in tls_list],
    }
    path = os.path.join(out_dir, 'decision_space.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Decision space: d={d}  saved to {path}")
    print(f"  TLS IDs: {[t['id'] for t in tls_list]}")
    for i, (tid, pidx) in enumerate(var_map[:5]):
        print(f"  var[{i}]: TLS={tid} phase={pidx}  "
              f"default={defaults[i]:.1f}s  "
              f"bounds=[{bounds[i][0]:.1f},{bounds[i][1]:.1f}]")
    if d > 5:
        print(f"  ... ({d-5} more)")
    return info


if __name__ == "__main__":
    save_decision_space()
