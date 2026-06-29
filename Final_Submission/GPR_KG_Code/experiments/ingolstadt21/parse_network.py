"""
Parse RESCO ingolstadt21.net.xml to extract the 21 fixed-time TLS programs.

Outputs (decision_space.json, schema identical to the InTAS variant):
  d         : int                       total decision dimensions
  var_map   : list of (tls_id, phase_idx)  one entry per optimized phase
  bounds    : list of (lb, ub)
  defaults  : list of float             default green durations
  tls_list  : list of dicts             20 selected TLS metadata
"""

import os
import sys
import json
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ingolstadt21.config import (
    NET_FILE, GREEN_MIN, GREEN_MAX, RESULTS_DIR,
    MAIN_PHASE_MIN_GREEN, BOUND_LO_FRAC, BOUND_HI_FRAC,
)


def _is_vehicle_green(state: str) -> bool:
    """At least one vehicle-green lane (G/g) and no yellow signal."""
    return ('G' in state or 'g' in state) and ('y' not in state and 'Y' not in state)


def parse_real_tls(net_file: str = NET_FILE):
    """Parse all <tlLogic> programs from ingolstadt21.net.xml.

    Returns
    -------
    tls_list : list of dicts, one per TLS, sorted by tls_id
    """
    tree = ET.parse(net_file)
    root = tree.getroot()

    tls_programs = {}
    for tl_logic in root.findall('.//tlLogic'):
        tls_id  = tl_logic.get('id')
        prog_id = tl_logic.get('programID', '0')
        phases  = []
        for ph in tl_logic.findall('phase'):
            phases.append({
                'duration': float(ph.get('duration', 30)),
                'state':    ph.get('state', ''),
            })
        # Keep program with most phases (consistent with InTAS rule)
        if tls_id not in tls_programs or len(phases) > len(tls_programs[tls_id]['phases']):
            tls_programs[tls_id] = {
                'id':         tls_id,
                'program_id': prog_id,
                'phases':     phases,
            }

    tls_list = []
    for t in sorted(tls_programs.values(), key=lambda t: t['id']):
        green_idx = [i for i, ph in enumerate(t['phases'])
                     if _is_vehicle_green(ph['state'])]
        default_greens = [t['phases'][i]['duration'] for i in green_idx]
        cycle = sum(ph['duration'] for ph in t['phases'])
        tls_list.append({
            'id':              t['id'],
            'program_id':      t['program_id'],
            'phases':          t['phases'],
            'green_phase_idx': green_idx,
            'default_greens':  default_greens,
            'cycle':           cycle,
        })
    return tls_list


def build_decision_space(tls_list):
    """Build the flat decision variable vector: (tls_id, phase_idx) plus bounds.

    Auxiliary phases with default < MAIN_PHASE_MIN_GREEN are filtered out and
    kept frozen at their default in sumo_sim._apply_signal_plan (which only
    touches phases listed in var_map).
    """
    import numpy as np
    var_map  = []
    bounds   = []
    defaults = []
    for t in tls_list:
        for local_i, phase_idx in enumerate(t['green_phase_idx']):
            dg = t['default_greens'][local_i]
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
    print(f"  TLS IDs ({len(tls_list)}): {[t['id'][:24] for t in tls_list]}")
    n_per_tls = {}
    for tid, _ in var_map:
        n_per_tls[tid] = n_per_tls.get(tid, 0) + 1
    print(f"  phases-per-TLS distribution: {sorted(n_per_tls.values())}")
    print(f"  default greens: min={defaults.min():.0f}s  median={float(np.median(defaults)):.0f}s  max={defaults.max():.0f}s")
    print(f"  bound spans:    min={(np.array([b[1]-b[0] for b in bounds])).min():.1f}s  "
          f"max={(np.array([b[1]-b[0] for b in bounds])).max():.1f}s")
    for i, (tid, pidx) in enumerate(var_map[:5]):
        print(f"  var[{i}]: TLS={tid[:30]} phase={pidx}  "
              f"default={defaults[i]:.1f}s  "
              f"bounds=[{bounds[i][0]:.1f},{bounds[i][1]:.1f}]")
    if d > 5:
        print(f"  ... ({d-5} more)")
    return info


if __name__ == "__main__":
    save_decision_space()
