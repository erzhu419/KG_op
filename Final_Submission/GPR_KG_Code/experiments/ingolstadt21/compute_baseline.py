"""
Phase 0: compute baseline T0/A0/E0 for the ingolstadt21 fixed-time plan.

Run ONCE before optimization experiments:
    python -m experiments.ingolstadt21.compute_baseline
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.ingolstadt21.config import RESULTS_DIR, ROUTE_FILE, N_BASELINE
from experiments.ingolstadt21.sumo_sim import simulate


def compute_and_save(n_reps: int = None, verbose: bool = True):
    if n_reps is None:
        n_reps = int(os.environ.get("INGOLSTADT21_BASELINE_REPS", N_BASELINE))

    os.makedirs(RESULTS_DIR, exist_ok=True)
    bl_path = os.path.join(RESULTS_DIR, 'baseline.json')
    if os.path.exists(bl_path):
        print(f"Baseline already exists at {bl_path}. Delete to recompute.")
        return json.load(open(bl_path))

    ds_path = os.path.join(RESULTS_DIR, 'decision_space.json')
    if not os.path.exists(ds_path):
        print("decision_space.json not found - running parse_network first...")
        from experiments.ingolstadt21.parse_network import save_decision_space
        save_decision_space()

    with open(ds_path) as f:
        ds = json.load(f)
    var_map   = [tuple(v) for v in ds['var_map']]
    default_x = np.array(ds['defaults'])

    raw_T, raw_A, raw_E = [], [], []
    start_rep = 0
    partial_path = bl_path + '.partial'
    if os.path.exists(partial_path):
        try:
            with open(partial_path) as f:
                prev = json.load(f)
            raw_T = list(prev.get('raw_T', []))
            raw_A = list(prev.get('raw_A', []))
            raw_E = list(prev.get('raw_E', []))
            start_rep = len(raw_T)
            if start_rep >= n_reps:
                print(f"Partial already has {start_rep} >= {n_reps} reps. "
                      f"Writing final baseline.json.", flush=True)
            else:
                print(f"Resuming from partial: {start_rep}/{n_reps} done.",
                      flush=True)
        except Exception as e:
            print(f"Could not read partial ({e}); starting fresh.", flush=True)
            raw_T, raw_A, raw_E = [], [], []
            start_rep = 0

    rng = np.random.default_rng(0)

    print(f"\nComputing ingolstadt21 baseline with {n_reps} replications...",
          flush=True)
    t_start = time.time()
    for _ in range(start_rep):
        rng.integers(0, 100000)
    for rep in range(start_rep, n_reps):
        seed = int(rng.integers(0, 100000))
        t_rep = time.time()
        y = simulate(var_map, default_x, route_file=ROUTE_FILE,
                     T0=1.0, A0=1.0, E0=1.0, seed=seed)
        dt_rep = time.time() - t_rep
        raw_T.append(float(y[0]))
        raw_A.append(float(y[1]))
        raw_E.append(float(y[2]))
        if verbose:
            done = rep + 1 - start_rep
            eta = (n_reps - rep - 1) * dt_rep / 60
            print(f"  rep {rep+1:02d}/{n_reps} (#{done} this session): "
                  f"avg_tt={y[0]:.1f}s  atkinson={y[1]:.4f}  "
                  f"CO2={y[2]:.1f}kg  dt={dt_rep:.0f}s  ETA={eta:.0f}min",
                  flush=True)

        partial = {
            'status':  'running',
            'n_done':  rep + 1,
            'n_total': n_reps,
            'raw_T':   raw_T, 'raw_A': raw_A, 'raw_E': raw_E,
            'elapsed_sec': float(time.time() - t_start),
        }
        with open(partial_path, 'w') as f:
            json.dump(partial, f, indent=2)

    T0    = float(np.mean(raw_T))
    A0    = float(np.mean(raw_A))
    E0    = float(np.mean(raw_E))
    T0_se = float(np.std(raw_T, ddof=1) / np.sqrt(n_reps))
    A0_se = float(np.std(raw_A, ddof=1) / np.sqrt(n_reps))
    E0_se = float(np.std(raw_E, ddof=1) / np.sqrt(n_reps))

    result = {
        'T0': T0, 'T0_se': T0_se, 'T0_unit': 'seconds',
        'A0': A0, 'A0_se': A0_se, 'A0_unit': 'Atkinson_index',
        'E0': E0, 'E0_se': E0_se, 'E0_unit': 'kg_CO2',
        'n_reps': n_reps,
        'signal_plan': 'fixed_time_RESCO_ingolstadt21',
        'sim_window_s': [57600, 61200],
        'raw_T': raw_T, 'raw_A': raw_A, 'raw_E': raw_E,
    }
    with open(bl_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"ingolstadt21 baseline ({n_reps} reps):")
    print(f"  T0 = {T0:.2f} +/- {T0_se:.2f} s    (avg travel time)")
    print(f"  A0 = {A0:.4f} +/- {A0_se:.4f}      (Atkinson equity index)")
    print(f"  E0 = {E0:.2f} +/- {E0_se:.2f} kg   (total CO2, PM peak)")
    print(f"Saved to {bl_path}")
    return result


if __name__ == "__main__":
    compute_and_save()
