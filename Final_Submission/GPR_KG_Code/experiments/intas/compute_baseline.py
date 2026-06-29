"""
Step 0: Compute baseline values T0, A0, E0 for the real Ingolstadt signal plan.

Run this ONCE before any optimisation experiments:
    python -m experiments.intas.compute_baseline

Outputs  results/intas/baseline.json  with T0, A0, E0 and their standard errors.
"""

import os
import sys
import json
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.intas.config import ROUTE_DIR, RESULTS_DIR, N_BASELINE
from experiments.intas.parse_network import parse_real_tls, build_decision_space
from experiments.intas.sumo_sim import simulate


def compute_and_save(n_reps: int = None, verbose: bool = True):
    if n_reps is None:
        # Allow override via env var for faster iteration
        n_reps = int(os.environ.get("INTAS_BASELINE_REPS", N_BASELINE))
    """
    Run n_reps simulations with the real Ingolstadt signal plan and save
    the resulting T0, A0, E0 baselines.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    bl_path = os.path.join(RESULTS_DIR, 'baseline.json')
    if os.path.exists(bl_path):
        print(f"Baseline already exists at {bl_path}. Delete to recompute.")
        return json.load(open(bl_path))

    # Load decision space
    ds_path = os.path.join(RESULTS_DIR, 'decision_space.json')
    if not os.path.exists(ds_path):
        print("decision_space.json not found — running parse_network first...")
        from experiments.intas.parse_network import save_decision_space
        save_decision_space()

    import json as _json
    with open(ds_path) as f:
        ds = _json.load(f)
    var_map   = [tuple(v) for v in ds['var_map']]
    default_x = np.array(ds['defaults'])

    # Route files for demand scenarios
    route_files = sorted(glob.glob(os.path.join(ROUTE_DIR, 'InTAS_*.rou.xml')))
    if not route_files:
        raise FileNotFoundError(f"No route files in {ROUTE_DIR}")

    # ── Resume from partial file if it exists ─────────────────────────────
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
                      f"Writing final baseline.json from existing data.",
                      flush=True)
            else:
                print(f"Resuming from partial: {start_rep}/{n_reps} done, "
                      f"continuing with rep {start_rep+1}.", flush=True)
        except Exception as e:
            print(f"Could not read partial ({e}); starting fresh.", flush=True)
            raw_T, raw_A, raw_E = [], [], []
            start_rep = 0

    rng = np.random.default_rng(0)

    print(f"\nComputing baseline with {n_reps} replications "
          f"(real Ingolstadt signal plan)...", flush=True)
    import time as _time
    t_start = _time.time()
    # Skip RNG state to match previous runs' seeds (deterministic)
    for _ in range(start_rep):
        rng.integers(0, 100000)
    for rep in range(start_rep, n_reps):
        rf   = route_files[rep % len(route_files)]
        seed = int(rng.integers(0, 100000))
        t_rep = _time.time()
        y    = simulate(var_map, default_x, rf,
                        T0=1.0, A0=1.0, E0=1.0, seed=seed)
        dt_rep = _time.time() - t_rep
        # y = [avg_tt, atkinson, CO2_kg]  (unnormalised)
        raw_T.append(float(y[0]))
        raw_A.append(float(y[1]))
        raw_E.append(float(y[2]))
        if verbose:
            done_now = rep + 1 - start_rep
            eta_min = (n_reps - rep - 1) * dt_rep / 60
            print(f"  rep {rep+1:02d}/{n_reps} ({done_now} this session):  "
                  f"avg_tt={y[0]:.1f}s  atkinson={y[1]:.4f}  "
                  f"CO2={y[2]:.1f}kg  "
                  f"[route: {os.path.basename(rf)}]  "
                  f"dt={dt_rep:.0f}s  ETA={eta_min:.0f}min",
                  flush=True)

        # ── Incremental save for progress visibility ─────────────────────
        partial = {
            'status':  'running',
            'n_done':  rep + 1,
            'n_total': n_reps,
            'raw_T':   raw_T, 'raw_A': raw_A, 'raw_E': raw_E,
            'elapsed_sec': float(_time.time() - t_start),
        }
        with open(bl_path + '.partial', 'w') as f:
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
        'signal_plan': 'real_Ingolstadt',
        'sim_window_s': [7*3600, 9*3600],
        'raw_T': raw_T, 'raw_A': raw_A, 'raw_E': raw_E,
    }
    with open(bl_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Baseline results ({n_reps} reps):")
    print(f"  T0 = {T0:.2f} ± {T0_se:.2f} s  (avg travel time)")
    print(f"  A0 = {A0:.4f} ± {A0_se:.4f}    (Atkinson equity index)")
    print(f"  E0 = {E0:.2f} ± {E0_se:.2f} kg (total CO2, morning peak)")
    print(f"Saved to {bl_path}")
    return result


if __name__ == "__main__":
    compute_and_save()
