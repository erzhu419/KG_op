"""Batch runner for all experiments.

Runs GPR-KG and GPR-KG-nV on RZDT1/2/5, 10 reps each.
Matches Draft_1017-Bao.docx Section 5.1 experimental design.
"""
import os
import sys
import json
import time
import traceback
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import run_gpr_kg


# Experiment configuration (matching old paper Section 5.1)
CONFIG = {
    'problems': ['RZDT1', 'RZDT2', 'RZDT5'],
    'methods': [True, False],  # True=with VEPM, False=without VEPM
    'n_reps': 10,
    'seed_base': 1000,
    'N': 100,
    'd': 5,
    'N0': 50,
    'K1': 20,
    'K2': 2,
    'm': 40,
    'n_thr': 20,
    'output_dir': 'results',
}


def run_single(args):
    """Run a single experiment (for multiprocessing)."""
    problem_name, seed, use_vepm, output_dir = args
    method_name = 'GPR_KG' if use_vepm else 'GPR_KG_nV'
    fname = f"{problem_name}_{method_name}_seed{seed}.json"
    fpath = os.path.join(output_dir, fname)

    # Skip if already done
    if os.path.exists(fpath):
        print(f"  Skipping {fname} (already exists)")
        return fname, True

    try:
        result = run_gpr_kg(
            problem_name=problem_name,
            seed=seed,
            N=CONFIG['N'],
            d=CONFIG['d'],
            N0=CONFIG['N0'],
            K1=CONFIG['K1'],
            K2=CONFIG['K2'],
            use_vepm=use_vepm,
            verbose=False,
        )

        os.makedirs(output_dir, exist_ok=True)
        with open(fpath, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"  Done: {fname} | HV={result['hv_final']:.4f} "
              f"IGD={result['igd_final']:.4f} Time={result['time_total']:.1f}s")
        return fname, True
    except Exception as e:
        print(f"  FAILED: {fname} | {e}")
        traceback.print_exc()
        return fname, False


def run_all(n_workers=None):
    """Run all experiments."""
    output_dir = CONFIG['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Build task list
    tasks = []
    for problem in CONFIG['problems']:
        for use_vepm in CONFIG['methods']:
            for rep in range(CONFIG['n_reps']):
                seed = CONFIG['seed_base'] + rep
                tasks.append((problem, seed, use_vepm, output_dir))

    total = len(tasks)
    print(f"\nTotal experiments: {total}")
    print(f"  Problems: {CONFIG['problems']}")
    print(f"  Methods: GPR-KG (VEPM), GPR-KG-nV (no VEPM)")
    print(f"  Reps: {CONFIG['n_reps']} (seeds {CONFIG['seed_base']}-{CONFIG['seed_base']+CONFIG['n_reps']-1})")
    print(f"  Parameters: N={CONFIG['N']}, N0={CONFIG['N0']}, d={CONFIG['d']}")
    print()

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    t_start = time.time()

    if n_workers > 1:
        print(f"Running with {n_workers} workers...")
        with Pool(n_workers) as pool:
            results = pool.map(run_single, tasks)
    else:
        print("Running sequentially...")
        results = [run_single(task) for task in tasks]

    t_total = time.time() - t_start
    n_success = sum(1 for _, ok in results if ok)
    n_failed = total - n_success
    print(f"\n{'='*60}")
    print(f"Completed: {n_success}/{total} ({n_failed} failed)")
    print(f"Total time: {t_total:.1f}s")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--problem', default=None,
                        help='Run only this problem (RZDT1/RZDT2/RZDT5)')
    parser.add_argument('--reps', type=int, default=None,
                        help='Override number of reps')
    parser.add_argument('--force', action='store_true',
                        help='Remove existing results and re-run')
    args = parser.parse_args()

    if args.problem:
        CONFIG['problems'] = [args.problem]
    if args.reps:
        CONFIG['n_reps'] = args.reps
    if args.force:
        # Remove existing results to force re-run
        import glob
        for problem in CONFIG['problems']:
            for method_key in ['GPR_KG', 'GPR_KG_nV']:
                pattern = os.path.join(CONFIG['output_dir'],
                                       f'{problem}_{method_key}_seed*.json')
                for f in glob.glob(pattern):
                    os.remove(f)
                    print(f"  Removed: {os.path.basename(f)}")

    run_all(n_workers=args.workers)
