"""
Master experiment runner for Section 6 of the paper.

Runs all experiments organized by subsection with:
- Incremental result saving (skip completed runs on resume)
- Per-replication JSON files with full intermediate data
- Summary statistics computation
- Progress reporting

Usage:
    python -m experiments.run_all                    # Run all sections
    python -m experiments.run_all --section 6.2      # Run specific section
    python -m experiments.run_all --section 6.2 --method GPR-KG  # Specific method
"""

import sys
import os
import json
import time
import argparse
import numpy as np

# Add parent directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT1, RZDT2, RZDT5, RZDT3, RZDT4, RZDT6, compute_hypervolume_2d
from metrics import compute_igd, compute_cvr, wilcoxon_test
from experiments.config import *

from methods.random_search import RandomSearch
from methods.gpr_kg_method import GPRKGMethod
from methods.gpr_kg_nv import GPRKGnVMethod
from methods.cehvi_method import cEHVIMethod
from methods.cparego_method import cParEGOMethod
from methods.nsga2_kriging import NSGA2Kriging
from methods.nsga2_direct import NSGA2Direct


# ============================================================
# Method registry
# ============================================================
ALL_METHODS = {
    'GPR-KG':    lambda: GPRKGMethod(),
    'GPR-KG-nV': lambda: GPRKGnVMethod(),
    'cEHVI':     lambda: cEHVIMethod(),
    'cParEGO':   lambda: cParEGOMethod(),
    'NSGA-II-K': lambda: NSGA2Kriging(),
    'NSGA-II-D': lambda: NSGA2Direct(),
    'RS':        lambda: RandomSearch(),
}

PROBLEM_CLASSES = {
    'RZDT1': RZDT1,
    'RZDT2': RZDT2,
    'RZDT5': RZDT5,
    # Heteroscedastic benchmark suite (Section 6.5 extended)
    'RZDT3': RZDT3,  # 5-band discontinuous PF, oscillating noise (3.5x)
    'RZDT4': RZDT4,  # multi-modal landscape, exponential noise (6x)
    'RZDT6': RZDT6,  # non-uniform density, gradient noise (8.3x)
}


def result_path(section_dir, method_name, problem_name, rep, suffix=""):
    """Generate result file path."""
    safe_name = method_name.replace('-', '_')
    fname = f"{problem_name}_{safe_name}_rep{rep+1}{suffix}.json"
    return os.path.join(section_dir, fname)


def run_single(method_name, problem_cls, problem_kwargs, N, n0, seed,
               save_path, feasibility_ratio=0.5):
    """Run a single (method, problem, rep) combination and save results."""
    if os.path.exists(save_path):
        print(f"    [SKIP] {save_path} already exists")
        return json.load(open(save_path, 'r'))

    prob = problem_cls(**problem_kwargs)
    prob.calibrate_constraint(feasibility_ratio)

    method = ALL_METHODS[method_name]()
    t0 = time.time()
    result = method.run(prob, N=N, n0=n0, seed=seed)
    result['total_wall_time'] = time.time() - t0
    result['seed'] = seed
    result['problem_kwargs'] = {k: v for k, v in problem_kwargs.items()}
    result['feasibility_ratio'] = feasibility_ratio

    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


def aggregate_results(section_dir, method_name, problem_name, n_reps, suffix=""):
    """Load and aggregate results across replications."""
    hvs, igds, cvrs, times, time_per_iters = [], [], [], [], []

    for rep in range(n_reps):
        path = result_path(section_dir, method_name, problem_name, rep, suffix)
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            r = json.load(f)
        hvs.append(r['hv_final'])
        igds.append(r['igd_final'])
        cvrs.append(r['cvr_final'])
        times.append(r.get('total_wall_time', r.get('total_time_sec', 0)))
        time_per_iters.append(r.get('time_per_iter_mean', 0))

    if len(hvs) == 0:
        return None

    return {
        'method': method_name,
        'problem': problem_name,
        'n_reps': len(hvs),
        'hv_mean': float(np.mean(hvs)),
        'hv_se': float(np.std(hvs) / np.sqrt(len(hvs))),
        'hv_std': float(np.std(hvs)),
        'hv_values': hvs,
        'igd_mean': float(np.mean(igds)),
        'igd_se': float(np.std(igds) / np.sqrt(len(igds))),
        'igd_values': igds,
        'cvr_mean': float(np.mean(cvrs)),
        'cvr_se': float(np.std(cvrs) / np.sqrt(len(cvrs))),
        'cvr_values': cvrs,
        'time_mean': float(np.mean(times)),
        'time_per_iter_mean': float(np.mean(time_per_iters)),
    }


# ============================================================
# Section 6.2: Standard Benchmarks
# ============================================================
def run_section_62(methods=None, n_reps=N_REPS):
    """7 methods x 3 problems x n_reps. Table 1 + Figure 1."""
    print("\n" + "=" * 70)
    print("  SECTION 6.2: Standard Benchmarks")
    print("=" * 70)

    section_dir = os.path.join(BASE_DIR, "results", "sec62")
    os.makedirs(section_dir, exist_ok=True)

    if methods is None:
        methods = list(ALL_METHODS.keys())

    problems = ['RZDT1', 'RZDT2', 'RZDT5']
    problem_kwargs = {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': DEFAULT_SIGMA}

    for problem_name in problems:
        prob_cls = PROBLEM_CLASSES[problem_name]
        for method_name in methods:
            print(f"\n  {method_name} on {problem_name}:")
            for rep in range(n_reps):
                seed = SEED_BASE + rep
                save_path = result_path(section_dir, method_name, problem_name, rep)
                print(f"    Rep {rep+1}/{n_reps}...", end=" ", flush=True)
                t0 = time.time()
                result = run_single(method_name, prob_cls, problem_kwargs,
                                    DEFAULT_N, DEFAULT_N0, seed, save_path,
                                    FEASIBILITY_MODERATE)
                print(f"HV={result['hv_final']:.4f}, "
                      f"IGD={result['igd_final']:.4f}, "
                      f"CVR={result['cvr_final']:.4f}, "
                      f"{time.time()-t0:.1f}s")

    # Aggregate and save summary
    summary = {}
    for problem_name in problems:
        summary[problem_name] = {}
        for method_name in methods:
            agg = aggregate_results(section_dir, method_name, problem_name, n_reps)
            if agg:
                summary[problem_name][method_name] = agg

    with open(os.path.join(section_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print_table1(summary)
    return summary


# ============================================================
# Section 6.3: Scalability
# ============================================================
def run_section_63(methods=None, n_reps=N_REPS):
    """7 methods x 3 dimensions x n_reps on RZDT1. Table 2 + Figure 2."""
    print("\n" + "=" * 70)
    print("  SECTION 6.3: Scalability Study")
    print("=" * 70)

    section_dir = os.path.join(BASE_DIR, "results", "sec63")
    os.makedirs(section_dir, exist_ok=True)

    if methods is None:
        methods = list(ALL_METHODS.keys())

    for d in SCALABILITY_DIMS:
        problem_kwargs = {'d': d, 'L': DEFAULT_L, 'sigma': DEFAULT_SIGMA}
        problem_name = f"RZDT1_d{d}"

        for method_name in methods:
            print(f"\n  {method_name} on RZDT1 (d={d}):")
            for rep in range(n_reps):
                seed = SEED_BASE + rep
                save_path = result_path(section_dir, method_name, problem_name, rep)
                print(f"    Rep {rep+1}/{n_reps}...", end=" ", flush=True)
                t0 = time.time()
                result = run_single(method_name, RZDT1, problem_kwargs,
                                    DEFAULT_N, DEFAULT_N0, seed, save_path,
                                    FEASIBILITY_MODERATE)
                elapsed = time.time() - t0
                print(f"HV={result['hv_final']:.4f}, {elapsed:.1f}s")

    # Aggregate
    summary = {}
    for d in SCALABILITY_DIMS:
        problem_name = f"RZDT1_d{d}"
        summary[problem_name] = {}
        for method_name in methods:
            agg = aggregate_results(section_dir, method_name, problem_name, n_reps)
            if agg:
                summary[problem_name][method_name] = agg

    with open(os.path.join(section_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Section 6.4: Constraint Tightness
# ============================================================
def run_section_64(methods=None, n_reps=N_REPS):
    """7 methods x 3 constraint levels x n_reps on RZDT1. Table 3."""
    print("\n" + "=" * 70)
    print("  SECTION 6.4: Constraint Tightness Study")
    print("=" * 70)

    section_dir = os.path.join(BASE_DIR, "results", "sec64")
    os.makedirs(section_dir, exist_ok=True)

    if methods is None:
        methods = list(ALL_METHODS.keys())

    feas_levels = {
        'loose': FEASIBILITY_LOOSE,
        'moderate': FEASIBILITY_MODERATE,
        'tight': FEASIBILITY_TIGHT,
    }
    problem_kwargs = {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': DEFAULT_SIGMA}

    for level_name, feas_ratio in feas_levels.items():
        problem_name = f"RZDT1_{level_name}"
        for method_name in methods:
            print(f"\n  {method_name} on RZDT1 ({level_name} constraint):")
            for rep in range(n_reps):
                seed = SEED_BASE + rep
                save_path = result_path(section_dir, method_name, problem_name, rep)
                print(f"    Rep {rep+1}/{n_reps}...", end=" ", flush=True)
                t0 = time.time()
                result = run_single(method_name, RZDT1, problem_kwargs,
                                    DEFAULT_N, DEFAULT_N0, seed, save_path,
                                    feas_ratio)
                elapsed = time.time() - t0
                print(f"HV={result['hv_final']:.4f}, CVR={result['cvr_final']:.4f}, {elapsed:.1f}s")

    # Aggregate
    summary = {}
    for level_name in feas_levels:
        problem_name = f"RZDT1_{level_name}"
        summary[problem_name] = {}
        for method_name in methods:
            agg = aggregate_results(section_dir, method_name, problem_name, n_reps)
            if agg:
                summary[problem_name][method_name] = agg

    with open(os.path.join(section_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Section 6.5: Noise Sensitivity
# ============================================================
def run_section_65(methods=None, n_reps=N_REPS):
    """7 methods x 4 noise levels x n_reps on RZDT1. Table 4."""
    print("\n" + "=" * 70)
    print("  SECTION 6.5: Noise Sensitivity Study")
    print("=" * 70)

    section_dir = os.path.join(BASE_DIR, "results", "sec65")
    os.makedirs(section_dir, exist_ok=True)

    if methods is None:
        methods = list(ALL_METHODS.keys())

    noise_configs = [
        ('sigma001', {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': 0.01, 'heteroscedastic': False}),
        ('sigma01',  {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': 0.1,  'heteroscedastic': False}),
        ('sigma10',  {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': 1.0,  'heteroscedastic': False}),
        ('hetero',   {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': 0.1,  'heteroscedastic': True}),
    ]

    for noise_name, problem_kwargs in noise_configs:
        problem_name = f"RZDT1_{noise_name}"
        for method_name in methods:
            print(f"\n  {method_name} on RZDT1 ({noise_name}):")
            for rep in range(n_reps):
                seed = SEED_BASE + rep
                save_path = result_path(section_dir, method_name, problem_name, rep)
                print(f"    Rep {rep+1}/{n_reps}...", end=" ", flush=True)
                t0 = time.time()
                result = run_single(method_name, RZDT1, problem_kwargs,
                                    DEFAULT_N, DEFAULT_N0, seed, save_path,
                                    FEASIBILITY_MODERATE)
                elapsed = time.time() - t0
                print(f"HV={result['hv_final']:.4f}, CVR={result['cvr_final']:.4f}, {elapsed:.1f}s")

    # Aggregate
    summary = {}
    for noise_name, _ in noise_configs:
        problem_name = f"RZDT1_{noise_name}"
        summary[problem_name] = {}
        for method_name in methods:
            agg = aggregate_results(section_dir, method_name, problem_name, n_reps)
            if agg:
                summary[problem_name][method_name] = agg

    with open(os.path.join(section_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Section 6.6: Ablation Study
# ============================================================
def run_section_66(n_reps=N_REPS):
    """Ablation variants x n_reps on RZDT1. Table 5 + Figure 3."""
    print("\n" + "=" * 70)
    print("  SECTION 6.6: Ablation Study")
    print("=" * 70)

    section_dir = os.path.join(BASE_DIR, "results", "sec66")
    os.makedirs(section_dir, exist_ok=True)

    problem_kwargs = {'d': DEFAULT_D, 'L': DEFAULT_L, 'sigma': DEFAULT_SIGMA}

    # Ablation configurations
    ablation_configs = [
        ('default', 'GPR-KG', {}),
        ('vepm_off', 'GPR-KG-nV', {}),
        # Candidate set variants
        ('cand_30', 'GPR-KG', {'K1': 10, 'K2': 1}),
        ('cand_150', 'GPR-KG', {'K1': 50, 'K2': 5}),
    ]

    for config_name, base_method, override_params in ablation_configs:
        print(f"\n  Ablation: {config_name}")
        for rep in range(n_reps):
            seed = SEED_BASE + rep
            save_path = result_path(section_dir, config_name, "RZDT1", rep)
            print(f"    Rep {rep+1}/{n_reps}...", end=" ", flush=True)
            t0 = time.time()

            if os.path.exists(save_path):
                print(f"[SKIP]")
                continue

            prob = RZDT1(**problem_kwargs)
            prob.calibrate_constraint(FEASIBILITY_MODERATE)

            if base_method == 'GPR-KG':
                method = GPRKGMethod(**override_params)
            elif base_method == 'GPR-KG-nV':
                method = GPRKGnVMethod(**override_params)

            result = method.run(prob, N=DEFAULT_N, n0=DEFAULT_N0, seed=seed)
            result['ablation_config'] = config_name
            result['override_params'] = override_params

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            elapsed = time.time() - t0
            print(f"HV={result['hv_final']:.4f}, {elapsed:.1f}s")

    # Aggregate
    summary = {}
    for config_name, _, _ in ablation_configs:
        agg = aggregate_results(section_dir, config_name, "RZDT1", n_reps)
        if agg:
            summary[config_name] = agg

    with open(os.path.join(section_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================
# Utilities
# ============================================================
def print_table1(summary):
    """Print Table 1 format."""
    print(f"\n{'='*90}")
    print("  TABLE 1: Standard Benchmark Results")
    print(f"{'='*90}")
    print(f"{'Method':15s}", end="")
    for p in ['RZDT1', 'RZDT2', 'RZDT5']:
        print(f" | {'HV':>8s} {'IGD':>8s} {'CVR':>8s}", end="")
    print()
    print("-" * 90)

    for method in METHOD_NAMES:
        print(f"{method:15s}", end="")
        for p in ['RZDT1', 'RZDT2', 'RZDT5']:
            if p in summary and method in summary[p]:
                s = summary[p][method]
                print(f" | {s['hv_mean']:>8.4f} {s['igd_mean']:>8.4f} {s['cvr_mean']:>8.4f}", end="")
            else:
                print(f" | {'---':>8s} {'---':>8s} {'---':>8s}", end="")
        print()


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Section 6 experiments")
    parser.add_argument('--section', type=str, default='all',
                        help='Section to run: 6.2, 6.3, 6.4, 6.5, 6.6, or all')
    parser.add_argument('--method', type=str, default=None,
                        help='Specific method to run (e.g., GPR-KG)')
    parser.add_argument('--n_reps', type=int, default=N_REPS,
                        help=f'Number of replications (default: {N_REPS})')
    args = parser.parse_args()

    methods = [args.method] if args.method else None

    sections = {
        '6.2': lambda: run_section_62(methods, args.n_reps),
        '6.3': lambda: run_section_63(methods, args.n_reps),
        '6.4': lambda: run_section_64(methods, args.n_reps),
        '6.5': lambda: run_section_65(methods, args.n_reps),
        '6.6': lambda: run_section_66(args.n_reps),
    }

    if args.section == 'all':
        for sec_name, sec_func in sections.items():
            sec_func()
    elif args.section in sections:
        sections[args.section]()
    else:
        print(f"Unknown section: {args.section}")
        print(f"Available: {', '.join(sections.keys())}, all")
