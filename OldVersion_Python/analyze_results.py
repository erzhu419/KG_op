"""Analyze experiment results and generate comparison tables.

Reproduces the analysis from Draft_1017-Bao.docx Table 1.
"""
import os
import json
import glob
import numpy as np


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
PROBLEMS = ['RZDT1', 'RZDT2', 'RZDT5']
METHODS = {'GPR_KG': 'With VEPM', 'GPR_KG_nV': 'Without VEPM'}
METRICS = ['hv_final', 'igd_final', 'n_lpos', 'n_infeasible', 'n_pareto', 'n_fes', 'time_total']
METRIC_LABELS = {
    'hv_final': 'HV',
    'igd_final': 'IGD',
    'n_lpos': '#LPOS',
    'n_infeasible': '#Infeas',
    'n_pareto': '#Pareto',
    'n_fes': '#FEs',
    'time_total': 'Time (s)',
}


def load_results():
    """Load all result JSON files."""
    results = {}
    for method_key in METHODS:
        for problem in PROBLEMS:
            pattern = os.path.join(RESULTS_DIR, f'{problem}_{method_key}_seed*.json')
            files = sorted(glob.glob(pattern))
            key = (problem, method_key)
            results[key] = []
            for f in files:
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                    results[key].append(data)
                except (json.JSONDecodeError, IOError):
                    pass
    return results


def print_table(results):
    """Print results table matching Draft_1017-Bao.docx Table 1 format."""
    print("\n" + "=" * 90)
    print("Reproduction of Old Paper Table 1: GPR-KG with VEPM vs Without VEPM")
    print("=" * 90)

    for problem in PROBLEMS:
        print(f"\n--- {problem} ---")
        header = f"{'Method':<20}"
        for metric in METRICS:
            header += f" {METRIC_LABELS[metric]:>12}"
        print(header)
        print("-" * len(header))

        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            data_list = results.get(key, [])
            if not data_list:
                print(f"{method_label:<20}  (no data)")
                continue

            row = f"{method_label:<20}"
            for metric in METRICS:
                vals = [d[metric] for d in data_list if metric in d]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    row += f" {mean:>8.4f}({std:.3f})"
                else:
                    row += f" {'---':>12}"
            row += f"  [n={len(data_list)}]"
            print(row)

    # Target values from old paper (Table 1: only with-VEPM results reported)
    print("\n--- Target values from Draft_1017-Bao.docx Table 1 ---")
    print(f"{'Problem':<10} {'#FEs':>6} {'#TPOS':>6} {'Avg #LPOS':>10} {'Avg IGD':>10}")
    print(f"{'RZDT1':<10} {'111':>6} {'32':>6} {'18.5':>10} {'0.055':>10}")
    print(f"{'RZDT2':<10} {'111':>6} {'32':>6} {'16.3':>10} {'0.103':>10}")
    print(f"{'RZDT5':<10} {'111':>6} {'12':>6} {'11.1':>10} {'0.005':>10}")

    # Direct comparison: our GPR-KG (with VEPM) vs paper targets
    print("\n--- Comparison: Our GPR-KG vs Paper Targets ---")
    targets = {
        'RZDT1': {'igd': 0.055, 'lpos': 18.5},
        'RZDT2': {'igd': 0.103, 'lpos': 16.3},
        'RZDT5': {'igd': 0.005, 'lpos': 11.1},
    }
    print(f"{'Problem':<10} {'Our IGD':>10} {'Paper IGD':>10} {'Our #LPOS':>10} {'Paper #LPOS':>12}")
    for problem in PROBLEMS:
        key = (problem, 'GPR_KG')
        data_list = results.get(key, [])
        if data_list:
            our_igd = np.mean([d['igd_final'] for d in data_list])
            our_lpos = np.mean([d['n_lpos'] for d in data_list])
            t = targets[problem]
            print(f"{problem:<10} {our_igd:>10.4f} {t['igd']:>10.4f} {our_lpos:>10.1f} {t['lpos']:>12.1f}")
        else:
            print(f"{problem:<10}  (no data)")


def print_latex_table(results):
    """Print LaTeX-formatted table."""
    print("\n% LaTeX Table")
    print("\\begin{tabular}{l" + "c" * len(METRICS) + "}")
    print("\\toprule")
    header = "Method"
    for metric in METRICS:
        header += f" & {METRIC_LABELS[metric]}"
    header += " \\\\"
    print(header)
    print("\\midrule")

    for problem in PROBLEMS:
        print(f"\\multicolumn{{{len(METRICS)+1}}}{{l}}{{\\textbf{{{problem}}}}} \\\\")
        for method_key, method_label in METHODS.items():
            key = (problem, method_key)
            data_list = results.get(key, [])
            row = f"  {method_label}"
            for metric in METRICS:
                vals = [d[metric] for d in data_list if metric in d]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals)
                    row += f" & {mean:.3f}$\\pm${std:.3f}"
                else:
                    row += " & ---"
            row += " \\\\"
            print(row)

    print("\\bottomrule")
    print("\\end{tabular}")


def main():
    results = load_results()

    # Count total results
    total = sum(len(v) for v in results.values())
    print(f"Loaded {total} result files from {RESULTS_DIR}")

    if total == 0:
        print("No results found. Run experiments first with run_all.py")
        return

    print_table(results)
    print_latex_table(results)


if __name__ == '__main__':
    main()
