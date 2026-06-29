"""
Generate LaTeX table content from experiment results.

Reads JSON results from results/sec6X/ directories and produces
formatted LaTeX table rows with:
- Bold for best values
- Dagger (†) for statistically significant improvements (Wilcoxon p < 0.05)
"""

import sys
import os
import json
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from metrics import wilcoxon_test
from experiments.config import METHOD_NAMES


def load_summary(section_dir):
    """Load summary.json from a section directory."""
    path = os.path.join(section_dir, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def format_mean_se(mean, se, bold=False, dagger=False):
    """Format mean ± SE for LaTeX."""
    dag = "$^\\dagger$" if dagger else ""
    if bold:
        return f"\\textbf{{{mean:.3f}}} ({se:.3f}){dag}"
    return f"{mean:.3f} ({se:.3f}){dag}"


def format_mean_se_short(mean, se, bold=False, dagger=False):
    """Short format: mean(SE)."""
    dag = "$^\\dagger$" if dagger else ""
    if bold:
        return f"\\textbf{{{mean:.3f}}}{dag}"
    return f"{mean:.3f}{dag}"


def generate_table1():
    """Table 1: Standard benchmarks (Section 6.2).

    Format: method | RZDT1(HV, IGD, CVR) | RZDT2(HV, IGD, CVR) | RZDT5(HV, IGD, CVR)
    """
    section_dir = os.path.join(BASE_DIR, "results", "sec62")
    summary = load_summary(section_dir)
    if summary is None:
        print("No Section 6.2 results found.")
        return ""

    problems = ['RZDT1', 'RZDT2', 'RZDT5']
    metrics = ['hv', 'igd', 'cvr']
    directions = {'hv': 'max', 'igd': 'min', 'cvr': 'min'}  # higher/lower is better

    lines = []
    for method in METHOD_NAMES:
        parts = [method.replace('-', '{-}')]
        for prob in problems:
            if prob not in summary or method not in summary[prob]:
                parts.extend(['---'] * 3)
                continue

            data = summary[prob][method]
            for metric in metrics:
                mean = data[f'{metric}_mean']
                se = data[f'{metric}_se']

                # Check if best
                all_means = {m: summary[prob][m][f'{metric}_mean']
                             for m in METHOD_NAMES if m in summary[prob]}
                if directions[metric] == 'max':
                    is_best = mean >= max(all_means.values()) - 1e-10
                else:
                    is_best = mean <= min(all_means.values()) + 1e-10

                # Wilcoxon test: GPR-KG vs this method
                is_dag = False
                if method == 'GPR-KG' and len(all_means) > 1:
                    # Find best non-GPR-KG method
                    competitors = {m: v for m, v in all_means.items() if m != 'GPR-KG'}
                    if directions[metric] == 'max':
                        best_comp = max(competitors.items(), key=lambda x: x[1])
                    else:
                        best_comp = min(competitors.items(), key=lambda x: x[1])

                    best_comp_name = best_comp[0]
                    if best_comp_name in summary[prob]:
                        gpr_vals = data[f'{metric}_values']
                        comp_vals = summary[prob][best_comp_name][f'{metric}_values']
                        if len(gpr_vals) == len(comp_vals):
                            test = wilcoxon_test(gpr_vals, comp_vals)
                            is_dag = test['significant']

                parts.append(format_mean_se_short(mean, se, bold=is_best, dagger=is_dag))

        lines.append(" & ".join(parts) + " \\\\")

    return "\n".join(lines)


def generate_table2():
    """Table 2: Scalability (Section 6.3).

    Format: method | d=5(HV, Time) | d=10(HV, Time) | d=20(HV, Time)
    """
    section_dir = os.path.join(BASE_DIR, "results", "sec63")
    summary = load_summary(section_dir)
    if summary is None:
        print("No Section 6.3 results found.")
        return ""

    dims = [5, 10, 20]
    lines = []
    for method in METHOD_NAMES:
        parts = [method.replace('-', '{-}')]
        for d in dims:
            prob_name = f"RZDT1_d{d}"
            if prob_name not in summary or method not in summary[prob_name]:
                parts.extend(['---', '---'])
                continue

            data = summary[prob_name][method]
            hv_mean = data['hv_mean']
            time_mean = data['time_per_iter_mean']

            # Check if best HV
            all_hv = {m: summary[prob_name][m]['hv_mean']
                      for m in METHOD_NAMES if m in summary[prob_name]}
            is_best = hv_mean >= max(all_hv.values()) - 1e-10

            parts.append(format_mean_se_short(hv_mean, data['hv_se'], bold=is_best))
            parts.append(f"{time_mean:.2f}")

        lines.append(" & ".join(parts) + " \\\\")

    return "\n".join(lines)


def generate_table3():
    """Table 3: Constraint tightness (Section 6.4)."""
    section_dir = os.path.join(BASE_DIR, "results", "sec64")
    summary = load_summary(section_dir)
    if summary is None:
        print("No Section 6.4 results found.")
        return ""

    levels = ['loose', 'moderate', 'tight']
    metrics = ['hv', 'igd', 'cvr']
    directions = {'hv': 'max', 'igd': 'min', 'cvr': 'min'}

    lines = []
    for method in METHOD_NAMES:
        parts = [method.replace('-', '{-}')]
        for level in levels:
            prob_name = f"RZDT1_{level}"
            if prob_name not in summary or method not in summary[prob_name]:
                parts.extend(['---'] * 3)
                continue

            data = summary[prob_name][method]
            for metric in metrics:
                mean = data[f'{metric}_mean']
                se = data[f'{metric}_se']
                all_means = {m: summary[prob_name][m][f'{metric}_mean']
                             for m in METHOD_NAMES if m in summary[prob_name]}
                if directions[metric] == 'max':
                    is_best = mean >= max(all_means.values()) - 1e-10
                else:
                    is_best = mean <= min(all_means.values()) + 1e-10
                parts.append(format_mean_se_short(mean, se, bold=is_best))

        lines.append(" & ".join(parts) + " \\\\")

    return "\n".join(lines)


def generate_table4():
    """Table 4: Noise sensitivity (Section 6.5)."""
    section_dir = os.path.join(BASE_DIR, "results", "sec65")
    summary = load_summary(section_dir)
    if summary is None:
        print("No Section 6.5 results found.")
        return ""

    noise_names = ['sigma001', 'sigma01', 'sigma10', 'hetero']
    noise_labels = ['0.01', '0.1', '1.0', 'Hetero']

    lines = []
    for method in METHOD_NAMES:
        parts = [method.replace('-', '{-}')]
        for noise_name in noise_names:
            prob_name = f"RZDT1_{noise_name}"
            if prob_name not in summary or method not in summary[prob_name]:
                parts.extend(['---', '---'])
                continue

            data = summary[prob_name][method]
            hv_mean = data['hv_mean']
            cvr_mean = data['cvr_mean']

            all_hv = {m: summary[prob_name][m]['hv_mean']
                      for m in METHOD_NAMES if m in summary[prob_name]}
            is_best_hv = hv_mean >= max(all_hv.values()) - 1e-10

            all_cvr = {m: summary[prob_name][m]['cvr_mean']
                       for m in METHOD_NAMES if m in summary[prob_name]}
            is_best_cvr = cvr_mean <= min(all_cvr.values()) + 1e-10

            parts.append(format_mean_se_short(hv_mean, data['hv_se'], bold=is_best_hv))
            parts.append(format_mean_se_short(cvr_mean, data['cvr_se'], bold=is_best_cvr))

        lines.append(" & ".join(parts) + " \\\\")

    return "\n".join(lines)


def generate_table5():
    """Table 5: Ablation study (Section 6.6)."""
    section_dir = os.path.join(BASE_DIR, "results", "sec66")
    summary = load_summary(section_dir)
    if summary is None:
        print("No Section 6.6 results found.")
        return ""

    lines = []
    for config_name, data in summary.items():
        parts = [config_name]
        parts.append(f"{data['hv_mean']:.3f}")
        parts.append(f"{data['igd_mean']:.3f}")
        parts.append(f"{data['cvr_mean']:.3f}")
        parts.append(f"{data['time_per_iter_mean']:.2f}")
        lines.append(" & ".join(parts) + " \\\\")

    return "\n".join(lines)


def generate_all_tables():
    """Generate all tables and print."""
    print("\n" + "=" * 70)
    print("TABLE 1: Standard Benchmarks")
    print("=" * 70)
    print(generate_table1())

    print("\n" + "=" * 70)
    print("TABLE 2: Scalability")
    print("=" * 70)
    print(generate_table2())

    print("\n" + "=" * 70)
    print("TABLE 3: Constraint Tightness")
    print("=" * 70)
    print(generate_table3())

    print("\n" + "=" * 70)
    print("TABLE 4: Noise Sensitivity")
    print("=" * 70)
    print(generate_table4())

    print("\n" + "=" * 70)
    print("TABLE 5: Ablation Study")
    print("=" * 70)
    print(generate_table5())


if __name__ == '__main__':
    generate_all_tables()
