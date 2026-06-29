"""
Master script to update the paper with experiment results.

1. Aggregates all available results from results/sec6X directories
2. Generates publication-quality figures (PDF + PNG)
3. Produces LaTeX table content
4. Updates the paper .tex file by replacing placeholder values

Run this whenever new results are available.
"""

import sys
import os
import json
import numpy as np
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from gpr_kg import compute_hypervolume_2d
from metrics import compute_igd, compute_cvr, wilcoxon_test
from experiments.config import METHOD_NAMES, REF_POINT, N_REPS

PAPER_PATH = os.path.join(os.path.dirname(BASE_DIR),
                          "Final_Revised_Manuscript_OR.tex")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def collect_results(section_dir, problem_name, method_name, n_reps=30):
    """Collect all available results for a method-problem pair."""
    safe_name = method_name.replace('-', '_')
    hvs, igds, cvrs, times, time_per_iters = [], [], [], [], []
    hv_histories = []

    for rep in range(1, n_reps + 1):
        path = os.path.join(section_dir, f"{problem_name}_{safe_name}_rep{rep}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        hvs.append(data['hv_final'])
        igds.append(data['igd_final'])
        cvrs.append(data['cvr_final'])
        times.append(data.get('total_wall_time', data.get('total_time_sec', 0)))
        time_per_iters.append(data.get('time_per_iter_mean', 0))
        if 'hv_history' in data:
            hv_histories.append(data['hv_history'])

    if not hvs:
        return None

    n = len(hvs)
    return {
        'n': n,
        'hv_mean': np.mean(hvs), 'hv_se': np.std(hvs)/np.sqrt(n),
        'hv_std': np.std(hvs), 'hv_values': hvs,
        'igd_mean': np.mean(igds), 'igd_se': np.std(igds)/np.sqrt(n),
        'igd_std': np.std(igds), 'igd_values': igds,
        'cvr_mean': np.mean(cvrs), 'cvr_se': np.std(cvrs)/np.sqrt(n),
        'cvr_std': np.std(cvrs), 'cvr_values': cvrs,
        'time_mean': np.mean(times),
        'time_per_iter_mean': np.mean(time_per_iters),
        'hv_histories': hv_histories,
    }


def generate_all_summaries():
    """Generate summary JSONs for all sections."""
    sections = {
        'sec62': {'problems': ['RZDT1', 'RZDT2', 'RZDT5']},
        'sec63': {'problems': ['RZDT1_d5', 'RZDT1_d10', 'RZDT1_d20']},
        'sec64': {'problems': ['RZDT1_loose', 'RZDT1_moderate', 'RZDT1_tight']},
        'sec65': {'problems': ['RZDT1_sigma001', 'RZDT1_sigma01',
                               'RZDT1_sigma10', 'RZDT1_hetero']},
        'sec66': {'problems': ['RZDT1']},  # ablation configs
    }

    for sec_name, sec_info in sections.items():
        section_dir = os.path.join(BASE_DIR, 'results', sec_name)
        if not os.path.exists(section_dir):
            continue

        summary = {}
        for prob in sec_info['problems']:
            summary[prob] = {}
            # For sec66, use ablation config names
            if sec_name == 'sec66':
                method_list = ['default', 'vepm_off', 'cand_30', 'cand_150']
            else:
                method_list = METHOD_NAMES

            for method in method_list:
                data = collect_results(section_dir, prob, method)
                if data:
                    summary[prob][method] = data
                    # Remove hv_histories from summary (too large)
                    summary[prob][method] = {
                        k: v for k, v in data.items() if k != 'hv_histories'
                    }

        with open(os.path.join(section_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"  {sec_name}: summary generated")

    return sections


def generate_figure1():
    """Figure 1: HV convergence on RZDT1."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['axes.labelsize'] = 14

    section_dir = os.path.join(BASE_DIR, 'results', 'sec62')
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {
        'GPR-KG': '#1f77b4', 'GPR-KG-nV': '#ff7f0e', 'cEHVI': '#2ca02c',
        'cParEGO': '#d62728', 'NSGA-II-K': '#9467bd', 'NSGA-II-D': '#8c564b',
        'RS': '#7f7f7f',
    }
    linestyles = {
        'GPR-KG': '-', 'GPR-KG-nV': '--', 'cEHVI': '-.', 'cParEGO': ':',
        'NSGA-II-K': '-', 'NSGA-II-D': '--', 'RS': ':',
    }

    nsga2d_final = None

    for method in METHOD_NAMES:
        data = collect_results(section_dir, 'RZDT1', method)
        if data is None or not data['hv_histories']:
            continue

        histories = data['hv_histories']
        all_stages = sorted(set(s for h in histories for s, _ in h))
        if not all_stages:
            continue

        hv_matrix = np.full((len(histories), len(all_stages)), np.nan)
        for i, hist in enumerate(histories):
            d = dict(hist)
            for j, s in enumerate(all_stages):
                if s in d:
                    hv_matrix[i, j] = d[s]

        # Forward fill
        for i in range(hv_matrix.shape[0]):
            for j in range(1, hv_matrix.shape[1]):
                if np.isnan(hv_matrix[i, j]):
                    hv_matrix[i, j] = hv_matrix[i, j-1]

        valid = ~np.isnan(hv_matrix).any(axis=0)
        stages = np.array(all_stages)[valid]
        vals = hv_matrix[:, valid]
        mean = np.mean(vals, axis=0)
        se = np.std(vals, axis=0) / np.sqrt(len(histories))

        color = colors.get(method, 'black')
        ls = linestyles.get(method, '-')
        ax.plot(stages, mean, color=color, linestyle=ls, linewidth=2, label=method)
        ax.fill_between(stages, mean - se, mean + se, color=color, alpha=0.15)

        if method == 'NSGA-II-D' and len(mean) > 0:
            nsga2d_final = mean[-1]

    if nsga2d_final is not None:
        ax.axhline(y=nsga2d_final, color='#8c564b', linestyle=':', alpha=0.5)

    ax.set_xlabel('Number of Simulation Evaluations')
    ax.set_ylabel('Dominated Hypervolume')
    ax.set_title('HV Convergence on RZDT1 ($d=5$, $\\sigma=0.1$)')
    ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    for ext in ['pdf', 'png']:
        plt.savefig(os.path.join(FIGURES_DIR, f'fig1_convergence.{ext}'),
                    dpi=300 if ext == 'pdf' else 200, bbox_inches='tight')
    plt.close()
    print("  Figure 1 saved.")


def format_table_value(mean, is_best=False, is_dagger=False):
    """Format a table cell value."""
    val = f"{mean:.3f}"
    if is_best:
        val = f"\\textbf{{{val}}}"
    if is_dagger:
        val += "$^\\dagger$"
    return val


def generate_table1_latex():
    """Generate Table 1 LaTeX content."""
    section_dir = os.path.join(BASE_DIR, 'results', 'sec62')
    problems = ['RZDT1', 'RZDT2', 'RZDT5']
    metrics = [('hv', 'max'), ('igd', 'min'), ('cvr', 'min')]

    lines = []
    for method in METHOD_NAMES:
        parts = [method]
        for prob in problems:
            data = collect_results(section_dir, prob, method)
            for metric, direction in metrics:
                if data is None:
                    parts.append('---')
                    continue

                mean = data[f'{metric}_mean']

                # Find best
                all_vals = {}
                for m in METHOD_NAMES:
                    d = collect_results(section_dir, prob, m)
                    if d:
                        all_vals[m] = d[f'{metric}_mean']

                if direction == 'max':
                    is_best = mean >= max(all_vals.values()) - 1e-10
                else:
                    is_best = mean <= min(all_vals.values()) + 1e-10

                # Wilcoxon for GPR-KG
                is_dag = False
                if method == 'GPR-KG' and len(all_vals) > 1:
                    comps = {m: v for m, v in all_vals.items() if m != 'GPR-KG'}
                    if direction == 'max':
                        best_comp_name = max(comps, key=comps.get)
                    else:
                        best_comp_name = min(comps, key=comps.get)
                    comp_data = collect_results(section_dir, prob, best_comp_name)
                    if comp_data and len(data[f'{metric}_values']) >= 10:
                        n_min = min(len(data[f'{metric}_values']),
                                    len(comp_data[f'{metric}_values']))
                        test = wilcoxon_test(
                            data[f'{metric}_values'][:n_min],
                            comp_data[f'{metric}_values'][:n_min])
                        is_dag = test['significant']

                parts.append(format_table_value(mean, is_best, is_dag))

        # Format LaTeX line
        line = parts[0].ljust(15)
        for i in range(1, len(parts)):
            line += f" & {parts[i]}"
        line += " \\\\"
        lines.append(line)

    return "\n".join(lines)


def print_all_results():
    """Print current status of all experiments."""
    print("\n" + "=" * 80)
    print("  CURRENT EXPERIMENT RESULTS")
    print("=" * 80)

    # Table 1
    section_dir = os.path.join(BASE_DIR, 'results', 'sec62')
    problems = ['RZDT1', 'RZDT2', 'RZDT5']
    print(f"\n{'Method':15s}", end="")
    for p in problems:
        print(f" | {'HV':>8s} {'IGD':>7s} {'CVR':>5s} (n)", end="")
    print()
    print("-" * 95)

    for method in METHOD_NAMES:
        print(f"{method:15s}", end="")
        for prob in problems:
            data = collect_results(section_dir, prob, method)
            if data:
                print(f" | {data['hv_mean']:>8.4f} {data['igd_mean']:>7.4f} "
                      f"{data['cvr_mean']:>5.3f} ({data['n']:>2d})", end="")
            else:
                print(f" | {'---':>8s} {'---':>7s} {'---':>5s} ( 0)", end="")
        print()

    # Sections 6.3-6.5 brief summary
    for sec, configs in [
        ('sec63', ['RZDT1_d5', 'RZDT1_d10', 'RZDT1_d20']),
        ('sec64', ['RZDT1_loose', 'RZDT1_moderate', 'RZDT1_tight']),
        ('sec65', ['RZDT1_sigma001', 'RZDT1_sigma01', 'RZDT1_sigma10', 'RZDT1_hetero']),
    ]:
        print(f"\n--- {sec} ---")
        sec_dir = os.path.join(BASE_DIR, 'results', sec)
        for cfg in configs:
            completed = sum(1 for m in METHOD_NAMES
                           for r in range(1, 31)
                           if os.path.exists(os.path.join(sec_dir,
                               f"{cfg}_{m.replace('-','_')}_rep{r}.json")))
            total = len(METHOD_NAMES) * 30
            print(f"  {cfg}: {completed}/{total}")


if __name__ == '__main__':
    print("Generating summaries...")
    generate_all_summaries()

    print("\nGenerating Figure 1...")
    generate_figure1()

    print("\nTable 1 LaTeX:")
    print(generate_table1_latex())

    print_all_results()
