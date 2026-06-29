"""Generate LaTeX table content from experiment results."""
import json, glob, numpy as np, os, sys
from scipy.stats import wilcoxon

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METHODS = ['GPR-KG', 'GPR-KG-nV', 'cEHVI', 'cParEGO', 'NSGA-II-K', 'NSGA-II-D', 'RS']

def load_vals(section_dir, prob, method, metric):
    safe = method.replace('-', '_')
    files = sorted(glob.glob(os.path.join(section_dir, f'{prob}_{safe}_rep*.json')))
    vals = []
    for f in files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            vals.append(d[metric])
        except (json.JSONDecodeError, KeyError):
            pass  # Skip corrupt or incomplete files
    return vals

def fmt_val(vals, all_method_vals, method, direction, show_dag=True):
    """Format a metric value: bold if best, dagger if GPR-KG significantly better."""
    if not vals:
        return '---'
    mean = np.mean(vals)

    # Check if best
    best_means = {m: np.mean(v) for m, v in all_method_vals.items() if v}
    if direction == 'max':
        is_best = mean >= max(best_means.values()) - 1e-10
    else:
        is_best = mean <= min(best_means.values()) + 1e-10

    # Wilcoxon for GPR-KG
    dag = False
    if show_dag and method == 'GPR-KG' and len(vals) >= 10:
        others = {m: v for m, v in all_method_vals.items()
                  if m != 'GPR-KG' and len(v) >= 10}
        if others:
            if direction == 'max':
                best_other = max(others, key=lambda m: np.mean(others[m]))
            else:
                best_other = min(others, key=lambda m: np.mean(others[m]))
            n_min = min(len(vals), len(others[best_other]))
            try:
                _, p = wilcoxon(vals[:n_min], others[best_other][:n_min])
                dag = p < 0.05
            except:
                pass

    s = f'{mean:.3f}'
    if is_best:
        s = f'\\textbf{{{s}}}'
    if dag:
        s += '$^\\dagger$'
    return s


def generate_table1():
    """Table 1: Section 6.2 Standard Benchmarks."""
    sec = os.path.join(BASE_DIR, 'results', 'sec62')
    problems = ['RZDT1', 'RZDT2', 'RZDT5']
    metrics = [('hv_final', 'max'), ('igd_final', 'min'), ('cvr_final', 'min')]

    print("% TABLE 1: Standard Benchmarks")
    for method in METHODS:
        parts = [method.ljust(14)]
        for prob in problems:
            for metric, direction in metrics:
                all_vals = {m: load_vals(sec, prob, m, metric) for m in METHODS}
                vals = all_vals[method]
                parts.append(fmt_val(vals, all_vals, method, direction))

        # Format: method & v1 & v2 & v3 & v4 & v5 & v6 & v7 & v8 & v9 \\
        line = parts[0]
        # Group by problem (3 metrics each)
        for i in range(3):  # 3 problems
            for j in range(3):  # 3 metrics
                idx = 1 + i * 3 + j
                line += f' & {parts[idx]}'
            if i < 2:
                line += '\n             '
        line += ' \\\\'
        print(line)


def generate_table2():
    """Table 2: Section 6.3 Scalability."""
    sec = os.path.join(BASE_DIR, 'results', 'sec63')
    dims = [5, 10, 20]

    print("\n% TABLE 2: Scalability")
    for method in METHODS:
        parts = [method.ljust(14)]
        # HV for each dimension
        for d in dims:
            prob = f'RZDT1_d{d}'
            all_vals = {m: load_vals(sec, prob, m, 'hv_final') for m in METHODS}
            parts.append(fmt_val(all_vals[method], all_vals, method, 'max'))
        # Time per iter for each dimension
        for d in dims:
            prob = f'RZDT1_d{d}'
            vals = load_vals(sec, prob, method, 'time_per_iter_mean')
            if vals:
                parts.append(f'{np.mean(vals):.1f}')
            else:
                parts.append('---')

        line = parts[0] + ' & ' + ' & '.join(parts[1:4])
        line += '\n              & ' + ' & '.join(parts[4:7]) + ' \\\\'
        print(line)


def generate_table3():
    """Table 3: Section 6.4 Constraint Tightness."""
    sec = os.path.join(BASE_DIR, 'results', 'sec64')
    levels = ['loose', 'moderate', 'tight']
    metrics = [('hv_final', 'max'), ('igd_final', 'min'), ('cvr_final', 'min')]

    print("\n% TABLE 3: Constraint Tightness")
    for method in METHODS:
        parts = [method.ljust(14)]
        for level in levels:
            prob = f'RZDT1_{level}'
            for metric, direction in metrics:
                all_vals = {m: load_vals(sec, prob, m, metric) for m in METHODS}
                parts.append(fmt_val(all_vals[method], all_vals, method, direction))

        line = parts[0]
        for i in range(3):
            for j in range(3):
                idx = 1 + i * 3 + j
                line += f' & {parts[idx]}'
            if i < 2:
                line += '\n             '
        line += ' \\\\'
        print(line)


def generate_table4():
    """Table 4: Section 6.5 Noise Sensitivity."""
    sec = os.path.join(BASE_DIR, 'results', 'sec65')
    noise_configs = ['sigma001', 'sigma01', 'sigma10', 'hetero']

    print("\n% TABLE 4: Noise Sensitivity")
    for method in METHODS:
        parts = [method.ljust(14)]
        # HV for each noise level
        for noise in noise_configs:
            prob = f'RZDT1_{noise}'
            all_vals = {m: load_vals(sec, prob, m, 'hv_final') for m in METHODS}
            parts.append(fmt_val(all_vals[method], all_vals, method, 'max'))
        # CVR for each noise level
        for noise in noise_configs:
            prob = f'RZDT1_{noise}'
            all_vals = {m: load_vals(sec, prob, m, 'cvr_final') for m in METHODS}
            parts.append(fmt_val(all_vals[method], all_vals, method, 'min', show_dag=False))

        line = parts[0] + ' & ' + ' & '.join(parts[1:5])
        line += '\n              & ' + ' & '.join(parts[5:9]) + ' \\\\'
        print(line)


def generate_table5():
    """Table 5: Section 6.6 Ablation Study."""
    sec = os.path.join(BASE_DIR, 'results', 'sec66')
    configs = ['default', 'vepm_off', 'cand_30', 'cand_150']
    labels = {
        'default': ('Default (GPR-KG)', '\\multicolumn{2}{l}{\\emph{Default (GPR-KG)}}'),
        'vepm_off': ('Off (GPR-KG-nV)', 'VEPM & Off (GPR-KG-nV)'),
        'cand_30': ('30', 'Candidates & 30'),
        'cand_150': ('150', ' & 150'),
    }

    print("\n% TABLE 5: Ablation Study")
    for cfg in configs:
        safe = cfg
        files = sorted(glob.glob(os.path.join(sec, f'RZDT1_{safe}_rep*.json')))
        if not files:
            print(f"% {cfg}: no data")
            continue
        hvs, igds, cvrs, tpis = [], [], [], []
        for f in files:
            with open(f) as fh:
                d = json.load(fh)
            hvs.append(d['hv_final'])
            igds.append(d['igd_final'])
            cvrs.append(d['cvr_final'])
            tpis.append(d.get('time_per_iter_mean', 0))

        hv_s = f'{np.mean(hvs):.3f}'
        igd_s = f'{np.mean(igds):.3f}'
        cvr_s = f'{np.mean(cvrs):.2f}'
        t_s = f'{np.mean(tpis):.1f}'

        if cfg == 'default':
            hv_s = f'\\textbf{{{hv_s}}}'
            igd_s = f'\\textbf{{{igd_s}}}'
            cvr_s = f'\\textbf{{{cvr_s}}}'

        label = labels[cfg][1]
        print(f'  {label} & {hv_s} & {igd_s} & {cvr_s} & {t_s} \\\\')


if __name__ == '__main__':
    generate_table1()
    generate_table2()
    generate_table3()
    generate_table4()
    generate_table5()
