"""
Phase 1: Heteroscedastic noise verification.

Samples M diverse signal plans via Latin hypercube, runs R independent
replications at each plan, and tests whether per-plan noise variances
differ across plans (Levene's / Bartlett's tests).  Saves per-rep raw
data incrementally so the experiment can resume after interruption.

Produces:
  results/intas/hetero_test.json           — final aggregated summary
  results/intas/hetero_test.partial.jsonl  — per-rep incremental log
  results/figures/figH_hetero.pdf          — scatter + box plot

Env overrides (for interactive iteration):
  INTAS_HETERO_M   — number of plans  (default 10)
  INTAS_HETERO_R   — reps per plan    (default 10)

Usage:
    python -u -m experiments.intas.run_hetero_test
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from experiments.intas.intas_problem import InTASProblem
from experiments.intas.config import RESULTS_DIR

FIG_DIR = os.path.join(os.path.dirname(RESULTS_DIR), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DEFAULT_M = int(os.environ.get("INTAS_HETERO_M", 10))
DEFAULT_R = int(os.environ.get("INTAS_HETERO_R", 10))


def _lhs_int_plans(prob: InTASProblem, m: int, seed: int = 0):
    """Latin hypercube sample of m diverse plans (integer-rounded)."""
    rng = np.random.default_rng(seed)
    lo, hi = prob.int_bounds()
    d = prob.d
    # Basic LHS: split each dim into m strata, shuffle, rescale
    plans = []
    perms = [rng.permutation(m) for _ in range(d)]
    for i in range(m):
        x = []
        for k in range(d):
            u = (perms[k][i] + rng.random()) / m        # in [0,1)
            v = int(round(lo[k] + u * (hi[k] - lo[k])))
            v = max(int(lo[k]), min(int(hi[k]), v))
            x.append(v)
        plans.append(tuple(x))
    return plans


def _load_partial(partial_path: str):
    """Load per-rep records if partial file exists.

    Returns dict: {plan_id -> list of [f1, f2, f3]}
    """
    data = {}
    if not os.path.exists(partial_path):
        return data
    with open(partial_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pid = rec.get('plan_id')
            y   = rec.get('y')
            if pid is None or y is None:
                continue
            data.setdefault(pid, []).append(list(y))
    return data


def _append_partial(partial_path: str, record: dict):
    with open(partial_path, 'a') as f:
        f.write(json.dumps(record) + "\n")


def run(m: int = None, r: int = None, seed: int = 42):
    if m is None:
        m = DEFAULT_M
    if r is None:
        r = DEFAULT_R

    out_path      = os.path.join(RESULTS_DIR, 'hetero_test.json')
    partial_path  = os.path.join(RESULTS_DIR, 'hetero_test.partial.jsonl')
    plans_path    = os.path.join(RESULTS_DIR, 'hetero_test.plans.json')

    if os.path.exists(out_path):
        print(f"Final hetero_test.json already exists. Delete to recompute.")
        return json.load(open(out_path))

    prob = InTASProblem()

    # ── Load or generate plans (deterministic given seed, cached) ─────────
    # Supports extension: if cached m < requested m, we keep the first
    # m_cached plans (plus any partial.jsonl data that refers to them) and
    # append (m - m_cached) additional LHS plans with a derived seed.
    if os.path.exists(plans_path):
        plans_data = json.load(open(plans_path))
        plans = [tuple(p) for p in plans_data['plans']]
        m_cached = len(plans)
        if m_cached == m:
            print(f"Loaded {m} cached plans from {plans_path}", flush=True)
        elif m_cached < m:
            # Extend with new LHS plans deterministically seeded from
            # the original seed + m_cached to avoid duplicates.
            extra = _lhs_int_plans(prob, m - m_cached,
                                   seed=seed + m_cached * 101)
            plans = plans + extra
            with open(plans_path, 'w') as f:
                json.dump({'plans': [list(p) for p in plans],
                           'm': m, 'seed': seed,
                           'extended_from': m_cached}, f, indent=2)
            print(f"Extended cached plans from m={m_cached} to m={m}; "
                  f"first {m_cached} kept, {m - m_cached} added",
                  flush=True)
        else:
            raise AssertionError(
                f"Plans cached with m={m_cached} but requested m={m}; "
                f"delete {plans_path} to regenerate a smaller set.")
    else:
        plans = _lhs_int_plans(prob, m, seed=seed)
        with open(plans_path, 'w') as f:
            json.dump({'plans': [list(p) for p in plans],
                       'm': m, 'seed': seed}, f, indent=2)
        print(f"Generated {m} LHS plans, cached to {plans_path}", flush=True)

    # ── Load previously completed reps from partial file ──────────────────
    partial = _load_partial(partial_path)
    total_done   = sum(len(v) for v in partial.values())
    total_needed = m * r
    print(f"\nPhase 1: Heteroscedastic noise verification "
          f"(M={m} plans × R={r} reps)", flush=True)
    print(f"  d={prob.d}  T0={prob.T0:.1f}s  A0={prob.A0:.4f}  "
          f"E0={prob.E0:.1f}kg", flush=True)
    if total_done > 0:
        print(f"  Resuming from partial: {total_done}/{total_needed} sims "
              f"already done.", flush=True)

    t_start = time.time()
    # Iterate plan-major, rep-minor — if plan 3 has 7/10 reps done, continue
    # that plan before moving to plan 4.
    n_done_this_session = 0
    for i, plan in enumerate(plans):
        pid = i
        done_for_plan = len(partial.get(pid, []))
        if done_for_plan >= r:
            continue
        for rep in range(done_for_plan, r):
            t_sim = time.time()
            y = prob.simulate(plan)
            sim_dt = time.time() - t_sim
            y_list = [float(v) for v in y]
            record = {
                'plan_id': pid,
                'rep':     rep,
                'y':       y_list,
                'sim_sec': float(sim_dt),
                'ts':      time.time(),
            }
            _append_partial(partial_path, record)
            partial.setdefault(pid, []).append(y_list)
            n_done_this_session += 1
            total_done += 1
            eta_min = (total_needed - total_done) * sim_dt / 60
            print(f"  plan {pid+1:02d}/{m} rep {rep+1:02d}/{r}  "
                  f"f1={y[0]:.3f}  f3={y[2]:.3f}  "
                  f"dt={sim_dt:.0f}s  "
                  f"session={n_done_this_session}  "
                  f"total={total_done}/{total_needed}  "
                  f"ETA={eta_min:.0f}min", flush=True)

    # ── All reps done; build final summary ────────────────────────────────
    results = []
    for i, plan in enumerate(plans):
        arr = np.array(partial[i])
        mean = arr.mean(axis=0)
        var  = arr.var(axis=0, ddof=1)
        results.append({
            'plan_id':  i,
            'plan':     list(plan),
            'mean_f1':  float(mean[0]),
            'mean_f2':  float(mean[1]),
            'mean_f3':  float(mean[2]),
            'var_f1':   float(var[0]),
            'var_f2':   float(var[1]),
            'var_f3':   float(var[2]),
            'reps':     arr.tolist(),
        })

    # ── Per-objective Levene / Bartlett ────────────────────────────────────
    mean_f1 = np.array([r['mean_f1'] for r in results])
    hetero_by_obj = {}
    for j, name in enumerate(['f1', 'f2', 'f3']):
        groups = [np.array(rr['reps'])[:, j] for rr in results]
        lev_s, lev_p = stats.levene(*groups)
        try:
            bar_s, bar_p = stats.bartlett(*groups)
        except Exception:
            bar_s, bar_p = float('nan'), float('nan')
        plan_vars = np.array([np.var(g, ddof=1) for g in groups])
        corr = float(np.corrcoef(mean_f1, plan_vars)[0, 1]) if len(plan_vars) > 2 else float('nan')
        hetero_by_obj[name] = {
            'levene_stat':   float(lev_s),
            'levene_p':      float(lev_p),
            'bartlett_stat': float(bar_s),
            'bartlett_p':    float(bar_p),
            'corr_mean_f1_vs_var': corr,
            'reject_homoscedasticity': bool(lev_p < 0.05),
            'min_var':       float(plan_vars.min()),
            'max_var':       float(plan_vars.max()),
            'var_ratio':     float(plan_vars.max() / plan_vars.min()) if plan_vars.min() > 0 else float('inf'),
        }

    summary = {
        'M': m, 'R': r,
        'wall_time_sec': float(time.time() - t_start),
        'hetero_by_obj': hetero_by_obj,
        'reject_homoscedasticity_f3': bool(hetero_by_obj['f3']['levene_p'] < 0.05),
        'results': results,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Phase 1 summary ===", flush=True)
    for name, h in hetero_by_obj.items():
        print(f"  {name}: Levene p={h['levene_p']:.4f}  "
              f"Bartlett p={h['bartlett_p']:.4f}  "
              f"corr(E[f1],var)={h['corr_mean_f1_vs_var']:+.3f}  "
              f"var_ratio={h['var_ratio']:.2f}x  "
              f"reject H0: {h['reject_homoscedasticity']}", flush=True)
    return summary


def plot(summary: dict):
    results = summary['results']
    mean_f1 = np.array([r['mean_f1'] for r in results])
    var_f3  = np.array([r['var_f3']  for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Panel (a): σ²(f3) vs E[f1] scatter ─────────────────────────────
    ax = axes[0]
    sc = ax.scatter(mean_f1, var_f3, c=var_f3, cmap='Reds', s=80,
                    edgecolors='k', linewidths=0.5)
    plt.colorbar(sc, ax=ax, label=r'$\hat\sigma^2(f^3)$')
    ax.set_xlabel(r'$\mathbb{E}[f^1]$ (efficiency ratio)')
    ax.set_ylabel(r'Sample variance $\hat\sigma^2(f^3)$')
    ax.set_title(r'Emission noise variance vs.\ efficiency level')
    if len(mean_f1) >= 2 and mean_f1.std() > 0:
        z = np.polyfit(mean_f1, var_f3, 1)
        xx = np.linspace(mean_f1.min(), mean_f1.max(), 100)
        r_pearson = np.corrcoef(mean_f1, var_f3)[0, 1]
        ax.plot(xx, np.polyval(z, xx), 'b--', linewidth=1.2, alpha=0.7,
                label=f"linear fit (r={r_pearson:.2f})")
        ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel (b): box plot of f3 by plan, sorted by E[f1] ─────────────
    ax2 = axes[1]
    f3_groups = [np.array(r['reps'])[:, 2] for r in results]
    order = np.argsort(mean_f1)
    f3_sorted = [f3_groups[i] for i in order]
    bp = ax2.boxplot(f3_sorted, patch_artist=True, showfliers=False,
                     medianprops={'color': 'red', 'linewidth': 1.5})
    cmap = plt.cm.coolwarm
    for patch, i in zip(bp['boxes'], range(len(order))):
        patch.set_facecolor(cmap(i / max(1, len(order)-1)))
    ax2.set_xlabel(r'Signal plans (sorted by $\mathbb{E}[f^1]$, low$\to$high)')
    ax2.set_ylabel(r'$f^3$ distribution (per plan)')
    ax2.set_title(r'Noise distribution varies across plans')
    ax2.axhline(1.0, linestyle='--', color='gray', linewidth=0.8,
                label='Baseline level')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis='y')
    lev_p = summary['hetero_by_obj']['f3']['levene_p']
    ax2.text(0.02, 0.97,
             f"Levene $p$={lev_p:.3f} ({'reject $H_0$' if lev_p<0.05 else 'fail to reject'})",
             transform=ax2.transAxes, fontsize=8, va='top',
             color='red' if lev_p < 0.05 else 'gray')

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        fp = os.path.join(FIG_DIR, f'figH_hetero.{ext}')
        fig.savefig(fp, dpi=180, bbox_inches='tight')
        print(f"  saved {fp}")
    plt.close(fig)


def main():
    summary = run()
    print(f"\n--- final check ---", flush=True)
    for name, h in summary['hetero_by_obj'].items():
        print(f"  {name}: Levene p={h['levene_p']:.4f}  "
              f"reject H0: {h['reject_homoscedasticity']}")
    plot(summary)


if __name__ == "__main__":
    main()
