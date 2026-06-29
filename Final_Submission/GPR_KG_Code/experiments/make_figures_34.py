"""
Generate Fig.3 (posterior mean RMSE history),
Fig.4 (#infeasible in reported Pareto set over iterations),
Fig.5a (VEPM vs pooled variance tracking at a fixed unsampled TPOS).

Reads snapshots recorded by experiments/run_instrumented.py from
results/instrumented/{PROB}/{METHOD}_rep{rr}.json.

Usage:
    python -m experiments.make_figures_34
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT1, RZDT2, RZDT5_RR

INSTR_ROOT = os.path.join(BASE_DIR, "results", "instrumented")
OUT_DIR    = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

PROBLEM_CLASSES = {
    "RZDT1":    RZDT1,
    "RZDT2":    RZDT2,
    "RZDT5_RR": RZDT5_RR,
}
PROBLEM_NAMES = ["RZDT1", "RZDT2", "RZDT5_RR"]
LABELS_ABC    = ["(a) RZDT1", "(b) RZDT2", r"(c) RZDT5\_RR"]

METHODS = ["GPR-KG", "GPR-KG-nV"]
SAFE    = {m: m.replace("-", "_") for m in METHODS}
COLORS  = {"GPR-KG": "#d62728", "GPR-KG-nV": "#1f77b4"}
MARKERS = {"GPR-KG": "o",       "GPR-KG-nV": "s"}


def _mk(name):
    p = PROBLEM_CLASSES[name](d=5, sigma=0.04, heteroscedastic=True, alpha=0.05)
    p.tau = 0.0
    return p


def load_reps(prob_name, method):
    pattern = os.path.join(INSTR_ROOT, prob_name, f"{SAFE[method]}_rep*.json")
    files = sorted(glob.glob(pattern))
    out = []
    for f in files:
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


# ────────────────────────────────────────────────────────────
#  FIGURE 3: posterior mean RMSE at eval_x, averaged over the
#  3 objectives and |eval_x| evaluation points.
# ────────────────────────────────────────────────────────────
def fig3():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for idx, pname in enumerate(PROBLEM_NAMES):
        ax = axes[idx]
        for m in METHODS:
            reps = load_reps(pname, m)
            if not reps:
                continue

            # true_means is a (K,3) array stored once per rep (identical across reps)
            true_means = np.array(reps[0]['instrument_config']['true_means'])  # (K,3)

            # Gather per-rep RMSE trajectories
            all_iter, all_rmse = [], []
            for r in reps:
                iters, rmses = [], []
                for snap in r['instrument_log']:
                    mu = np.stack([snap['mu0_eval'],
                                   snap['mu1_eval'],
                                   snap['mu2_eval']], axis=1)  # (K,3)
                    diff = mu - true_means
                    rmse = float(np.sqrt(np.mean(diff ** 2)))
                    iters.append(snap['n'])
                    rmses.append(rmse)
                all_iter.append(np.array(iters))
                all_rmse.append(np.array(rmses))

            grid = all_iter[0]
            Y = np.vstack([np.interp(grid, x, y) for x, y in zip(all_iter, all_rmse)])
            mean = Y.mean(axis=0)
            se   = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)

            ax.plot(grid, mean, color=COLORS[m], marker=MARKERS[m],
                    markersize=4.5, linewidth=1.3,
                    label=m if idx == 0 else None)
            ax.fill_between(grid, mean - se, mean + se,
                            color=COLORS[m], alpha=0.15)

        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel(r"RMSE$(\hat\mu - f)$")
        ax.set_title(LABELS_ABC[idx], fontsize=11)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9,
               bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    for ext in ["png", "pdf"]:
        fp = os.path.join(OUT_DIR, f"fig3_rmse.{ext}")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        print(f"  saved {fp}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
#  FIGURE 4: # infeasible solutions in reported Pareto set
#  over iterations.  Feasibility is checked against the true
#  constraint (prob.is_truly_feasible), mean ± SE across reps.
# ────────────────────────────────────────────────────────────
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for idx, pname in enumerate(PROBLEM_NAMES):
        ax   = axes[idx]
        prob = _mk(pname)
        feas_cache = {}  # tuple -> bool

        def is_feas(xt):
            if xt not in feas_cache:
                feas_cache[xt] = bool(prob.is_truly_feasible(xt))
            return feas_cache[xt]

        for m in METHODS:
            reps = load_reps(pname, m)
            if not reps:
                continue

            all_iter, all_infeas = [], []
            for r in reps:
                iters, cnts = [], []
                for snap in r['instrument_log']:
                    ps = [tuple(int(v) for v in x) for x in snap['pareto_set']]
                    n_bad = sum(1 for x in ps if not is_feas(x))
                    iters.append(snap['n'])
                    cnts.append(n_bad)
                all_iter.append(np.array(iters))
                all_infeas.append(np.array(cnts, dtype=float))

            grid = all_iter[0]
            Y = np.vstack([np.interp(grid, x, y) for x, y in zip(all_iter, all_infeas)])
            mean = Y.mean(axis=0)
            se   = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)

            ax.plot(grid, mean, color=COLORS[m], marker=MARKERS[m],
                    markersize=4.5, linewidth=1.3,
                    label=m if idx == 0 else None)
            ax.fill_between(grid, mean - se, mean + se,
                            color=COLORS[m], alpha=0.15)

        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel("# infeasible in reported PF")
        ax.set_title(LABELS_ABC[idx], fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=-0.1)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9,
               bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    for ext in ["png", "pdf"]:
        fp = os.path.join(OUT_DIR, f"fig4_infeas.{ext}")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        print(f"  saved {fp}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
#  FIGURE 5a: variance estimate at a fixed unsampled TPOS
#  vs the true σ² (horizontal dashed).  Uses constraint
#  objective (i=2, f^3) — most relevant for feasibility.
# ────────────────────────────────────────────────────────────
def fig5a(i_obj=2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for idx, pname in enumerate(PROBLEM_NAMES):
        ax = axes[idx]

        true_sigma_sq = None
        for m in METHODS:
            reps = load_reps(pname, m)
            if not reps:
                continue

            # true_sigma_ref is std dev — square it for variance comparison
            sig_ref = reps[0]['instrument_config']['true_sigma_ref']  # [σ1, σ2, σ3]
            true_sigma_sq = float(sig_ref[i_obj]) ** 2

            all_iter, all_var = [], []
            for r in reps:
                iters, vs = [], []
                for snap in r['instrument_log']:
                    if 'sigma2_ref' not in snap:
                        continue
                    iters.append(snap['n'])
                    vs.append(float(snap['sigma2_ref'][i_obj]))
                if not iters:
                    continue
                all_iter.append(np.array(iters))
                all_var.append(np.array(vs))

            if not all_iter:
                continue

            grid = all_iter[0]
            Y = np.vstack([np.interp(grid, x, y) for x, y in zip(all_iter, all_var)])
            mean = Y.mean(axis=0)
            se   = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0]) if Y.shape[0] > 1 else np.zeros_like(mean)

            ax.plot(grid, mean, color=COLORS[m], marker=MARKERS[m],
                    markersize=4.5, linewidth=1.3,
                    label=m if idx == 0 else None)
            ax.fill_between(grid, mean - se, mean + se,
                            color=COLORS[m], alpha=0.15)

        if true_sigma_sq is not None:
            ax.axhline(true_sigma_sq, linestyle="--", color="black",
                       linewidth=1.0,
                       label=r"true $\sigma^2$" if idx == 0 else None)

        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel(r"$\hat\sigma^2_3$ at reference $x^*$")
        ax.set_title(LABELS_ABC[idx], fontsize=11)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", fontsize=9,
               bbox_to_anchor=(1.0, 0.5))
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    for ext in ["png", "pdf"]:
        fp = os.path.join(OUT_DIR, f"fig5a_variance.{ext}")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        print(f"  saved {fp}")
    plt.close(fig)


def main():
    print("[fig3] Posterior mean RMSE ...")
    fig3()
    print("[fig4] # infeasible in reported PF ...")
    fig4()
    print("[fig5a] Variance tracking ...")
    fig5a()
    print(f"\nDone. Figures at {OUT_DIR}/")


if __name__ == "__main__":
    main()
