"""
Generate Fig.1 (bi-objective scatter), Fig.2 (HV convergence),
and Fig.5 (VEPM ablation) for RZDT1, RZDT2, RZDT5_RR.

Output PNG and PDF files under  results/figures/  for embedding in the paper.

Usage:
    python -m experiments.make_figures
"""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT1, RZDT2, RZDT5_RR

OUT_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["GPR-KG", "GPR-KG-nV", "cEHVI", "cParEGO",
           "NSGA-II-K", "NSGA-II-D", "RS"]
SAFE = {m: m.replace("-", "_") for m in METHODS}

METHOD_COLORS = {
    "GPR-KG":    "#d62728",
    "GPR-KG-nV": "#1f77b4",
    "cEHVI":     "#2ca02c",
    "cParEGO":   "#9467bd",
    "NSGA-II-K": "#ff7f0e",
    "NSGA-II-D": "#8c564b",
    "RS":        "#7f7f7f",
}
METHOD_MARKERS = {
    "GPR-KG":    "o",
    "GPR-KG-nV": "s",
    "cEHVI":     "^",
    "cParEGO":   "D",
    "NSGA-II-K": "v",
    "NSGA-II-D": "P",
    "RS":        "x",
}

PROBLEMS = {
    "RZDT1": {
        "dir": os.path.join(BASE_DIR, "results", "d5_v2", "RZDT1"),
        "factory": lambda: _mk(RZDT1, d=5, sigma=0.04, tau=0.0),
    },
    "RZDT2": {
        "dir": os.path.join(BASE_DIR, "results", "d5_v2", "RZDT2"),
        "factory": lambda: _mk(RZDT2, d=5, sigma=0.04, tau=0.0),
    },
    "RZDT5_RR": {
        "dir": os.path.join(BASE_DIR, "results", "rzdt5rr", "RZDT5_RR"),
        "factory": lambda: _mk(RZDT5_RR, d=5, sigma=0.04, tau=0.0),
    },
}


def _mk(cls, d, sigma, tau):
    p = cls(d=d, sigma=sigma, heteroscedastic=True, alpha=0.05)
    p.tau = tau
    return p


def load_reps(prob_dir, method):
    files = sorted(glob.glob(os.path.join(prob_dir, f"{SAFE[method]}_rep*.json")))
    out = []
    for f in files:
        try:
            out.append(json.load(open(f)))
        except Exception:
            pass
    return out


def pareto_curve_and_points(prob):
    """Return (curve_x, curve_y) smooth Pareto front and (pts_f1, pts_f2) integer TPOS points."""
    # curve: from true_pareto_curve if available
    if hasattr(prob, "true_pareto_curve"):
        t, f2 = prob.true_pareto_curve()
        cx = np.array(t)
        cy = np.array(f2)
    else:
        cx = np.linspace(0, 1, 200)
        cy = np.full_like(cx, np.nan)
    # TPOS integer points
    lo, hi = prob.int_bounds()
    xs, ys = [], []
    for x1 in range(lo[0], hi[0] + 1):
        x = tuple([x1] + [0] * (prob.d - 1))
        if prob.is_truly_feasible(x):
            f1, f2, _ = prob.true_objectives(x)
            xs.append(f1)
            ys.append(f2)
    return cx, cy, np.array(xs), np.array(ys)


# ────────────────────────────────────────────────────────────
#  FIGURE 1: bi-objective scatter, one subplot per problem
# ────────────────────────────────────────────────────────────
def fig1():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    labels_abc = ["(a) RZDT1", "(b) RZDT2", "(c) RZDT5\\_RR"]
    prob_names = ["RZDT1", "RZDT2", "RZDT5_RR"]

    for idx, pname in enumerate(prob_names):
        ax = axes[idx]
        info = PROBLEMS[pname]
        prob = info["factory"]()
        cx, cy, tx, ty = pareto_curve_and_points(prob)

        # theoretical front curve
        # use f1-f2 curve on feasible range (filtered by true constraint on x1 grid)
        lo, hi = prob.int_bounds()
        feas_x1 = [x1 for x1 in range(lo[0], hi[0] + 1)
                   if prob.is_truly_feasible(tuple([x1] + [0] * (prob.d - 1)))]
        if feas_x1:
            # split into contiguous segments (RZDT1/2 have two tails)
            segs = []
            cur = [feas_x1[0]]
            for x1 in feas_x1[1:]:
                if x1 == cur[-1] + 1:
                    cur.append(x1)
                else:
                    segs.append(cur)
                    cur = [x1]
            segs.append(cur)
            for seg in segs:
                f1s, f2s = [], []
                for x1 in seg:
                    f1, f2, _ = prob.true_objectives(tuple([x1] + [0] * (prob.d - 1)))
                    f1s.append(f1); f2s.append(f2)
                ax.plot(f1s, f2s, "k-", linewidth=1.2, alpha=0.7,
                        label="Theoretical PF" if seg is segs[0] else None)

        # TPOS black dots
        ax.scatter(tx, ty, c="black", s=18, marker=".",
                   label=f"TPOS ({len(tx)})", zorder=3)

        # each method's final PF (across 10 reps)
        for m in METHODS:
            reps = load_reps(info["dir"], m)
            if not reps:
                continue
            all_pts = []
            for r in reps:
                pts = r.get("pareto_objectives_true", [])
                for p in pts:
                    all_pts.append(p)
            if not all_pts:
                continue
            arr = np.array(all_pts)
            ax.scatter(arr[:, 0], arr[:, 1],
                       c=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                       s=30, alpha=0.55, edgecolors="none",
                       label=m if idx == 0 else None)

        ax.set_xlabel(r"$f^1$")
        if idx == 0:
            ax.set_ylabel(r"$f^2$")
        ax.set_title(labels_abc[idx], fontsize=11)
        ax.grid(alpha=0.3)
        # axis limits
        ax.set_xlim(-0.02, 1.02)
        if pname == "RZDT5_RR":
            ax.set_ylim(-0.05, 1.1)
        else:
            ax.set_ylim(-0.05, 1.3)

    # shared legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    # dedupe
    seen = set(); H, L = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="center right", fontsize=8,
               bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0, 0.87, 1])
    for ext in ["png", "pdf"]:
        fp = os.path.join(OUT_DIR, f"fig1_scatter.{ext}")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        print(f"  saved {fp}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
#  FIGURE 2: HV convergence vs simulation budget
# ────────────────────────────────────────────────────────────
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    prob_names = ["RZDT1", "RZDT2", "RZDT5_RR"]
    labels_abc = ["(a) RZDT1", "(b) RZDT2", "(c) RZDT5\\_RR"]

    for idx, pname in enumerate(prob_names):
        ax = axes[idx]
        info = PROBLEMS[pname]
        for m in METHODS:
            reps = load_reps(info["dir"], m)
            if not reps:
                continue
            # each rep has hv_history = [(n_sim, HV), ...]
            # interpolate onto a common grid so we can average
            all_x, all_y = [], []
            for r in reps:
                hist = r.get("hv_history", [])
                if not hist:
                    continue
                xs = np.array([h[0] for h in hist], dtype=float)
                ys = np.array([h[1] for h in hist], dtype=float)
                all_x.append(xs); all_y.append(ys)
            if not all_x:
                continue
            # common grid: take the first run's grid and interpolate others onto it
            grid = all_x[0]
            stacked = []
            for xs, ys in zip(all_x, all_y):
                stacked.append(np.interp(grid, xs, ys))
            Y = np.vstack(stacked)
            mean = Y.mean(axis=0)
            se = Y.std(axis=0, ddof=1) / np.sqrt(Y.shape[0])
            ax.plot(grid, mean, color=METHOD_COLORS[m],
                    marker=METHOD_MARKERS[m], markersize=4.5,
                    linewidth=1.3, label=m if idx == 0 else None)
            ax.fill_between(grid, mean - se, mean + se,
                            color=METHOD_COLORS[m], alpha=0.12)

        # HV_true reference line
        if reps:
            hv_upper = reps[0].get("hv_upper", None)
            if hv_upper:
                ax.axhline(hv_upper, linestyle="--", color="black",
                           linewidth=0.9,
                           label="HV$^{*}$" if idx == 0 else None)

        ax.set_xlabel("# simulations")
        if idx == 0:
            ax.set_ylabel("HV")
        ax.set_title(labels_abc[idx], fontsize=11)
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set(); H, L = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="center right", fontsize=8,
               bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0, 0.87, 1])
    for ext in ["png", "pdf"]:
        fp = os.path.join(OUT_DIR, f"fig2_hv_convergence.{ext}")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        print(f"  saved {fp}")
    plt.close(fig)


# ────────────────────────────────────────────────────────────
#  FIGURE 5: VEPM ablation (GPR-KG vs GPR-KG-nV final PF)
# ────────────────────────────────────────────────────────────
def fig5():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    prob_names = ["RZDT1", "RZDT2", "RZDT5_RR"]
    labels_abc = ["(a) RZDT1", "(b) RZDT2", "(c) RZDT5\\_RR"]

    for idx, pname in enumerate(prob_names):
        ax = axes[idx]
        info = PROBLEMS[pname]
        prob = info["factory"]()
        lo, hi = prob.int_bounds()
        feas_x1 = [x1 for x1 in range(lo[0], hi[0] + 1)
                   if prob.is_truly_feasible(tuple([x1] + [0] * (prob.d - 1)))]
        if feas_x1:
            segs = []; cur = [feas_x1[0]]
            for x1 in feas_x1[1:]:
                if x1 == cur[-1] + 1: cur.append(x1)
                else: segs.append(cur); cur = [x1]
            segs.append(cur)
            for seg in segs:
                f1s, f2s = [], []
                for x1 in seg:
                    f1, f2, _ = prob.true_objectives(tuple([x1] + [0] * (prob.d - 1)))
                    f1s.append(f1); f2s.append(f2)
                ax.plot(f1s, f2s, "k-", linewidth=1.2, alpha=0.7,
                        label="Theoretical PF" if seg is segs[0] else None)

        # GPR-KG (red) vs GPR-KG-nV (blue)
        for m in ["GPR-KG", "GPR-KG-nV"]:
            reps = load_reps(info["dir"], m)
            all_pts = []
            for r in reps:
                all_pts += r.get("pareto_objectives_true", [])
            if not all_pts:
                continue
            arr = np.array(all_pts)
            ax.scatter(arr[:, 0], arr[:, 1],
                       c=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                       s=36, alpha=0.55, edgecolors="none",
                       label=m if idx == 0 else None)

        ax.set_xlabel(r"$f^1$")
        if idx == 0:
            ax.set_ylabel(r"$f^2$")
        ax.set_title(labels_abc[idx], fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        if pname == "RZDT5_RR":
            ax.set_ylim(-0.05, 1.1)
        else:
            ax.set_ylim(-0.05, 1.3)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set(); H, L = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            H.append(h); L.append(l); seen.add(l)
    fig.legend(H, L, loc="center right", fontsize=9,
               bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    for ext in ["png", "pdf"]:
        fp = os.path.join(OUT_DIR, f"fig5_vepm_ablation.{ext}")
        fig.savefig(fp, dpi=180, bbox_inches="tight")
        print(f"  saved {fp}")
    plt.close(fig)


def main():
    print("[fig1] Bi-objective scatter ...")
    fig1()
    print("[fig2] HV convergence ...")
    fig2()
    print("[fig5] VEPM ablation ...")
    fig5()
    print(f"\nDone. Figures at {OUT_DIR}/")


if __name__ == "__main__":
    main()
