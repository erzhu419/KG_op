"""
Build extended per-problem summary tables for RZDT1, RZDT2, RZDT5_RR.

Reads per-rep JSON results from:
  - results/d5_v2/RZDT1/  (RZDT1)
  - results/d5_v2/RZDT2/  (RZDT2)
  - results/rzdt5rr/RZDT5_RR/  (RZDT5_RR)

Computes per rep for each method:
  - hv_final, igd_final, cvr_final, hv_ratio (already in JSON)
  - n_nondom: number of reported non-dominated solutions (n_pareto_solutions)
  - n_infeasible: how many of those are ACTUALLY infeasible under true constraint
  - n_tpos_found: how many match theoretical Pareto-optimal solutions (x_j=0 j>=2
                  AND x_1 in feasible Pareto range for that problem)

Aggregates across 10 reps (mean ± SE). Saves:
  - results/extended_tables/<PROB>_table.json   (structured)
  - results/extended_tables/<PROB>_table.csv    (for paste)
  - results/extended_tables/combined.json       (all three, single file)

Usage:
    python -m experiments.build_extended_tables
"""
import os, sys, json, glob
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from gpr_kg import RZDT1, RZDT2, RZDT5_RR

OUT_DIR = os.path.join(BASE_DIR, "results", "extended_tables")
os.makedirs(OUT_DIR, exist_ok=True)

METHODS = ["GPR-KG", "GPR-KG-nV", "cEHVI", "cParEGO",
           "NSGA-II-K", "NSGA-II-D", "RS"]
METHOD_SAFE = {m: m.replace("-", "_") for m in METHODS}

PROBLEMS = {
    "RZDT1": {
        "dir": os.path.join(BASE_DIR, "results", "d5_v2", "RZDT1"),
        "factory": lambda: _mk(RZDT1, d=5, sigma=0.04, tau=0.0),
        "n_tpos": 21,
        "tau": 0.0, "sigma": 0.04,
    },
    "RZDT2": {
        "dir": os.path.join(BASE_DIR, "results", "d5_v2", "RZDT2"),
        "factory": lambda: _mk(RZDT2, d=5, sigma=0.04, tau=0.0),
        "n_tpos": 34,
        "tau": 0.0, "sigma": 0.04,
    },
    "RZDT5_RR": {
        "dir": os.path.join(BASE_DIR, "results", "rzdt5rr", "RZDT5_RR"),
        "factory": lambda: _mk(RZDT5_RR, d=5, sigma=0.04, tau=0.0),
        "n_tpos": 46,
        "tau": 0.0, "sigma": 0.04,
    },
}


def _mk(cls, d, sigma, tau):
    p = cls(d=d, sigma=sigma, heteroscedastic=True, alpha=0.05)
    p.tau = tau
    return p


def is_tpos(x, prob, tpos_x1_set):
    """Check if x is a TPOS: x_j=0 for j>=2 AND x_1 in TPOS range."""
    x = tuple(int(v) for v in x)
    if any(x[j] != 0 for j in range(1, prob.d)):
        return False
    return x[0] in tpos_x1_set


def compute_tpos_set(prob):
    """Return set of x_1 values that are TPOS."""
    lo, hi = prob.int_bounds()
    tpos_x1 = set()
    for x1 in range(lo[0], hi[0] + 1):
        x = tuple([x1] + [0] * (prob.d - 1))
        if prob.is_truly_feasible(x):
            tpos_x1.add(x1)
    return tpos_x1


def is_infeasible_true(x, prob):
    """A solution is INFEASIBLE (under chance constraint) if the true quantile
    exceeds tau.  Matches is_truly_feasible logic."""
    return not prob.is_truly_feasible(tuple(int(v) for v in x))


def process_rep(rep_json, prob, tpos_x1_set):
    """Return dict of per-rep metrics."""
    sols = rep_json.get("pareto_solutions", [])
    n_nd = len(sols)
    n_inf = 0
    n_tpos = 0
    for x in sols:
        if is_infeasible_true(x, prob):
            n_inf += 1
        if is_tpos(x, prob, tpos_x1_set):
            n_tpos += 1
    return {
        "rep": rep_json.get("rep"),
        "hv_final": rep_json.get("hv_final"),
        "igd_final": rep_json.get("igd_final"),
        "cvr_final": rep_json.get("cvr_final"),
        "hv_ratio": rep_json.get("hv_ratio"),
        "wall_time_sec": rep_json.get("wall_time_sec"),
        "n_nondom": n_nd,
        "n_infeasible": n_inf,
        "n_tpos_found": n_tpos,
    }


def mean_se(vals):
    arr = np.asarray(vals, dtype=float)
    n = len(arr)
    if n == 0:
        return (None, None)
    m = float(np.nanmean(arr))
    if n < 2:
        return (m, None)
    se = float(np.nanstd(arr, ddof=1) / np.sqrt(n))
    return (m, se)


def aggregate(prob_name, info):
    prob = info["factory"]()
    tpos_x1_set = compute_tpos_set(prob)
    assert len(tpos_x1_set) == info["n_tpos"], \
        f"{prob_name}: expected {info['n_tpos']} TPOS, got {len(tpos_x1_set)}"

    method_summary = {}
    for m in METHODS:
        safe = METHOD_SAFE[m]
        files = sorted(glob.glob(os.path.join(info["dir"], f"{safe}_rep*.json")))
        if not files:
            print(f"  [warn] {prob_name}/{m}: no rep files found")
            continue
        reps = []
        for f in files:
            try:
                rep = json.load(open(f))
                reps.append(process_rep(rep, prob, tpos_x1_set))
            except Exception as e:
                print(f"  [warn] {f}: {e}")
        if not reps:
            continue

        agg = {"n_reps": len(reps)}
        for k in ["hv_final", "igd_final", "cvr_final", "hv_ratio",
                  "n_nondom", "n_infeasible", "n_tpos_found", "wall_time_sec"]:
            m_, se = mean_se([r[k] for r in reps if r[k] is not None])
            agg[k + "_mean"] = m_
            agg[k + "_se"] = se
        agg["reps_raw"] = reps
        method_summary[m] = agg

    return {
        "problem": prob_name,
        "n_tpos_theoretical": info["n_tpos"],
        "tau": info["tau"],
        "sigma": info["sigma"],
        "d": prob.d,
        "N_FEs": 150,
        "n0": 30,
        "alpha": 0.05,
        "methods": method_summary,
    }


def write_csv(combined):
    for pname, entry in combined.items():
        csv_path = os.path.join(OUT_DIR, f"{pname}_table.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("Method,HV mean,HV SE,HV%,IGD mean,IGD SE,CVR mean,CVR SE,"
                    "NonDom mean,NonDom SE,Infeas mean,Infeas SE,"
                    "TPOS_found mean,TPOS_found SE,"
                    "#TPOS_theo,Time mean (s),Time SE (s)\n")
            for m, v in entry["methods"].items():
                hv = v["hv_final_mean"]
                hv_se = v["hv_final_se"]
                hv_pct = 100.0 * hv / 1.0  # placeholder; use hv_ratio
                hv_pct = 100.0 * (v["hv_ratio_mean"] or 0.0)
                f.write(f"{m},{hv:.4f},{hv_se or 0:.4f},{hv_pct:.1f},"
                        f"{v['igd_final_mean']:.4f},{v['igd_final_se'] or 0:.4f},"
                        f"{v['cvr_final_mean']:.4f},{v['cvr_final_se'] or 0:.4f},"
                        f"{v['n_nondom_mean']:.2f},{v['n_nondom_se'] or 0:.2f},"
                        f"{v['n_infeasible_mean']:.2f},{v['n_infeasible_se'] or 0:.2f},"
                        f"{v['n_tpos_found_mean']:.2f},{v['n_tpos_found_se'] or 0:.2f},"
                        f"{entry['n_tpos_theoretical']},"
                        f"{v['wall_time_sec_mean']:.1f},{v['wall_time_sec_se'] or 0:.1f}\n")
        print(f"  Saved {csv_path}")


def print_table(pname, entry):
    print("\n" + "=" * 105)
    print(f"  {pname}   (#TPOS_theo={entry['n_tpos_theoretical']}, "
          f"tau={entry['tau']}, sigma={entry['sigma']}, d={entry['d']}, "
          f"N_FEs={entry['N_FEs']})")
    print("=" * 105)
    header = (f"{'Method':12s} {'HV':>7s}±{'SE':>5s} {'HV%':>5s}  "
              f"{'IGD':>7s}±{'SE':>5s}  {'CVR':>5s}  "
              f"{'#ND':>5s}  {'#Inf':>5s}  {'#TPOS':>6s}  {'t(s)':>7s}")
    print(header)
    print("-" * 105)
    for m, v in entry["methods"].items():
        hv_pct = 100.0 * (v["hv_ratio_mean"] or 0.0)
        print(f"{m:12s} "
              f"{v['hv_final_mean']:>6.4f}±{v['hv_final_se'] or 0:>5.4f} "
              f"{hv_pct:>4.1f}%  "
              f"{v['igd_final_mean']:>6.4f}±{v['igd_final_se'] or 0:>5.4f}  "
              f"{v['cvr_final_mean']:>5.3f}  "
              f"{v['n_nondom_mean']:>5.2f}  "
              f"{v['n_infeasible_mean']:>5.2f}  "
              f"{v['n_tpos_found_mean']:>6.2f}  "
              f"{v['wall_time_sec_mean']:>7.1f}")


def main():
    combined = {}
    for pname, info in PROBLEMS.items():
        print(f"\n[+] Processing {pname} ...")
        combined[pname] = aggregate(pname, info)
        print_table(pname, combined[pname])

        # individual JSON
        jpath = os.path.join(OUT_DIR, f"{pname}_table.json")
        with open(jpath, "w") as f:
            json.dump(combined[pname], f, indent=2)
        print(f"  Saved {jpath}")

    # combined
    cpath = os.path.join(OUT_DIR, "combined.json")
    with open(cpath, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\n[done] combined.json saved to {cpath}")
    write_csv(combined)


if __name__ == "__main__":
    main()
