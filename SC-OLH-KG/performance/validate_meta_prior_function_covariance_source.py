"""A17 source-only function covariance validation with outer seed exclusion."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.function_covariance_source import node_grid, outer_priors, predictive_scores, selected_function_forecasts
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.source_calibrated_adaptation import fit_source_calibration, prior_adjusted_log_score
from performance.source_holdout_calibration import fit_source_fold
from performance.target_risk_residual import predictive_group_score
from performance.run_meta_prior_cross_head_risk_escape import contrast

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_function_covariance_source_a17_20261003"


def profile_arrays(row, rho):
    data = row["scores_by_rho"][str(rho)]
    return np.array(data["relative_log_evidence"]), np.array(data["component_log_density"])


def calibrate(rows, nodes, rho):
    grid = node_grid(nodes)
    grouped = {}
    # One archive/prior block per tensor evaluation; retain only seed-node sums.
    for row in rows:
        key = row["arm"], row["seed"]
        evidence, predictive = profile_arrays(row, rho)
        scores = predictive_scores(evidence, predictive, grid[0]).sum(axis=0)
        grouped[key] = grouped.get(key, 0.) + scores
    return {a: outer_priors({s: value for (arm, s), value in grouped.items() if arm == a}, grid)
            for a in ("fixed_source", "risk_ard_mixture")}


def summarize_validation(result, spec, frozen):
    seeds, groups = spec["source_cohort"]["seeds"], spec["source_cohort"]["archive_groups"]
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
    summaries = []
    for g in groups:
        for a in ("fixed_source", "risk_ard_mixture"):
            metrics = {k: [np.mean([r[k] for r in result["rows"]
                if (r["seed"], r["archive_group"], r["arm"]) == (s, g, a)]) for s in seeds]
                for k in ("mean_gain", "mean_gain_vs_same_pool_ir")}
            summaries.append({"archive_group": g, "arm": a, **contrast(metrics["mean_gain"], indices),
                              "versus_same_pool_ir": contrast(metrics["mean_gain_vs_same_pool_ir"], indices)})
    calls = frozen["source_tasks"] * frozen["source_points"] * frozen["replicates"]
    result.update(complete=True, groups=summaries, source_forecasts=8192,
        optimizer_converged_folds=sum(f["optimizer_converged"] for f in result["folds"]),
        historical_source_information={"calls_per_archive": calls, "outer_training_archives": 30,
            "outer_training_calls": 30 * calls, "held_archive_calls": calls, "all_32_source_archive_calls": 32 * calls},
        eta_ard_source_refits=len(result["folds"]), full_eta_source_refits=len(result["folds"]))


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = json.loads((ROOT / "performance/meta_prior_function_covariance_source_plan.json").read_text())
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A17 whole-stage local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "folds": [],
        "server_jobs": 0, "server_pulls": 0, "new_simulation_samples": 0,
        "new_source_calls": 0, "new_target_calls": 0, "new_fg_source_selection_fits": 0,
        "target_paths_scored": 0, "kg_calls": 0, "new_searches": 0,
        "new_random_verifications": 0, "controls": {"max_original_i_error": 0., "max_saved_a9_ratio_error": 0.}}
    rows = []
    try:
        cohort = spec["source_cohort"]
        seeds, groups = cohort["seeds"], cohort["archive_groups"]
        source = json.loads((ROOT / spec["inputs"][2]).read_text())
        old14 = json.loads((ROOT / spec["inputs"][1]).read_text())
        frozen = json.loads((ROOT / spec["inputs"][3]).read_text())["base_protocol"]
        expected = {(s, g, h, a) for s in seeds for g in groups for h in range(8)
                    for a in ("fixed_source", "risk_ard_mixture")}
        old = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in source["rows"]}
        fg = {(f["seed"], f["archive_group"], f["held_task"]): f for f in old14["folds"]}
        if not source["complete"] or not old14["source_complete"] or len(source["rows"]) != len(expected) or set(old) != expected:
            raise ValueError("A17 source control cohort changed")
        if len(fg) != 256 or len(old14["folds"]) != 256:
            raise ValueError("A17 source selection cohort changed")
        for s in seeds:
            for g in groups:
                archive_start = time.process_time()
                tasks, _ = prepare_sources(s, "shared_structure" if g == "shared" else "separate_head_structure", frozen)
                reference = np.random.default_rng(51000 + s).uniform(-2., 2., (frozen["reference_bank_size"], frozen["psi_dim"]))
                for h in range(8):
                    fold_start = archive_start if h == 0 else time.process_time()
                    full, structural, info = fit_source_fold(tasks, h, reference)
                    if info["training_tasks"] != fg[s, g, h]["training_tasks"] or h in info["training_tasks"]:
                        raise ValueError("f/g selections and eta priors use different source tasks")
                    selected = fg[s, g, h]["fg_selected_indices"]
                    by_rho = {}
                    for rho in (16, 32, 48):
                        fs = selected_function_forecasts([structural, full], tasks[h], selected[0], rho)
                        gs = selected_function_forecasts([structural, full], tasks[h], selected[1], rho)
                        learned_old = old[s, g, h, "risk_ard_mixture"]["forecasts"]
                        errors = [abs(new["source"] - previous["source"]) for forecasts in (fs, gs)
                                  for new, previous in zip(forecasts, learned_old)]
                        result["controls"]["max_original_i_error"] = max(result["controls"]["max_original_i_error"], *errors)
                        if max(errors) > 1e-8:
                            raise ValueError("A17 reconstructed learned-source I forecast changed")
                        if [(f["step"], f["candidate"]) for f in fs] != [(f["step"], f["candidate"]) for f in learned_old]:
                            raise ValueError("A17 forecast chronology changed")
                        by_rho[str(rho)] = {
                            "relative_log_evidence": [[0., f["relative_log_evidence"], gg["relative_log_evidence"],
                                float(np.log1p(-o["null_posterior"]) - np.log(o["null_posterior"]))]
                                for f, gg, o in zip(fs, gs, learned_old)],
                            "component_log_density": [[o["source"], f["adapted"], gg["adapted"], o["adapted"]]
                                for f, gg, o in zip(fs, gs, learned_old)]}
                    rows.append({"seed": s, "archive_group": g, "held_task": h, "arm": "risk_ard_mixture",
                                 "scores_by_rho": by_rho})
                    task = tasks[h]
                    generic_old = old[s, g, h, "fixed_source"]["forecasts"]
                    for t, previous in zip(range(8, 24), generic_old):
                        score = predictive_group_score([full], task["psi"], np.arange(t), task["response"][:t, 2],
                            task["noise"][:t, 2], 0., t, task["response"][t:t + 1, 2], task["noise"][t:t + 1, 2])
                        error = abs(score - previous["source"])
                        result["controls"]["max_original_i_error"] = max(result["controls"]["max_original_i_error"], error)
                        if error > 1e-8:
                            raise ValueError("A17 reconstructed generic-source I forecast changed")
                    generic_data = {"relative_log_evidence": [[0., 0., 0., float(np.log1p(-o["null_posterior"]) - np.log(o["null_posterior"]))] for o in generic_old],
                                    "component_log_density": [[o["source"], o["source"], o["source"], o["adapted"]] for o in generic_old]}
                    rows.append({"seed": s, "archive_group": g, "held_task": h, "arm": "fixed_source",
                                 "scores_by_rho": {str(rho): generic_data for rho in (16, 32, 48)}})
                    result["folds"].append({"seed": s, "archive_group": g, **info,
                        "fold_cpu_seconds": time.process_time() - fold_start})
                    if len(result["folds"]) == 1:
                        first = time.process_time() - fold_start
                        projection = time.process_time() - start + first * 255 * 1.5 + 60.
                        result["admission"] = {"retained_first_complete_fold_cpu_seconds": first,
                            "projected_stage_cpu_seconds": projection, "calibration_cpu_reserve": 60.,
                            "safety_multiplier": 1.5, "cap_cpu_seconds": cap, "admitted": projection < cap}
                        print(json.dumps({"admission": result["admission"]}), flush=True)
                        if projection >= cap:
                            raise TimeoutError("full-source projection exceeds A17 CPU cap")
                print(json.dumps({"source_folds_completed": len(result["folds"])}), flush=True)
        original_records = [f for row in source["rows"] for f in row["forecasts"]]
        result["controls"]["max_saved_a9_ratio_error"] = max(abs(float(prior_adjusted_log_score(
            f["source"], f["adapted"], f["null_posterior"], .5)) - f["a9"]) for f in original_records)
        if result["controls"]["max_saved_a9_ratio_error"] > 1e-8:
            raise ValueError("saved R/I evidence does not reconstruct A9")
        fits = calibrate(rows, 16, 32)
        result["calibrations"] = [{"arm": arm, **fit} for arm, ff in fits.items() for fit in ff.values()]
        baselines = {}
        for arm in ("fixed_source", "risk_ard_mixture"):
            for held in seeds:
                forecasts = [f for r in source["rows"] if r["arm"] == arm and r["seed"] != held for f in r["forecasts"]]
                baselines[arm, held] = fit_source_calibration(forecasts, 64)
        scores = {}
        validation = []
        for row in rows:
            key = row["seed"], row["archive_group"], row["held_task"], row["arm"]
            evidence, predictive = profile_arrays(row, 32)
            prior = fits[row["arm"]][row["seed"]]["macro_prior_mass"]
            new = predictive_scores(evidence, predictive, prior)
            previous = old[key]["forecasts"]
            baseline = np.array([prior_adjusted_log_score(f["source"], f["adapted"], f["null_posterior"],
                                                       baselines[row["arm"], row["seed"]]["q_mean"]) for f in previous])
            scores[key] = new
            validation.append({**{k: row[k] for k in ("seed", "archive_group", "held_task", "arm")},
                "log_scores": new.tolist(), "same_pool_ir_scores": baseline.tolist(),
                "mean_gain": float(np.mean(new - predictive[:, 0])),
                "mean_gain_vs_same_pool_ir": float(np.mean(new - baseline))})
        checks = []
        for n, rho in ((24, 32), (32, 32), (16, 16), (16, 48)):
            alternate = calibrate(rows, n, rho)
            error = 0.
            for row in rows:
                key = row["seed"], row["archive_group"], row["held_task"], row["arm"]
                e, p = profile_arrays(row, rho)
                alternative_score = predictive_scores(e, p, alternate[row["arm"]][row["seed"]]["macro_prior_mass"])
                error = max(error, float(np.max(np.abs(alternative_score - scores[key]))))
            checks.append({"macro_nodes_per_axis": n, "rho_nodes": rho, "max_source_log_score_change": error})
            print(json.dumps(checks[-1]), flush=True)
        generic_error = max(abs(fits["fixed_source"][s]["q_mean"] - baselines["fixed_source", s]["q_mean"]) for s in seeds)
        result["controls"]["max_generic_same_pool_a12_q_error"] = generic_error
        passed = max(c["max_source_log_score_change"] for c in checks) <= 1e-4 and generic_error <= 1e-8
        result.update(numerical_check={"checks": checks, "tolerance": 1e-4, "passed": passed}, rows=validation)
        if not passed:
            raise ValueError("A17 quadrature or generic calibration check failed")
        summarize_validation(result, spec, frozen)
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in
                Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "source_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": rows}, allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()
