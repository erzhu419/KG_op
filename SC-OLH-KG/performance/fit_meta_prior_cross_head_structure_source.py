"""A14 source calibration on 256 whole-function holdouts; no target replay."""
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
from performance.cross_head_structure_mixture import (CrossHeadStructureMixture,
    calibrate_source_archives, fit_source_components, forecast_sequence, predictive_scores)
from performance.run_meta_prior_mechanism_v2 import prepare_sources

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_cross_head_structure_mixture_a14_20261003"


def source_statistics(rows, fits, spec):
    alphas = {(f["seed"], f["archive_group"]): f for f in fits}
    fold_scores = []
    for row in rows:
        alpha = alphas[row["seed"], row["archive_group"]]["alpha_mean"]
        base = np.array([f["independent_log_score"] for f in row["forecasts"]])
        score = predictive_scores(row["forecasts"], alpha)
        fold_scores.append({**{k: row[k] for k in ("seed", "archive_group", "held_task")},
            "mean_independent_log_score": float(base.mean()), "mean_joint_log_score": float(score.mean()),
            "mean_gain": float(np.mean(score - base))})
    cohort = spec["source_cohort"]
    seeds = cohort["seeds"]
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(len(seeds), size=(20000, len(seeds)))
    groups = []
    for archive in cohort["archive_groups"]:
        gains = []
        for seed in seeds:
            rr = [r for r in fold_scores if (r["seed"], r["archive_group"]) == (seed, archive)]
            if sorted(r["held_task"] for r in rr) != list(range(cohort["tasks_per_archive"])):
                raise ValueError("incomplete A14 source archive")
            gains.append(np.mean([r["mean_gain"] for r in rr]))
        gains = np.array(gains)
        group_fits = [f for f in fits if f["archive_group"] == archive]
        groups.append({"archive_group": archive, "seed_archives": len(seeds),
            "mean_gain": float(gains.mean()),
            "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
            "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
            "mean_alpha": float(np.mean([f["alpha_mean"] for f in group_fits])),
            "mean_alpha_sd": float(np.mean([f["alpha_sd"] for f in group_fits])), "seed_gains": gains.tolist()})
    return fold_scores, groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_cross_head_structure_mixture_plan.json").read_text())
    frozen = json.loads((ROOT / spec["target_cohort"]["input_artifacts"] / "execution_plan.json").read_text())["base_protocol"]
    cohort = spec["source_cohort"]
    if frozen["source_tasks"] != cohort["tasks_per_archive"]:
        raise ValueError("A14 source task count changed")
    old = json.loads((ROOT / "paper_artifacts/meta_prior_source_holdout_calibration_a11_20261003/summary.json").read_text())
    controls = {(r["seed"], r["archive_group"], r["held_task"]): r for r in old["rows"] if r["arm"] == "risk_ard_mixture"}
    expected = {(s, a, t) for s in cohort["seeds"] for a in cohort["archive_groups"] for t in range(cohort["tasks_per_archive"])}
    if not old["complete"] or len(expected) != cohort["held_function_fits"] or set(controls) != expected:
        raise ValueError("A14 original source control cohort changed")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A14 source phase reached the stage CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "source_complete": False, "target_complete": False,
        "status": "source_holdout_in_progress", "folds": [], "max_original_source_score_error": 0.,
        "max_factorization_score_error": 0., "server_experiment_jobs": 0, "new_server_pulls": 0,
        "new_source_simulation_calls": 0, "new_target_simulation_calls": 0,
        "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0,
        "scope": "Source CV calibrates alpha; these same calibration scores are not an independent nested alpha evaluation."}
    rows = []
    stop = False
    try:
        for seed in cohort["seeds"]:
            for archive in cohort["archive_groups"]:
                archive_start = time.process_time()
                regime = "shared_structure" if archive == "shared" else "separate_head_structure"
                tasks, _ = prepare_sources(seed, regime, frozen)
                reference = np.random.default_rng(51000 + seed).uniform(-2., 2.,
                    (frozen["reference_bank_size"], frozen["psi_dim"]))
                for held in range(cohort["tasks_per_archive"]):
                    fold_start = archive_start if held == 0 else time.process_time()
                    full, structural, info = fit_source_components(tasks, held, reference,
                                                                  frozen["learned_active_dim_per_head"])
                    if any(np.any(p.mean[1:]) for p in full + structural):
                        raise ValueError("nonintercept source means changed")
                    task = tasks[held]
                    model = CrossHeadStructureMixture(structural, full, task["psi"], reference)
                    steps = range(cohort["forecast_steps"][0], cohort["forecast_steps"][1] + 1)
                    forecasts = forecast_sequence(model, range(cohort["forecast_steps"][1]),
                        lambda t, i: (task["response"][t], task["noise"][t]),
                        float(task["noise"][0, 2]), cohort["initial_held_function_points"])
                    control = controls[seed, archive, held]["forecasts"]
                    if [f["step"] for f in forecasts] != list(steps) or [f["step"] for f in control] != list(steps):
                        raise ValueError("A14 source forecast chronology changed")
                    base = np.array([f["independent_log_score"] for f in forecasts])
                    error = float(np.max(np.abs(base - [f["source"] for f in control])))
                    factor_error = float(np.max(np.abs(predictive_scores(forecasts, 0.) - base)))
                    if max(error, factor_error) > 1e-8:
                        raise ValueError(f"A14 source control changed: {error}, {factor_error}")
                    result["max_original_source_score_error"] = max(result["max_original_source_score_error"], error)
                    result["max_factorization_score_error"] = max(result["max_factorization_score_error"], factor_error)
                    rows.append({"seed": seed, "archive_group": archive, "held_task": held, "forecasts": forecasts})
                    result["folds"].append({"seed": seed, "archive_group": archive, **info,
                        "fold_cpu_seconds": time.process_time() - fold_start})
                    if len(rows) == 1:
                        seconds = time.process_time() - fold_start
                        forecast = (seconds * cohort["held_function_fits"] * spec["resource_forecast"]["source_fold_safety_factor"]
                                    + spec["resource_forecast"]["target_calibration_and_io_reserve_cpu_seconds"])
                        result["admission"] = {"first_complete_fold_cpu_seconds": seconds,
                            "forecast_stage_cpu_seconds": forecast, "cap_cpu_seconds": cap, "admitted": forecast <= cap}
                        print("ADMISSION " + json.dumps(result["admission"]), flush=True)
                        if forecast > cap:
                            result["status"] = "partial_resource_admission_stopped"
                            stop = True
                            break
                if stop:
                    break
                print(f"SOURCE {len(rows)}/{cohort['held_function_fits']} complete folds", flush=True)
            if stop:
                break
        if len(rows) == cohort["held_function_fits"]:
            fits = calibrate_source_archives(rows, 64)
            fold_scores, groups = source_statistics(rows, fits, spec)
            checks = []
            for nodes in (32, 96):
                alternative = calibrate_source_archives(rows, nodes)
                lookup = {(f["seed"], f["archive_group"]): f for f in alternative}
                original = {(f["seed"], f["archive_group"]): f for f in fits}
                change = max(float(np.max(np.abs(predictive_scores(r["forecasts"], original[r["seed"], r["archive_group"]]["alpha_mean"])
                    - predictive_scores(r["forecasts"], lookup[r["seed"], r["archive_group"]]["alpha_mean"])))) for r in rows)
                checks.append({"nodes": nodes, "max_alpha_change": max(abs(a["alpha_mean"] - b["alpha_mean"]) for a, b in zip(fits, alternative)),
                               "max_source_log_score_change": change})
            numerical = {"orders": [32, 64, 96], "tolerance": 1e-4, "checks": checks,
                "passed": max(c["max_source_log_score_change"] for c in checks) <= 1e-4}
            result.update(source_complete=True, calibrations=fits, fold_scores=fold_scores,
                source_groups=groups, source_quadrature=numerical,
                optimizer_converged_folds=sum(f["optimizer_converged"] for f in result["folds"]),
                status="source_complete_gate_pending")
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      rusage_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      peak_rss_mb=float(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                             if line.startswith('VmHWM:'))) / 1024)
        (ARTIFACTS / "source_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": rows}, allow_nan=False).encode()))
        (ARTIFACTS / "source_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("source_complete", "status", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
