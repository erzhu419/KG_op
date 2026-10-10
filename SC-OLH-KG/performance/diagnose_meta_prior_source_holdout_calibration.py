"""A11 source-only held-task chronological prediction on frozen archives."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.source_holdout_calibration import (fit_source_fold, forecast_held_source,
    summarize_source_holdout)

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_source_holdout_calibration_a11_20261003"


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_source_holdout_calibration_diagnostic_plan.json").read_text())
    cohort = spec["cohort"]
    frozen = json.loads((ROOT / spec["input_artifacts"] / "execution_plan.json").read_text())["base_protocol"]
    a9 = json.loads((ROOT / spec["public_family_plan"]).read_text())
    metadata = json.loads((ROOT / spec["source_metadata"]).read_text())
    expected = {(s, a) for s in cohort["seeds"] for a in cohort["source_archive_groups"]}
    actual = [(r["seed"], r["archive_group"]) for r in metadata["rows"]]
    if len(actual) != cohort["archives"] or set(actual) != expected:
        raise ValueError("A11 source archive cohort differs from its plan")
    if frozen["source_tasks"] != cohort["held_tasks_per_archive"]:
        raise ValueError("A11 source task count changed")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A11 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "rows": [], "folds": [],
        "server_experiment_jobs": 0, "new_server_pulls": 0, "kg_calls": 0, "new_searches": 0,
        "new_source_calls": 0, "new_target_calls": 0, "new_random_verifications": 0,
        "original_source_calls_per_archive": frozen["source_tasks"] * frozen["source_points"] * frozen["replicates"],
        "scope": "Source-only task holdout; old source labels and outcome-free source designs; no calibration rule fitted."}
    stopped = False
    try:
        for seed in cohort["seeds"]:
            for archive in cohort["source_archive_groups"]:
                archive_start = time.process_time()
                regime = "shared_structure" if archive == "shared" else "separate_head_structure"
                tasks, _ = prepare_sources(seed, regime, frozen)
                reference = np.random.default_rng(51000 + seed).uniform(-2., 2.,
                    (frozen["reference_bank_size"], frozen["psi_dim"]))
                if any(len(t["psi"]) != frozen["source_points"] for t in tasks):
                    raise ValueError("A11 historical source point count changed")
                for held in range(cohort["held_tasks_per_archive"]):
                    fold_start = archive_start if held == 0 else time.process_time()
                    full, structural, info = fit_source_fold(tasks, held, reference)
                    if info["source_label_count"] != (len(tasks) - 1) * frozen["source_points"]:
                        raise ValueError("A11 source fit retained the wrong number of labels")
                    priors_by_arm = {"fixed_source": [full], "risk_ard_mixture": [structural, full]}
                    for arm in cohort["priors"]:
                        forecasts = forecast_held_source(priors_by_arm[arm], tasks[held],
                            a9["public_families"], a9["quadrature_nodes"], cohort["forecast_steps"])
                        result["rows"].append({"seed": seed, "archive_group": archive, "held_task": held,
                            "arm": arm, "forecasts": forecasts,
                            **{name + "_mean_log_score": float(np.mean([f[name] for f in forecasts]))
                               for name in ("source", "adapted", "a9")}})
                    result["folds"].append({"seed": seed, "archive_group": archive, **info,
                        "fold_cpu_seconds": time.process_time() - fold_start})
                    if len(result["folds"]) == 1:
                        seconds = time.process_time() - fold_start
                        forecast = time.process_time() - cpu_start + seconds * (cohort["source_fold_fits"] - 1)
                        result["resource_admission"] = {"seed": seed, "archive_group": archive, "held_task": held,
                            "first_fold_cpu_seconds": seconds, "forecast_cpu_seconds": forecast,
                            "cap_cpu_seconds": cap, "admitted": forecast <= cap}
                        (ARTIFACTS / "resource_admission.json").write_text(json.dumps(result["resource_admission"], indent=2) + "\n")
                        print("ADMISSION " + json.dumps(result["resource_admission"]), flush=True)
                        if forecast > cap:
                            result["stop_reason"] = "First-fold forecast exceeds local CPU cap; preserve partial evidence."
                            stopped = True
                            break
                if stopped:
                    break
                print(f"SOURCE HELD OUT {len(result['folds'])}/{cohort['source_fold_fits']} folds", flush=True)
            if stopped:
                break
        if len(result["folds"]) == cohort["source_fold_fits"] and len(result["rows"]) == cohort["predictive_paths"]:
            result.update(summarize_source_holdout(result["rows"], spec), complete=True,
                optimizer_converged_folds=sum(f["optimizer_converged"] for f in result["folds"]))
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        (ARTIFACTS / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "wall_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
