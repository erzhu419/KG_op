"""Confirm A17 primary quadrature from cached forecasts, without source fits."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from performance.validate_meta_prior_function_covariance_source import (
    ARTIFACTS, ROOT, calibrate, profile_arrays, summarize_validation,
)
from performance.function_covariance_source import predictive_scores


def main():
    start = time.process_time()
    wall_start = time.monotonic()
    result = json.loads((ARTIFACTS / "validation.json").read_text())
    spec = json.loads((ROOT / "performance/meta_prior_function_covariance_source_plan.json").read_text())
    initial = {k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb", "numerical_check", "controls", "failure")}
    (ARTIFACTS / "initial_quadrature_diagnostic.json").write_text(json.dumps(initial, indent=2) + "\n")
    remaining = int(spec["limits"]["local_cpu_seconds_cap"] - result["cpu_seconds"])
    resource.setrlimit(resource.RLIMIT_CPU, (remaining, remaining + 2))
    def stop(signum, frame):
        raise TimeoutError("A17 remaining CPU budget reached during quadrature confirmation")
    signal.signal(signal.SIGXCPU, stop)
    with gzip.open(ARTIFACTS / "source_forecasts.json.gz", "rt") as stream:
        rows = json.load(stream)["rows"]
    # The primary 16^3 fit stays frozen; only compare with a finer grid.
    primary = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in result["rows"]}
    alternate = calibrate(rows, 32, 32)
    error = 0.
    for row in rows:
        key = row["seed"], row["archive_group"], row["held_task"], row["arm"]
        e, p = profile_arrays(row, 32)
        score = predictive_scores(e, p, alternate[row["arm"]][row["seed"]]["macro_prior_mass"])
        error = max(error, float(np.max(np.abs(score - primary[key]["log_scores"]))))
    coarse = [c for c in result["numerical_check"]["checks"] if c["macro_nodes_per_axis"] == 8]
    checks = [c for c in result["numerical_check"]["checks"] if c["macro_nodes_per_axis"] != 8]
    checks.append({"macro_nodes_per_axis": 32, "rho_nodes": 32, "max_source_log_score_change": error})
    passed = max(c["max_source_log_score_change"] for c in checks) <= 1e-4 and max(result["controls"].values()) <= 1e-8
    if not passed:
        raise ValueError("higher-order quadrature does not confirm the frozen A17 primary result")
    result["numerical_check"] = {"primary_macro_order": 16, "accepted_macro_orders": [16, 24, 32],
        "rho_orders": [16, 32, 48], "checks": checks, "insufficient_coarse_checks": coarse,
        "tolerance": 1e-4, "passed": True, "primary_parameters_changed": False}
    frozen = json.loads((ROOT / spec["inputs"][3]).read_text())["base_protocol"]
    summarize_validation(result, spec, frozen)
    seconds = time.process_time() - start
    result["quadrature_confirmation_cpu_seconds"] = seconds
    result["cpu_seconds"] += seconds
    result["initial_run_wall_seconds"] = result.pop("wall_seconds")
    result["quadrature_confirmation_wall_seconds"] = time.monotonic() - wall_start
    result["wall_seconds"] = result["initial_run_wall_seconds"] + result["quadrature_confirmation_wall_seconds"]
    result["quadrature_confirmation_vm_hwm_mb"] = float(next(line.split()[1] for line in
        Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024
    result["initial_numerical_diagnostic"] = "initial_quadrature_diagnostic.json"
    result.pop("failure")
    (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "max_32_order_score_change": error,
                      "whole_stage_cpu_seconds": result["cpu_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
