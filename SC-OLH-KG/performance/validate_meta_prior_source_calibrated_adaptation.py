"""A12 source-calibrated model odds, scored on retained target paths."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.source_calibrated_adaptation import calibrate_existing_paths, prior_adjusted_log_score

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_source_calibrated_adaptation_a12_20261003"


def summarize(rows, spec):
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
    groups = []
    for regime in spec["target_cohort"]["regimes"]:
        for arm in spec["target_cohort"]["priors"]:
            rr = sorted((r for r in rows if (r["regime"], r["arm"]) == (regime, arm)), key=lambda r: r["seed"])
            gains = np.array([r["mean_calibrated_minus_source"] for r in rr])
            vs_a9 = np.array([r["mean_calibrated_minus_a9"] for r in rr])
            groups.append({"regime": regime, "arm": arm, "paths": len(rr), "mean_gain": float(gains.mean()),
                "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                "mean_gain_vs_a9": float(vs_a9.mean()),
                "vs_a9_paired_seed_bootstrap95": np.quantile(vs_a9[indices].mean(axis=1), [.025, .975]).tolist(),
                "mean_source_q": float(np.mean([r["forecasts"][0]["q_source"] for r in rr])),
                "mean_target_adaptation_mass": float(np.mean([f["target_adaptation_mass"] for r in rr for f in r["forecasts"]])),
                "seed_gains": gains.tolist()})
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    spec = json.loads((ROOT / "performance/meta_prior_source_calibrated_adaptation_plan.json").read_text())
    source = json.loads((ROOT / spec["source_scores"]).read_text())
    target = json.loads((ROOT / spec["target_scores"]).read_text())
    if not source["complete"] or not target["complete"]:
        raise ValueError("A12 requires complete source and target score artifacts")
    cohort = spec["target_cohort"]
    expected_target = {(s, r, a) for s in cohort["seeds"] for r in cohort["regimes"] for a in cohort["priors"]}
    expected_source = {(s, g, h, a) for s in cohort["seeds"] for g in ("shared", "separate_heads")
                       for h in range(8) for a in cohort["priors"]}
    if len(source["rows"]) != 512 or {(r["seed"], r["archive_group"], r["held_task"], r["arm"]) for r in source["rows"]} != expected_source:
        raise ValueError("A12 source score cohort changed")
    if len(target["rows"]) != spec["target_paths"] or {(r["seed"], r["regime"], r["arm"]) for r in target["rows"]} != expected_target:
        raise ValueError("A12 target score cohort changed")
    if any([f["step"] for f in r["forecasts"]] != list(range(9, 25)) for r in source["rows"] + target["rows"]):
        raise ValueError("A12 chronological forecast steps changed")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    cap = int(spec["limits"]["local_cpu_hours_cap"] * 3600)
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A12 reached its local CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    controls = [f for r in source["rows"] for f in r["forecasts"]] + [
        {**f, "null_posterior": f["a9_null_posterior"]} for r in target["rows"] for f in r["forecasts"]]
    max_error = max(abs(float(prior_adjusted_log_score(f["source"], f["adapted"], f["null_posterior"], .5)) - f["a9"]) for f in controls)
    if max_error > 1e-9:
        raise ValueError("A12 prior-odds formula does not reproduce A9")
    fits, rows = calibrate_existing_paths(source["rows"], target["rows"], 64)
    checks = []
    for nodes in (128, 192):
        alternative_fits, alternative_rows = calibrate_existing_paths(source["rows"], target["rows"], nodes)
        checks.append({"quadrature_nodes": nodes,
            "max_abs_q_change": max(abs(a["q_mean"] - b["q_mean"]) for a, b in zip(fits, alternative_fits)),
            "max_abs_target_log_score_change": max(abs(a["calibrated_log_score"] - b["calibrated_log_score"])
                for r, s in zip(rows, alternative_rows) for a, b in zip(r["forecasts"], s["forecasts"]))})
    numerical = {"orders": [64, 128, 192], "tolerance": 1e-4, "checks": checks,
        "passed": max(c["max_abs_target_log_score_change"] for c in checks) <= 1e-4}
    (ARTIFACTS / "quadrature_check.json").write_text(json.dumps(numerical, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": True, "calibrations": fits, "rows": rows,
        "groups": summarize(rows, spec), "max_a9_formula_error": max_error, "numerical_check_passed": numerical["passed"],
        "server_experiment_jobs": 0, "new_server_pulls": 0, "new_source_fits": 0, "new_source_calls": 0,
        "new_target_calls": 0, "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0,
        "cpu_seconds": time.process_time() - cpu_start, "wall_seconds": time.monotonic() - start,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "wall_seconds", "peak_rss_mb", "max_a9_formula_error", "numerical_check_passed")}), flush=True)


if __name__ == "__main__":
    main()
