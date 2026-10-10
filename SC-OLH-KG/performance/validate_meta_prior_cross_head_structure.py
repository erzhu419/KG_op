"""A14 chronological validation on the 160 frozen target query paths."""
import gzip
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
from scipy.special import polygamma

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.cross_head_structure_mixture import (CrossHeadStructureMixture,
    calibrate_source_archives, forecast_sequence, predictive_scores)
from performance.replay_meta_prior_calibration import restore_context
from performance.run_meta_prior_mechanism import observe

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_cross_head_structure_mixture_a14_20261003"


def summarize(rows, spec):
    cohort = spec["target_cohort"]
    seeds = cohort["seeds"]
    indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(len(seeds), size=(20000, len(seeds)))
    groups = []
    for regime in cohort["regimes"]:
        for arm in cohort["priors"]:
            rr = sorted((r for r in rows if (r["regime"], r["arm"]) == (regime, arm)), key=lambda r: r["seed"])
            if [r["seed"] for r in rr] != seeds:
                raise ValueError("A14 target paired-seed group incomplete")
            gains = np.array([r["mean_gain"] for r in rr])
            groups.append({"regime": regime, "arm": arm, "paths": len(rr), "mean_gain": float(gains.mean()),
                "paired_seed_bootstrap95": np.quantile(gains[indices].mean(axis=1), [.025, .975]).tolist(),
                "better": int(np.sum(gains > 1e-12)), "worse": int(np.sum(gains < -1e-12)),
                "seed_gains": gains.tolist()})
    return groups


def main():
    start, cpu_start = time.monotonic(), time.process_time()
    memory_start = {"rusage_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                    "vm_hwm_mb": float(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                            if line.startswith('VmHWM:'))) / 1024}
    print("RESOURCE_START " + json.dumps(memory_start), flush=True)
    spec = json.loads((ARTIFACTS / "execution_plan.json").read_text())
    source = json.loads((ARTIFACTS / "source_summary.json").read_text())
    gate = json.loads((ARTIFACTS / "source_gate.json").read_text())
    if not source["source_complete"] or not source["source_quadrature"]["passed"] or not gate["passed"]:
        raise ValueError("A14 target replay requires completed source calibration and a passed source gate")
    cohort = spec["target_cohort"]
    inputs = ROOT / cohort["input_artifacts"]
    frozen = json.loads((inputs / "execution_plan.json").read_text())["base_protocol"]
    protocol = {**frozen, **frozen["allocations"][cohort["allocation"]]}
    records = json.loads(gzip.decompress((inputs / "inputs.json.gz").read_bytes()))["records"]
    expected = {(s, r, a) for s in cohort["seeds"] for r in cohort["regimes"] for a in cohort["priors"]}
    key = lambda r: (r["seed"], r["regime"], r["arm"])
    if (len(records) != cohort["paths"] or {key(r["row"]) for r in records} != expected or
            any(r["row"]["difficulty"] != cohort["difficulty"] or r["row"]["allocation"] != cohort["allocation"] for r in records)):
        raise ValueError("A14 target cohort changed")
    old = json.loads((inputs / "summary.json").read_text())
    controls = {key(r): r for r in old["rows"] if r["update"] == "sample_variance_plugin"}
    original_scores = json.loads((ROOT / "paper_artifacts/meta_prior_risk_adaptation_calibration_a10_20261003/validation.json").read_text())
    scores = {key(r): r for r in original_scores["rows"]}
    if not old["old_reproduction_passed"] or not original_scores["complete"] or set(controls) != expected or set(scores) != expected:
        raise ValueError("A14 original target controls incomplete")
    source_rows = json.loads(gzip.decompress((ARTIFACTS / "source_forecasts.json.gz").read_bytes()))["rows"]
    alpha_fits = {(f["seed"], f["archive_group"]): f for f in source["calibrations"]}
    alternative_fits = {n: {(f["seed"], f["archive_group"]): f for f in calibrate_source_archives(source_rows, n)}
                        for n in (32, 96)}
    records.sort(key=lambda r: (r["row"]["seed"], cohort["regimes"].index(r["row"]["regime"]),
                                r["row"]["arm"] != "risk_ard_mixture"))
    remaining = int(spec["limits"]["local_cpu_hours_cap"] * 3600 - source["cpu_seconds"])
    if remaining <= 0:
        raise TimeoutError("A14 has no stage CPU budget remaining")
    cap = int(time.process_time()) + remaining
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def cpu_limit(signum, frame):
        raise TimeoutError("A14 target phase reached the remaining stage CPU budget")
    signal.signal(signal.SIGXCPU, cpu_limit)
    result = {"stage": spec["stage"], "complete": False, "rows": [], "memory_at_start": memory_start,
        "max_original_target_score_error": 0., "max_original_posterior_error": 0.,
        "max_factorization_score_error": 0., "quadrature_score_changes": {32: 0., 96: 0.},
        "server_experiment_jobs": 0, "new_server_pulls": 0, "new_source_model_selection": 0,
        "new_source_simulation_calls": 0, "new_target_simulation_calls": 0,
        "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0}
    context_key, context = None, None
    try:
        for count, record in enumerate(records, 1):
            row = record["row"]
            current = row["seed"], row["regime"]
            if current != context_key:
                context_key, context = current, restore_context(record, protocol)
            psi, reference, full, structural, truth, target_seed = context
            if len(row["queries"]) != protocol["target_points"] or row["queries"][:8] != row["initial_design"]:
                raise ValueError("A14 old query path changed")
            model = CrossHeadStructureMixture(structural if row["arm"] == "risk_ard_mixture" else full,
                                             full, psi, reference)
            visits = np.zeros(len(psi), dtype=int)
            def observation(t, index):
                response, noise = observe(truth[index], target_seed, index, int(visits[index]), protocol["replicates"])
                visits[index] += 1
                return response, noise
            forecasts = forecast_sequence(model, row["queries"], observation,
                float(polygamma(1, (protocol["replicates"] - 1) / 2.)), protocol["initial_points"])
            old_scores = scores[key(row)]["forecasts"]
            if [(f["step"], f["candidate"]) for f in forecasts] != [(f["step"], f["candidate"]) for f in old_scores]:
                raise ValueError("A14 original target forecast chronology changed")
            base = np.array([f["independent_log_score"] for f in forecasts])
            score_error = float(np.max(np.abs(base - [f["source"] for f in old_scores])))
            factor_error = float(np.max(np.abs(predictive_scores(forecasts, 0.) - base)))
            mean, var, weights = model.marginal_moments(np.array([row["recommendation"]]), 0.)
            observed = mean + var
            wanted = controls[key(row)]["posterior_mean"] + controls[key(row)]["posterior_variance"]
            if row["arm"] == "risk_ard_mixture":
                observed += weights
                wanted += [m["structural_weight"] for m in row["final_model_state"]]
            posterior_error = float(np.max(np.abs(np.array(observed) - wanted)))
            if max(score_error, posterior_error, factor_error) > 1e-8:
                raise ValueError(f"A14 target control changed: {score_error}, {posterior_error}, {factor_error}")
            for name, value in (("max_original_target_score_error", score_error),
                                ("max_original_posterior_error", posterior_error),
                                ("max_factorization_score_error", factor_error)):
                result[name] = max(result[name], value)
            archive = "separate_heads" if row["regime"] == "separate_head_structure" else "shared"
            alpha = alpha_fits[row["seed"], archive]["alpha_mean"]
            joint_scores = predictive_scores(forecasts, alpha)
            for n, fits in alternative_fits.items():
                alternative = predictive_scores(forecasts, fits[row["seed"], archive]["alpha_mean"])
                result["quadrature_score_changes"][n] = max(result["quadrature_score_changes"][n], float(np.max(np.abs(alternative - joint_scores))))
            for f, value in zip(forecasts, joint_scores):
                f["joint_log_score"] = float(value)
            result["rows"].append({**{k: row[k] for k in ("seed", "regime", "arm")}, "alpha_source": alpha,
                "forecasts": forecasts, "mean_gain": float(np.mean(joint_scores - base))})
            if count % 20 == 0:
                print(f"TARGET {count}/{cohort['paths']} complete paths", flush=True)
        result.update(complete=True, groups=summarize(result["rows"], spec),
            numerical_check_passed=max(result["quadrature_score_changes"].values()) <= 1e-4,
            status="chronological_validation_complete_decision_pending")
    finally:
        result.update(cpu_seconds=time.process_time() - cpu_start, wall_seconds=time.monotonic() - start,
                      rusage_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                      peak_rss_mb=float(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines()
                                             if line.startswith('VmHWM:'))) / 1024)
        result["stage_cpu_seconds"] = source["cpu_seconds"] + result["cpu_seconds"]
        (ARTIFACTS / "target_validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("complete", "numerical_check_passed", "cpu_seconds", "stage_cpu_seconds", "peak_rss_mb")}), flush=True)


if __name__ == "__main__":
    main()
