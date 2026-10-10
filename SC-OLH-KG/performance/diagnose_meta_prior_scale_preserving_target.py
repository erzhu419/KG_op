"""A22 frozen score accounting: sharing, joint q, and source-pool calibration."""
import json
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from performance.function_covariance_attribution import decompose
from performance.run_meta_prior_cross_head_risk_escape import contrast
from performance.validate_meta_prior_scale_preserving_source_covariance import read

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_scale_preserving_target_attribution_a22_20261003"


def key(row):
    return row["seed"], row["regime"], row["arm"]


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_scale_preserving_target_attribution_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A22 cached attribution CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "rows": [],
        "new_model_fits": 0, "new_predictive_forecasts": 0, "new_simulation_samples": 0,
        "server_jobs": 0, "server_pulls": 0, "kg_calls": 0, "new_searches": 0,
        "candidate_recommendations": 0, "new_random_verifications": 0}
    try:
        target, source, legacy = [read(p) for p in spec["inputs"]]
        if not all(d["complete"] for d in (target, source, legacy)):
            raise ValueError("A22 requires complete frozen evidence")
        target_spec = read("paper_artifacts/meta_prior_scale_preserving_target_validation_a21_20261003/execution_plan.json")
        cohort = target_spec["cohort"]
        expected = {(s, r, a) for s in cohort["seeds"] for r in cohort["regimes"] for a in cohort["priors"]}
        old = {key(r): r for r in legacy["rows"]}
        if len(target["rows"]) != 160 or len(old) != 160 or set(old) != expected or {key(r) for r in target["rows"]} != expected:
            raise ValueError("A22 target cohort changed")
        fits = {(f["arm"], f["held_seed"]): f for f in source["calibrations"]}
        baselines = {(f["arm"], f["held_seed"]): f for f in target["same_pool_ir_calibrations"]}
        errors = {"max_saved_score_error": 0., "max_total_accounting_error": 0.,
                  "max_head_accounting_error": 0., "max_generic_fg_effect": 0.}
        for row in target["rows"]:
            seed, regime, arm = key(row)
            forecasts = row["forecasts"]
            previous = old[key(row)]["forecasts"]
            if [(f["step"], f["candidate"]) for f in forecasts] != [(f["step"], f["candidate"]) for f in previous] or [f["step"] for f in forecasts] != list(range(9, 25)):
                raise ValueError("A22 forecast identity changed")
            prior = np.asarray(row["source_macro_prior_mass"])
            if prior.tolist() != fits[arm, seed]["macro_prior_mass"]:
                raise ValueError("A22 issued prior differs from frozen A20")
            evidence = np.array([f["relative_past_log_evidence"] for f in forecasts])
            predictive = np.array([f["component_log_density"] for f in forecasts])
            saved_scores = np.array([f["log_score"] for f in forecasts])
            ir_scores = np.array([f["same_pool_ir_score"] for f in forecasts])
            legacy_scores = np.array([f["calibrated_log_score"] for f in previous])
            accounting = decompose(evidence, predictive, prior, ir_scores)
            pool_effect = ir_scores - legacy_scores
            total_effect = accounting["fg_effect"] + accounting["q_effect"] + pool_effect
            errors["max_saved_score_error"] = max(errors["max_saved_score_error"], float(np.max(np.abs(accounting["log_score"] - saved_scores))))
            errors["max_total_accounting_error"] = max(errors["max_total_accounting_error"], float(np.max(np.abs(total_effect - (saved_scores - legacy_scores)))))
            errors["max_head_accounting_error"] = max(errors["max_head_accounting_error"], float(np.max(np.abs(accounting["f_accounting_effect"] + accounting["g_accounting_effect"] - accounting["fg_effect"]))))
            if arm == "fixed_source":
                errors["max_generic_fg_effect"] = max(errors["max_generic_fg_effect"], float(np.max(np.abs(accounting["fg_effect"]))))
            result["rows"].append({"seed": seed, "regime": regime, "arm": arm,
                **{name: float(np.mean(accounting[name])) for name in ("fg_effect", "q_effect", "f_accounting_effect", "g_accounting_effect", "pure_f_gain", "pure_g_gain", "pure_r_gain", "same_q_ir_gain")},
                "source_pool_calibration_effect": float(np.mean(pool_effect)), "total_vs_a12": float(np.mean(total_effect)),
                "issued_a20_q": float(prior[3]), "same_pool_ir_q": baselines[arm, seed]["q_mean"],
                "a12_per_archive_q": previous[0]["q_source"],
                "step9_posterior_mass": accounting["posterior_macro_mass"][0].tolist(),
                "step24_posterior_mass": accounting["posterior_macro_mass"][-1].tolist()})
        result["controls"] = errors
        if max(errors.values()) > 1e-12:
            raise ValueError("A22 frozen attribution identity failed")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
        names = ("fg_effect", "q_effect", "source_pool_calibration_effect", "total_vs_a12",
                 "f_accounting_effect", "g_accounting_effect", "pure_f_gain", "pure_g_gain", "pure_r_gain", "same_q_ir_gain")
        groups = []
        for regime in cohort["regimes"]:
            for arm in cohort["priors"]:
                rows = sorted((r for r in result["rows"] if (r["regime"], r["arm"]) == (regime, arm)), key=lambda r: r["seed"])
                groups.append({"regime": regime, "arm": arm,
                    **{name: contrast([r[name] for r in rows], indices) for name in names},
                    "mean_q": {name: float(np.mean([r[name] for r in rows])) for name in ("issued_a20_q", "same_pool_ir_q", "a12_per_archive_q")}})
        result.update(complete=True, groups=groups, target_paths=160, saved_forecasts_accounted=2560)
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "attribution.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()
