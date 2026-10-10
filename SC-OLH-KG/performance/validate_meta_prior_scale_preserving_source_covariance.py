"""A20 source-only covariance validation using recovered seven-task contexts."""
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
from performance.function_covariance_source import predictive_scores
from performance.run_meta_prior_mechanism_v2 import prepare_sources
from performance.run_meta_prior_cross_head_risk_escape import contrast
from performance.scale_preserving_source_covariance import scale_preserving_covariance, selected_scale_forecasts
from performance.source_calibrated_adaptation import prior_adjusted_log_score
from performance.target_risk_residual import predictive_group_score
from performance.validate_meta_prior_function_covariance_source import calibrate, profile_arrays
from representation.source_function_prior import FixedFunctionBasis, SourceFunctionPrior, head_batches

ARTIFACTS = ROOT / "paper_artifacts/meta_prior_scale_preserving_source_covariance_a20_20261003"


def read(path):
    path = ROOT / path
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def covariance_controls(priors, selected):
    errors = {"max_trace_error": 0., "max_intercept_error": 0., "max_within_set_scale_error": 0.,
              "max_no_sharing_error": 0., "max_nonintercept_mean": 0.}
    for prior in priors:
        original = np.diag(prior.covariance)[1:]
        errors["max_nonintercept_mean"] = max(errors["max_nonintercept_mean"], float(np.max(np.abs(prior.mean[1:]))))
        for ids in selected.values():
            errors["max_no_sharing_error"] = max(errors["max_no_sharing_error"],
                float(np.max(np.abs(scale_preserving_covariance(prior, ids, 0.) - prior.covariance))))
            for rho in (.4, 1.):
                changed = scale_preserving_covariance(prior, ids, rho)
                errors["max_trace_error"] = max(errors["max_trace_error"], abs(float(np.trace(changed[1:, 1:]) - original.sum())))
                errors["max_intercept_error"] = max(errors["max_intercept_error"],
                    float(np.max(np.abs(changed[0] - prior.covariance[0]))))
                diagonal = np.diag(changed)[1:]
                if np.any(diagonal <= 0.):
                    raise ValueError("A20 erased an existing risk direction")
                mask = np.isin(np.arange(len(original)), ids)
                ratio = diagonal / original
                for group in (mask, ~mask):
                    if np.any(group):
                        errors["max_within_set_scale_error"] = max(errors["max_within_set_scale_error"], float(np.ptp(ratio[group])))
    if max(errors.values()) > 1e-12:
        raise ValueError("A20 covariance changed frozen scales, budget, intercept or mean")
    return errors


def main():
    start, wall = time.process_time(), time.monotonic()
    spec = read("performance/meta_prior_scale_preserving_source_covariance_plan.json")
    cap = spec["limits"]["local_cpu_seconds_cap"]
    resource.setrlimit(resource.RLIMIT_CPU, (cap, cap + 2))
    def stop(signum, frame):
        raise TimeoutError("A20 whole-stage local CPU cap reached")
    signal.signal(signal.SIGXCPU, stop)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "execution_plan.json").write_text(json.dumps(spec, indent=2) + "\n")
    result = {"stage": spec["stage"], "complete": False, "folds": [], "controls": {},
        "new_eta_ard_refits": 0, "new_fg_selection_fits": 0, "server_jobs": 0, "server_pulls": 0,
        "new_simulation_samples": 0, "new_source_calls": 0, "new_target_calls": 0,
        "target_paths_scored": 0, "kg_calls": 0, "new_searches": 0, "new_random_verifications": 0}
    rows = []
    try:
        cohort = spec["cohort"]
        seeds, groups = cohort["seeds"], cohort["archive_groups"]
        contexts = read(spec["inputs"][0])["rows"]
        geometry, old17 = read(spec["inputs"][1]), read(spec["inputs"][3])
        saved_rows = read(spec["inputs"][2])["rows"]
        original = read(spec["inputs"][5])
        frozen = read(spec["inputs"][4])["base_protocol"]
        expected_folds = {(s, g, h) for s in seeds for g in groups for h in range(8)}
        expected_profiles = {(*key, a) for key in expected_folds for a in ("fixed_source", "risk_ard_mixture")}
        context = {(r["seed"], r["archive_group"], r["held_task"]): r for r in contexts}
        old = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in saved_rows}
        validation17 = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in old17["rows"]}
        original_rows = {(r["seed"], r["archive_group"], r["held_task"], r["arm"]): r for r in original["rows"]}
        if not all(d["complete"] for d in (geometry, old17, original)):
            raise ValueError("A20 requires completed A11, A17 and A19 evidence")
        if len(contexts) != 256 or set(context) != expected_folds or any(
                len(items) != 512 or set(items) != expected_profiles for items in (old, validation17, original_rows)):
            raise ValueError("A20 source cohort changed")
        controls = {"max_original_i_error": 0., "max_saved_r_i_evidence_error": 0.}
        for seed in seeds:
            for group in groups:
                archive_start = time.process_time()
                tasks, _ = prepare_sources(seed, "shared_structure" if group == "shared" else "separate_head_structure", frozen)
                reference = np.random.default_rng(51000 + seed).uniform(-2., 2., (frozen["reference_bank_size"], frozen["psi_dim"]))
                basis = FixedFunctionBasis(reference)
                for held in range(8):
                    fold_start = archive_start if held == 0 else time.process_time()
                    saved = context[seed, group, held]
                    training_ids = [i for i in range(8) if i != held]
                    if saved["training_tasks"] != training_ids:
                        raise ValueError("A20 held task entered source prior conditioning")
                    training = [tasks[i] for i in training_ids]
                    full = SourceFunctionPrior.fit(head_batches(training, 2), [t["noise"][:, 2] for t in training], reference, "fixed_source")
                    structural = SourceFunctionPrior(basis, np.r_[saved["eta_intercept_mean"], np.zeros(basis.feature_dim)],
                        np.diag([.25] + saved["eta_prior_function_variance"]), saved["eta_discrepancy_variance"], {})
                    selected = saved["fg_selected_indices"]
                    for name, value in covariance_controls([structural, full], selected).items():
                        controls[name] = max(controls.get(name, 0.), value)
                    learned_key = seed, group, held, "risk_ard_mixture"
                    by_rho = {}
                    for rho in (16, 32, 48):
                        fs = selected_scale_forecasts([structural, full], tasks[held], selected["f"], rho)
                        gs = selected_scale_forecasts([structural, full], tasks[held], selected["g"], rho)
                        previous_e, previous_p = profile_arrays(old[learned_key], rho)
                        errors = [abs(f["source"] - p) for forecasts in (fs, gs) for f, p in zip(forecasts, previous_p[:, 0])]
                        controls["max_original_i_error"] = max(controls["max_original_i_error"], *errors)
                        if max(errors) > 1e-8 or [(f["step"], f["candidate"]) for f in fs] != [(t + 1, t) for t in range(8, 24)]:
                            raise ValueError("A20 reconstructed I forecast or chronology changed")
                        by_rho[str(rho)] = {
                            "relative_log_evidence": [[0., f["relative_log_evidence"], gg["relative_log_evidence"], e[3]] for f, gg, e in zip(fs, gs, previous_e)],
                            "component_log_density": [[p[0], f["adapted"], gg["adapted"], p[3]] for f, gg, p in zip(fs, gs, previous_p)]}
                    rows.append({"seed": seed, "archive_group": group, "held_task": held, "arm": "risk_ard_mixture", "scores_by_rho": by_rho})
                    generic_key = seed, group, held, "fixed_source"
                    task = tasks[held]
                    _, generic_p = profile_arrays(old[generic_key], 32)
                    for t, previous in zip(range(8, 24), generic_p[:, 0]):
                        score = predictive_group_score([full], task["psi"], np.arange(t), task["response"][:t, 2], task["noise"][:t, 2],
                            0., t, task["response"][t:t + 1, 2], task["noise"][t:t + 1, 2])
                        error = abs(score - previous)
                        controls["max_original_i_error"] = max(controls["max_original_i_error"], error)
                        if error > 1e-8:
                            raise ValueError("A20 reconstructed generic I forecast changed")
                    rows.append(old[generic_key])
                    result["folds"].append({"seed": seed, "archive_group": group, "held_task": held,
                        "training_tasks": training_ids, "source_label_count": saved["source_label_count"],
                        "covariance_context_reused": True, "fold_cpu_seconds": time.process_time() - fold_start})
                    if len(result["folds"]) == 1:
                        first = time.process_time() - fold_start
                        projection = time.process_time() - start + first * 255 * 1.25 + 60.
                        result["admission"] = {"retained_first_complete_fold_cpu_seconds": first,
                            "projected_stage_cpu_seconds": projection, "calibration_cpu_reserve": 60.,
                            "safety_multiplier": 1.25, "cap_cpu_seconds": cap, "admitted": projection < cap}
                        print(json.dumps({"admission": result["admission"]}), flush=True)
                        if projection >= cap:
                            raise TimeoutError("A20 complete-cohort projection exceeds CPU cap")
                print(json.dumps({"source_folds_completed": len(result["folds"])}), flush=True)
        for row in rows:
            key = row["seed"], row["archive_group"], row["held_task"], row["arm"]
            e, p = profile_arrays(row, 32)
            forecasts = original_rows[key]["forecasts"]
            for evidence, predictive, forecast in zip(e, p, forecasts):
                ratio = float(np.log1p(-forecast["null_posterior"]) - np.log(forecast["null_posterior"]))
                score = float(prior_adjusted_log_score(predictive[0], predictive[3], forecast["null_posterior"], .5))
                controls["max_saved_r_i_evidence_error"] = max(controls["max_saved_r_i_evidence_error"],
                    abs(evidence[3] - ratio), abs(predictive[0] - forecast["source"]), abs(predictive[3] - forecast["adapted"]), abs(score - forecast["a9"]))
        if controls["max_saved_r_i_evidence_error"] > 1e-8:
            raise ValueError("A20 retained R/I evidence no longer reconstructs A9")
        fits = calibrate(rows, 16, 32)
        result["calibrations"] = [{"arm": arm, **fit} for arm, ff in fits.items() for fit in ff.values()]
        validation, scores = [], {}
        for row in rows:
            key = row["seed"], row["archive_group"], row["held_task"], row["arm"]
            e, p = profile_arrays(row, 32)
            score = predictive_scores(e, p, fits[row["arm"]][row["seed"]]["macro_prior_mass"])
            baseline = np.asarray(validation17[key]["same_pool_ir_scores"])
            scores[key] = score
            validation.append({**{k: row[k] for k in ("seed", "archive_group", "held_task", "arm")},
                "log_scores": score.tolist(), "same_pool_ir_scores": baseline.tolist(),
                "mean_gain": float(np.mean(score - p[:, 0])),
                "mean_gain_vs_same_pool_ir": float(np.mean(score - baseline)),
                "mean_gain_vs_a17": float(np.mean(score - validation17[key]["log_scores"])),
                "pure_f_gain": float(np.mean(p[:, 1] - p[:, 0])), "pure_g_gain": float(np.mean(p[:, 2] - p[:, 0]))})
        result["rows"] = validation
        checks = []
        for nodes, rho in ((24, 32), (32, 32), (16, 16), (16, 48)):
            alternate = calibrate(rows, nodes, rho)
            error = 0.
            for row in rows:
                key = row["seed"], row["archive_group"], row["held_task"], row["arm"]
                e, p = profile_arrays(row, rho)
                changed = predictive_scores(e, p, alternate[row["arm"]][row["seed"]]["macro_prior_mass"])
                error = max(error, float(np.max(np.abs(changed - scores[key]))))
            checks.append({"macro_nodes_per_axis": nodes, "rho_nodes": rho, "max_source_log_score_change": error})
            print(json.dumps(checks[-1]), flush=True)
        old_generic = {r["held_seed"]: r for r in old17["calibrations"] if r["arm"] == "fixed_source"}
        controls["max_generic_same_pool_a12_q_error"] = max(abs(fits["fixed_source"][s]["q_mean"] - old_generic[s]["q_mean"]) for s in seeds)
        controls["max_generic_flat_theta_error"] = max(float(np.max(np.abs(np.array(fits["fixed_source"][s]["theta_mean"]) - 1. / 3.))) for s in seeds)
        passed = max(c["max_source_log_score_change"] for c in checks) <= 1e-4 and max(controls.values()) <= 1e-8
        result.update(controls=controls, numerical_check={"primary_macro_order": 16, "primary_rho_order": 32,
            "checks": checks, "tolerance": 1e-4, "passed": passed, "primary_parameters_changed": False})
        if not passed:
            raise ValueError("A20 numerical or generic reduction check failed")
        indices = np.random.default_rng(spec["paired_bootstrap_seed"]).integers(16, size=(20000, 16))
        metrics = ("mean_gain", "mean_gain_vs_same_pool_ir", "mean_gain_vs_a17", "pure_f_gain", "pure_g_gain")
        summaries = []
        for group in groups:
            for arm in ("fixed_source", "risk_ard_mixture"):
                seed_metrics = {k: [float(np.mean([r[k] for r in validation if (r["seed"], r["archive_group"], r["arm"]) == (s, group, arm)])) for s in seeds] for k in metrics}
                summaries.append({"archive_group": group, "arm": arm,
                    **{k: contrast(values, indices) for k, values in seed_metrics.items()}})
        calls = frozen["source_tasks"] * frozen["source_points"] * frozen["replicates"]
        result.update(complete=True, groups=summaries, source_prior_profiles=len(rows), source_forecasts=8192,
            recovered_eta_contexts_reused=len(result["folds"]), full_eta_prior_refits=len(result["folds"]),
            historical_source_information={"calls_per_archive": calls, "outer_training_archives": 30,
                "outer_training_calls": 30 * calls, "held_archive_calls": calls, "all_32_source_archive_calls": 32 * calls})
    except Exception as exc:
        result["failure"] = str(exc)
        raise
    finally:
        result.update(cpu_seconds=time.process_time() - start, wall_seconds=time.monotonic() - wall,
            current_process_vm_hwm_mb=float(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))) / 1024)
        (ARTIFACTS / "source_forecasts.json.gz").write_bytes(gzip.compress(json.dumps({"rows": rows}, allow_nan=False).encode()))
        (ARTIFACTS / "validation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: result[k] for k in ("complete", "cpu_seconds", "current_process_vm_hwm_mb")}), flush=True)


if __name__ == "__main__":
    main()
